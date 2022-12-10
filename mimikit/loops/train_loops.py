import dataclasses as dtc
import hashlib
from typing import Optional, Tuple

import torch
from pytorch_lightning import LightningModule, Trainer
import os
import h5mapper as h5m

from .logger import LoggingHooks, AudioLogger
from .callbacks import EpochProgressBarCallback, GenerateCallback, MMKCheckpoint
from .samplers import IndicesSampler, TBPTTSampler
from .generate import GenerateLoop
from ..config import Config


__all__ = [
    "TrainARMConfig",
    "TrainLoop"
]


@dtc.dataclass
class TrainARMConfig(Config):
    root_dir: str = './trainings'
    batch_size: int = 16
    batch_length: int = 32
    downsampling: int = 1
    oversampling: int = 1
    shift_error: int = 0
    tbptt_chunk_length: Optional[int] = None
    in_mem_data: bool = True

    max_epochs: int = 2
    limit_train_batches: Optional[int] = None
    max_lr: float = 5e-4
    betas: Tuple[float, float] = (0.9, 0.93)
    div_factor: float = 3.
    final_div_factor: float = 1.
    pct_start: float = 0.
    cycle_momentum: bool = False

    CHECKPOINT_TRAINING: bool = True

    MONITOR_TRAINING: bool = True
    OUTPUT_TRAINING: str = ''

    every_n_epochs: int = 2
    n_examples: int = 3
    prompt_length: int = 32
    n_steps: int = 200
    temperature: Optional[Tuple[float]] = None


class TrainLoop(LoggingHooks,
                LightningModule):

    @classmethod
    def initialize_root_directory(cls,
                                  cfg: TrainARMConfig):
        ID = hashlib.sha256("".encode("utf-8")).hexdigest()
        print("****************************************************")
        print("ID IS :", ID)
        print("****************************************************")
        root_dir = os.path.join(cfg.root_dir, ID)
        os.makedirs(root_dir, exist_ok=True)
        if "mp3" in cfg.OUTPUT_TRAINING:
            output_dir = os.path.join(root_dir, "outputs")
            os.makedirs(output_dir, exist_ok=True)
            filename_template = os.path.join(output_dir, "epoch{epoch}_prm{prompt_idx}.wav")
        else:
            filename_template = ""

        with open(os.path.join(root_dir, "hp.json"), "w") as fp:
            pass
        return root_dir, filename_template

    @classmethod
    def get_dataloader(cls,
                       soundbank,
                       net,
                       input_feature,
                       target_feature,
                       cfg: TrainARMConfig):
        batch = (
            input_feature.batch_item(
                data=torch.from_numpy(soundbank.snd[:]) if cfg.in_mem_data else "snd",
                shift=0, length=cfg.batch_length, downsampling=cfg.downsampling),
            target_feature.batch_item(
                data=torch.from_numpy(soundbank.snd[:]) if cfg.in_mem_data else "snd",
                shift=net.shift, length=net.output_length(cfg.batch_length),
                downsampling=cfg.downsampling)
        )

        if cfg.tbptt_chunk_length is not None:
            if getattr(input_feature, 'domain', '') == 'time-freq' and isinstance(getattr(batch[0], 'getter', False),
                                                                                  h5m.Getter):
                item_len = batch[0].getter.length
                seq_len = item_len
                chunk_length = item_len * cfg.tbptt_chunk_length
                N = len(h5m.ProgrammableDataset(soundbank, batch))
            else:  # feature is in time domain
                seq_len = cfg.batch_length
                chunk_length = cfg.tbptt_chunk_length
                N = soundbank.snd.shape[0]
            loader_kwargs = dict(
                batch_sampler=TBPTTSampler(
                    N,
                    batch_size=cfg.batch_size,
                    chunk_length=chunk_length,
                    seq_len=seq_len,
                    oversampling=cfg.oversampling
                )
            )
        else:
            loader_kwargs = dict(batch_size=cfg.batch_size, shuffle=True)
        n_workers = max(os.cpu_count(), min(cfg.batch_size, os.cpu_count()))
        prefetch = (cfg.batch_size // n_workers) // 2
        return soundbank.serve(batch,
                               num_workers=n_workers,
                               prefetch_factor=max(prefetch, 1),
                               pin_memory=True,
                               persistent_workers=True,
                               **loader_kwargs
                               )

    @classmethod
    def get_optimizer(cls, net, dl, cfg: TrainARMConfig):
        opt = torch.optim.Adam(net.parameters(), lr=cfg.max_lr, betas=cfg.betas)
        sched = torch.OneCycleLR(
            opt,
            steps_per_epoch=min(len(dl), cfg.limit_train_batches) if cfg.limit_train_batches is not None else len(dl),
            epochs=cfg.max_epochs,
            max_lr=cfg.max_lr,
            div_factor=cfg.div_factor,
            final_div_factor=cfg.final_div_factor,
            pct_start=cfg.pct_start,
            cycle_momentum=cfg.cycle_momentum
        )
        return [opt], [{"scheduler": sched, "interval": "step", "frequency": 1}]

    @classmethod
    def get_callbacks(cls,
                      net,
                      soundbank,
                      input_feature,
                      target_feature,
                      root_dir,
                      filename_template,
                      cfg: TrainARMConfig):
        # Gen Loop
        max_i = soundbank.snd.shape[0] - getattr(input_feature, "hop_length", 1) * cfg.prompt_length
        g_dl = soundbank.serve(
            (input_feature.batch_item(shift=0, length=cfg.prompt_length, training=False),),
            sampler=IndicesSampler(N=cfg.n_examples,
                                   max_i=max_i,
                                   redraw=True),
            shuffle=False,
            batch_size=cfg.n_examples
        )
        gen_loop = GenerateLoop(
            network=net,
            dataloader=g_dl,
            inputs=(h5m.Input(None, h5m.AsSlice(dim=1, shift=-net.rf, length=net.rf),
                              setter=h5m.Setter(dim=1)),
                    *((h5m.Input(cfg.temperature, h5m.AsSlice(dim=1 + int(hasattr(net.hp, 'hop')), length=1),
                                 setter=None),)
                      if cfg.temperature is not None else ())),
            n_steps=cfg.n_steps,
            device='cuda' if torch.cuda.is_available() else 'cpu',
            time_hop=net.hp.get("hop", 1)
        )

        callbacks = []

        if cfg.CHECKPOINT_TRAINING:
            callbacks += [
                MMKCheckpoint(epochs=cfg.every_n_epochs, root_dir=root_dir)
            ]

        if cfg.MONITOR_TRAINING or cfg.OUTPUT_TRAINING:
            callbacks += [
                GenerateCallback(
                    generate_loop=gen_loop,
                    every_n_epochs=cfg.every_n_epochs,
                    output_features=[target_feature, ],
                    audio_logger=AudioLogger(
                        sr=target_feature.sr,
                        hop_length=getattr(target_feature, 'hop_length', 512),
                        **(dict(filename_template=filename_template,
                                target_dir=os.path.dirname(filename_template))
                           if 'mp3' in cfg.OUTPUT_TRAINING else {}),
                    ),
                )]

        gen_loop.plot_audios = gen_loop.play_audios = cfg.MONITOR_TRAINING

        return callbacks

    def __init__(self,
                 model_config, loader, net, loss_fn, optim,
                 # number of batches before reset_hidden is called
                 tbptt_len=None,
                 ):
        super().__init__()
        self.model_config = model_config
        self.loader = loader
        self.net = net
        self.loss_fn = loss_fn
        self.optim = optim
        self.tbptt_len = tbptt_len

    def forward(self, inputs):
        if not isinstance(inputs, (tuple, list)):
            inputs = inputs,
        return self.net(*inputs)

    def configure_optimizers(self):
        return self.optim

    def train_dataloader(self):
        return self.loader

    def on_train_batch_start(self, batch, batch_idx, dataloader_idx):
        if self.tbptt_len is not None and (batch_idx % self.tbptt_len) == 0:
            self.net.reset_hidden()

    def training_step(self, batch, batch_idx):
        batch, target = batch
        output = self.forward(batch)
        return self.loss_fn(output, target)

    def run(self, root_dir, max_epochs, callbacks, limit_train_batches):
        self.trainer = Trainer(
            default_root_dir=root_dir,
            max_epochs=max_epochs,
            limit_train_batches=limit_train_batches,
            callbacks=[EpochProgressBarCallback()].extend(callbacks),
            progress_bar_refresh_rate=10,
            process_position=1,
            logger=None,
            checkpoint_callback=False,
            num_sanity_val_steps=0,
            gpus=torch.cuda.device_count() if torch.cuda.is_available() else 0,
        )
        self.trainer.fit(self)
        return self
