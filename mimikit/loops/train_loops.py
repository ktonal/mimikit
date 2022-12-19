import dataclasses as dtc
import hashlib
from typing import Optional, Tuple

import torch
from pytorch_lightning import LightningModule, Trainer, callbacks as pl_callbacks
import os
import h5mapper as h5m

from .logger import LoggingHooks, AudioLogger
from .callbacks import EpochProgressBarCallback, GenerateCallback, MMKCheckpoint
from .samplers import IndicesSampler, TBPTTSampler
from .generate import GenerateLoop
from ..networks.arm import ARM
from ..config import Config


__all__ = [
    "TrainARMConfig",
    "TrainLoop"
]


# @dtc.dataclass
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
                                  network_config: Config,
                                  cfg: TrainARMConfig) -> Tuple[str, str]:
        ID = hashlib.sha256("".encode("utf-8")).hexdigest()
        print("****************************************************")
        print("ID IS :", ID)
        print("****************************************************")
        root_dir = os.path.join(cfg.root_dir, ID)
        os.makedirs(root_dir, exist_ok=True)
        if cfg.OUTPUT_TRAINING:
            output_dir = os.path.join(root_dir, "outputs")
            os.makedirs(output_dir, exist_ok=True)
            filename_template = os.path.join(output_dir, "epoch{epoch}_prm{prompt_idx}.mp3")
        else:
            filename_template = ""

        with open(os.path.join(root_dir, "hp.yaml"), "w") as fp:
            pass
        return root_dir, filename_template

    @classmethod
    def get_dataloader(cls,
                       soundbank,
                       net: ARM,
                       input_feature,
                       target_feature,
                       cfg: TrainARMConfig):

        batch = net.train_batch(cfg.batch_length, "step", cfg.downsampling)

        if cfg.tbptt_chunk_length is not None:
            # feature MUST be in time domain
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
        sched = torch.optim.lr_scheduler.OneCycleLR(
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
            (input_feature.batch_item(length=cfg.prompt_length, unit="step"),),
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
                    *((h5m.Input(torch.tensor(cfg.temperature).view(-1, 1).expand(cfg.n_examples, cfg.n_steps),
                                 h5m.AsSlice(dim=1, length=1),
                                 setter=None),)
                      if cfg.temperature is not None else ())),
            n_steps=cfg.n_steps,
            device='cuda' if torch.cuda.is_available() else 'cpu',
            time_hop=1
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
                        title_template="epoch={epoch}" if cfg.MONITOR_TRAINING else None,
                        file_template=filename_template if cfg.OUTPUT_TRAINING else None
                    ),
                )]

        gen_loop.plot_audios = gen_loop.play_audios = cfg.MONITOR_TRAINING

        return callbacks

    @classmethod
    def from_config(cls, soundbank, network: ARM, config: TrainARMConfig):
        inputs, targets = network.train_batch(config.batch_length)
        dataloader = cls.get_dataloader(
            soundbank, network, inputs[0], targets[0], config)
        optim = cls.get_optimizer(network, dataloader, config)
        return cls(config, soundbank, dataloader, network,
                   network.config.io_spec.targets[0].loss_fn, optim)

    @property
    def config(self):
        return self.train_config

    def __init__(self,
                 train_config: TrainARMConfig,
                 soundbank: h5m.SoundBank,
                 loader: torch.utils.data.DataLoader,
                 net: ARM,
                 loss_fn,
                 optim,
                 ):
        super().__init__()
        self.train_config = train_config
        self.soundbank = soundbank
        self.loader = loader
        self.net = net
        self.loss_fn = loss_fn
        self.optim = optim
        self.tbptt_len = self.train_config.tbptt_chunk_length
        if self.tbptt_len is not None:
            self.tbptt_len //= self.config.batch_length

    def forward(self, inputs):
        if not isinstance(inputs, (tuple, list)):
            inputs = inputs,
        return self.net(*inputs)

    def configure_optimizers(self):
        return self.optim

    def train_dataloader(self):
        return self.loader

    def on_train_batch_start(self, batch, batch_idx):
        if self.tbptt_len is not None and (batch_idx % self.tbptt_len) == 0:
            self.net.reset_hidden()

    def training_step(self, batch, batch_idx):
        batch, target = batch
        output = self.forward(batch)
        # print(batch[0].size(), output.size(), target[0].size())
        return self.loss_fn(output, target[0])

    def run(self):
        root_dir, output_template = self.initialize_root_directory(self.net.config, self.train_config)
        callbacks = self.get_callbacks(
            self.net, self.soundbank,
            self.net.config.io_spec.inputs[0].feature,
            self.net.config.io_spec.targets[0].feature,
            root_dir, output_template,
            self.train_config
        )
        self.trainer = Trainer(
            default_root_dir=root_dir,
            max_epochs=self.train_config.max_epochs,
            limit_train_batches=self.train_config.limit_train_batches,
            callbacks=[
                EpochProgressBarCallback(),
                pl_callbacks.TQDMProgressBar(refresh_rate=20, process_position=1),
                *callbacks],
            logger=None,
            enable_checkpointing=False,
            num_sanity_val_steps=0,
            gpus=torch.cuda.device_count() if torch.cuda.is_available() else 0,
        )
        self.trainer.fit(self)
        return self
