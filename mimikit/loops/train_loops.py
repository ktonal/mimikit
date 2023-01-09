import dataclasses as dtc
import hashlib
from typing import Optional, Tuple

import torch
from pytorch_lightning import LightningModule, Trainer, callbacks as pl_callbacks
import os
import h5mapper as h5m

from .logger import LoggingHooks
from .callbacks import EpochProgressBarCallback, GenerateCallback, MMKCheckpoint
from .samplers import TBPTTSampler
from .generate import GenerateLoopV2
from ..features.dataset import DatasetConfig
from ..features.item_spec import ItemSpec, Step
from ..networks.arm import ARM
from ..config import Config, NetworkConfig

__all__ = [
    "TrainARMConfig",
    "ARMHP",
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
    prompt_length_sec: float = .5
    outputs_duration_sec: float = 1.
    temperature: Optional[Tuple[float]] = None


@dtc.dataclass
class ARMHP(Config):
    data: DatasetConfig
    network: NetworkConfig
    training: TrainARMConfig


# noinspection PyCallByClass
class TrainLoop(LoggingHooks,
                LightningModule):

    @classmethod
    def get_os_paths(cls,
                     cfg: ARMHP
                     ) -> Tuple[str, str, str]:
        yaml = cfg.serialize()
        hash_ = hashlib.sha256(yaml.encode("utf-8")).hexdigest()[:8]
        root_dir = os.path.join(cfg.training.root_dir, hash_)
        output_dir = os.path.join(root_dir, "outputs")
        filename_template = os.path.join(output_dir, "epoch{epoch}_prm{prompt_idx}.mp3")
        return root_dir, hash_, filename_template

    @classmethod
    def get_dataloader(cls,
                       soundbank,
                       net: ARM,
                       cfg: TrainARMConfig):
        user_spec = ItemSpec(shift=0, length=cfg.batch_length,
                             stride=cfg.downsampling, unit=Step())
        batch = net.train_batch(user_spec)

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
        with_cuda = torch.cuda.is_available()
        return soundbank.serve(batch,
                               num_workers=n_workers,
                               prefetch_factor=2,
                               pin_memory=with_cuda,
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
                      root_dir,
                      filename_template,
                      cfg: TrainARMConfig):
        # Gen Loop
        gen_loop = GenerateLoopV2.from_config(
            GenerateLoopV2.Config(
                output_duration_sec=cfg.outputs_duration_sec,
                prompts_length_sec=cfg.prompt_length_sec,
                prompts_position_sec=(None,)*cfg.n_examples,
                parameters=dict(temperature=cfg.temperature),
                batch_size=cfg.n_examples,
                output_name_template=filename_template,
                display_waveform=cfg.MONITOR_TRAINING,
                write_waveform=cfg.OUTPUT_TRAINING
            ),
            network=net, soundbank=soundbank
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
                )]

        gen_loop.plot_audios = gen_loop.play_audios = cfg.MONITOR_TRAINING

        return callbacks

    @classmethod
    def from_config(cls, train_cfg: TrainARMConfig, soundbank, network: ARM):
        dataloader = cls.get_dataloader(soundbank, network, train_cfg)
        ds_cfg = DatasetConfig(
            destination=soundbank.filename,
            sources=tuple(soundbank.index.keys())
        )
        hp = ARMHP(training=train_cfg, network=network.config, data=ds_cfg)
        return cls(hp, soundbank, dataloader, network,
                   network.config.io_spec.loss_fn)

    @property
    def config(self):
        return self._config

    def __init__(self,
                 hp: ARMHP,
                 soundbank: h5m.TypedFile,
                 loader: torch.utils.data.DataLoader,
                 net: ARM,
                 loss_fn,
                 ):
        super().__init__()
        self._config = hp
        self.train_cfg = hp.training
        self.root_dir, self.hash_, self.output_template = self.get_os_paths(hp)
        self.soundbank = soundbank
        self.loader = loader
        self.net = net
        self.loss_fn = loss_fn
        self.tbptt_len = self.train_cfg.tbptt_chunk_length
        if self.tbptt_len is not None:
            self.tbptt_len //= self.train_cfg.batch_length
        self.callbacks = self.get_callbacks(
            self.net, self.soundbank, self.root_dir, self.output_template,
            self.train_cfg
        )

    def configure_optimizers(self):
        return self.get_optimizer(self.net, self.loader, self.config.training)

    def train_dataloader(self):
        return self.loader

    def on_train_batch_start(self, batch, batch_idx):
        if self.tbptt_len is not None and (batch_idx % self.tbptt_len) == 0:
            self.net.reset_hidden()

    def training_step(self, batch, batch_idx):
        batch, target = batch
        output = self.net.forward(batch)
        if not isinstance(output, tuple):
            output = output,
        return self.loss_fn(output, target)

    def run(self):
        os.makedirs(self.root_dir, exist_ok=True)
        os.makedirs(os.path.join(self.root_dir, "outputs"), exist_ok=True)
        self.save_hp()
        print("*" * 64)
        print("training's id is:", self.hash_)
        print("*" * 64)

        self.trainer = Trainer(
            default_root_dir=self.root_dir,
            max_epochs=self.train_cfg.max_epochs,
            limit_train_batches=self.train_cfg.limit_train_batches,
            callbacks=[
                EpochProgressBarCallback(),
                pl_callbacks.TQDMProgressBar(refresh_rate=20, process_position=1),
                *self.callbacks],
            logger=None,
            enable_checkpointing=False,
            num_sanity_val_steps=0,
            gpus=torch.cuda.device_count() if torch.cuda.is_available() else 0,
        )
        self.trainer.fit(self)
        try:
            self.loader._iterator._shutdown_workers()
        except:
            pass
        self.soundbank.close()
        return self

    def save_hp(self):
        with open(os.path.join(self.root_dir, "hp.yaml"), "w") as fp:
            fp.write(self.config.serialize())
