import dataclasses as dtc
import hashlib
from typing import Optional, Tuple, Dict

import torch
from pytorch_lightning import LightningModule, Trainer
import os

from torch.optim import Optimizer, Adam

import h5mapper as h5m

from .logger import LoggingHooks
from .callbacks import EpochProgressBarCallback, GenerateCallback, MMKCheckpoint, TrainingProgressBar, is_notebook
from .samplers import TBPTTSampler
from .generate import GenerateLoopV2
from ..utils import default_device
from ..features.dataset import DatasetConfig
from ..features.item_spec import ItemSpec
from ..networks.arm import ARM, NetworkConfig
from ..config import Config

__all__ = [
    "TrainARMConfig",
    "ARMHP",
    "TrainARMLoop"
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
    temperature: Optional[Tuple[float, ...]] = None
    trainer_kwargs: Dict = dtc.field(default_factory=dict)


# implements TrainingConfig structurally
@dtc.dataclass
class ARMHP(Config):
    dataset: DatasetConfig
    network: NetworkConfig
    training: TrainARMConfig


# noinspection PyCallByClass
class TrainARMLoop(LoggingHooks,
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
                       dataset: h5m.TypedFile,
                       net: ARM,
                       cfg: TrainARMConfig):
        user_spec = ItemSpec(shift=0, length=cfg.batch_length,
                             stride=cfg.downsampling, unit=net.config.io_spec.unit)
        batch = net.train_batch(user_spec)

        if cfg.tbptt_chunk_length is not None:
            # feature MUST be in time domain
            seq_len = cfg.batch_length
            chunk_length = cfg.tbptt_chunk_length
            # TODO: get rid of '.signal' assumption
            N = dataset.signal.shape[0]
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
        return dataset.serve(batch,
                             num_workers=n_workers,
                             prefetch_factor=2,
                             pin_memory=with_cuda,
                             persistent_workers=True,
                             **loader_kwargs
                             )

    @classmethod
    def get_optimizer(cls, net, dl, cfg: TrainARMConfig):
        opt = Adam(net.parameters(), lr=cfg.max_lr, betas=cfg.betas)
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
                      dataset,
                      root_dir,
                      filename_template,
                      cfg: TrainARMConfig):
        # Gen Loop
        gen_loop = GenerateLoopV2.from_config(
            GenerateLoopV2.Config(
                output_duration_sec=cfg.outputs_duration_sec,
                prompts_length_sec=cfg.prompt_length_sec,
                prompts_position_sec=(None,) * cfg.n_examples,
                parameters=dict(temperature=cfg.temperature),
                batch_size=cfg.n_examples,
                output_name_template=filename_template,
                display_waveform=cfg.MONITOR_TRAINING,
                write_waveform=cfg.OUTPUT_TRAINING
            ),
            network=net, dataset=dataset
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
    def from_config(cls, train_cfg: TrainARMConfig, dataset, network: ARM):
        dataloader = cls.get_dataloader(dataset, network, train_cfg)
        if not hasattr(dataset, "config"):
            ds_cfg = DatasetConfig(
                filename=dataset.filename,
                sources=tuple(dataset.index.keys())
            )
        else:
            ds_cfg = dataset.config
        hp = ARMHP(training=train_cfg, network=network.config, dataset=ds_cfg)
        return cls(hp, dataset, dataloader, network,
                   network.config.io_spec.loss_fn)

    @property
    def config(self) -> ARMHP:
        return self._config

    def __init__(self,
                 hp: ARMHP,
                 dataset: h5m.TypedFile,
                 loader: torch.utils.data.DataLoader,
                 net: ARM,
                 loss_fn,
                 ):
        super().__init__()
        self._config = hp
        self.train_cfg = hp.training
        self.root_dir, self.hash_, self.output_template = self.get_os_paths(hp)
        self.dataset = dataset
        self.loader = loader
        self.net = net
        self.loss_fn = loss_fn
        self.tbptt_len = self.train_cfg.tbptt_chunk_length
        if self.tbptt_len is not None:
            self.tbptt_len //= self.train_cfg.batch_length
        self.callbacks = self.get_callbacks(
            self.net, self.dataset, self.root_dir, self.output_template,
            self.train_cfg
        )
        self.opt = None

    def configure_optimizers(self):
        self.opt = self.get_optimizer(self.net, self.loader, self.config.training)
        return self.opt

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

    def on_train_epoch_end(self, *args):
        super(TrainARMLoop, self).on_train_epoch_end(*args)

    def run(self):
        os.makedirs(self.root_dir, exist_ok=True)
        os.makedirs(os.path.join(self.root_dir, "outputs"), exist_ok=True)
        self.save_hp()
        print("*" * 64)
        print("training's id is:", self.hash_)
        print("*" * 64)
        epochs_bar = [EpochProgressBarCallback()] if is_notebook() else ()
        self.trainer = Trainer(
            default_root_dir=self.root_dir,
            max_epochs=self.train_cfg.max_epochs,
            limit_train_batches=self.train_cfg.limit_train_batches,
            callbacks=[
                *epochs_bar,
                TrainingProgressBar(),
                *self.callbacks],
            logger=None,
            enable_checkpointing=False,
            num_sanity_val_steps=0,
            accelerator=default_device(),
            gpus=torch.cuda.device_count() if torch.cuda.is_available() else 0,
            **self.config.training.trainer_kwargs
        )
        self.trainer.fit(self)
        try:
            self.loader._iterator._shutdown_workers()
        except:
            pass
        self.dataset.close()
        return self

    def save_hp(self):
        with open(os.path.join(self.root_dir, "hp.yaml"), "w") as fp:
            fp.write(self.config.serialize())
