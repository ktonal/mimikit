import dataclasses as dtc
import hashlib
from typing import Optional, Tuple, Dict

import torch
from pytorch_lightning import LightningModule, Trainer
import os

from torch.optim import Adam, AdamW

import h5mapper as h5m
from .beta_scheduler import BetaScheduler

from .logger import LoggingHooks
from .callbacks import EpochProgressBarCallback, GenerateCallback, MMKCheckpoint, TrainingProgressBar, is_notebook
from .samplers import TBPTTSampler
from .generate import GenerateLoopV2, EncodeDecodeLoop
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
    sampling_jitter: int = 0
    shift_error: int = 0
    tbptt_chunk_length: Optional[int] = None

    max_epochs: int = 2
    limit_train_batches: Optional[int] = None
    max_lr: float = 5e-4
    betas: Tuple[float, float] = (0.9, 0.93)
    weight_decay: float = 0.
    div_factor: float = 3.
    final_div_factor: float = 1.
    pct_start: float = 0.
    max_beta: Optional[float] = None
    beta_div_factor: float = 3.
    beta_final_div_factor: float = 1.
    beta_pct_start: float = 0.
    cycle_momentum: bool = False

    CHECKPOINT_TRAINING: bool = True

    MONITOR_TRAINING: bool = True
    OUTPUT_TRAINING: str = ''

    save_optimizer: bool = False
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
        n_workers = min(cfg.batch_size, os.cpu_count())
        with_cuda = torch.cuda.is_available()
        return dataset.serve(batch,
                             sampling_jitter=cfg.sampling_jitter,
                             num_workers=n_workers,
                             prefetch_factor=1,
                             pin_memory=with_cuda,
                             persistent_workers=True,
                             **loader_kwargs
                             )

    @classmethod
    def get_lr_scheduler(cls, net, opt, dl, cfg: TrainARMConfig):
        steps_per_epoch = min(len(dl), cfg.limit_train_batches) if cfg.limit_train_batches is not None else len(dl)
        sched = torch.optim.lr_scheduler.OneCycleLR(
            opt,
            steps_per_epoch=steps_per_epoch,
            epochs=cfg.max_epochs,
            max_lr=cfg.max_lr,
            div_factor=cfg.div_factor,
            final_div_factor=cfg.final_div_factor,
            pct_start=cfg.pct_start,
            cycle_momentum=cfg.cycle_momentum
        )

        return {"scheduler": sched, "interval": "step", "frequency": 1}

    @classmethod
    def get_beta_scheduler(cls, net, opt, dl, cfg: TrainARMConfig):
        steps_per_epoch = min(len(dl), cfg.limit_train_batches) if cfg.limit_train_batches is not None else len(dl)
        beta_sched = BetaScheduler(
            opt,
            steps_per_epoch=steps_per_epoch,
            epochs=cfg.max_epochs,
            max_beta=cfg.max_beta,
            div_factor=cfg.beta_div_factor,
            final_div_factor=cfg.beta_final_div_factor,
            pct_start=cfg.beta_pct_start,
        )
        return {"scheduler": beta_sched, "interval": "step", "frequency": 1}

    @classmethod
    def get_optimizer(cls, net, dl, cfg: TrainARMConfig):
        opt = AdamW(net.parameters(), lr=cfg.max_lr, betas=cfg.betas, weight_decay=cfg.weight_decay)
        sched = [cls.get_lr_scheduler(net, opt, dl, cfg)]
        if cfg.max_beta is not None:
            sched += [cls.get_beta_scheduler(net, opt, dl, cfg)]
        return [opt], sched

    def reset_lr_scheduler(self, max_lr=None, div_factor=None, final_div_factor=None, pct_start=None):
        cfg = self.train_cfg
        dl = self.loader
        sched = torch.optim.lr_scheduler.OneCycleLR(
            self.opt[0][0],
            steps_per_epoch=min(len(dl), cfg.limit_train_batches) if cfg.limit_train_batches is not None else len(dl),
            epochs=cfg.max_epochs,
            max_lr=max_lr or cfg.max_lr,
            div_factor=div_factor or cfg.div_factor,
            final_div_factor=final_div_factor or cfg.final_div_factor,
            pct_start=pct_start or cfg.pct_start,
            cycle_momentum=cfg.cycle_momentum,
            # base_momentum=0.5,
            # max_momentum=0.05
        )
        sched = [{"scheduler": sched, "interval": "step", "frequency": 1}]
        self.opt = self.opt[0], sched
        return self

    @classmethod
    def get_callbacks(cls,
                      net,
                      dataset,
                      root_dir,
                      filename_template,
                      cfg: TrainARMConfig):

        callbacks = []

        if cfg.CHECKPOINT_TRAINING:
            callbacks += [
                MMKCheckpoint(epochs=cfg.every_n_epochs, root_dir=root_dir)
            ]
        if cfg.MONITOR_TRAINING or cfg.OUTPUT_TRAINING:
            if isinstance(net, ARM):
                # Gen Loop
                gen_loop = GenerateLoopV2.from_config(
                    GenerateLoopV2.Config(
                        output_duration_sec=cfg.outputs_duration_sec,
                        prompts_length_sec=cfg.prompt_length_sec,
                        prompts_position_sec=(None,) * cfg.n_examples,
                        parameters=dict(temperature=cfg.temperature),
                        batch_size=cfg.n_examples,
                        downsampling=cfg.downsampling,
                        output_name_template=filename_template,
                        display_waveform=cfg.MONITOR_TRAINING,
                        write_waveform=cfg.OUTPUT_TRAINING
                    ),
                    network=net, dataset=dataset
                )
            else:  # AutoEncoder Loop
                gen_loop = EncodeDecodeLoop.from_config(
                    EncodeDecodeLoop.Config(
                        prompts_length_sec=max(cfg.prompt_length_sec, cfg.outputs_duration_sec),
                        prompts_position_sec=(None,) * cfg.n_examples,
                        parameters=dict(temperature=cfg.temperature),
                        batch_size=cfg.n_examples,
                        downsampling=cfg.downsampling,
                        output_name_template=filename_template,
                        display_waveform=cfg.MONITOR_TRAINING,
                        write_waveform=cfg.OUTPUT_TRAINING
                    )
                )
            callbacks += [
                GenerateCallback(
                    generate_loop=gen_loop,
                    every_n_epochs=cfg.every_n_epochs,
                )]

            gen_loop.plot_audios = gen_loop.play_audios = cfg.MONITOR_TRAINING

        return callbacks

    @classmethod
    def from_config(cls, train_cfg: TrainARMConfig, dataset, network: ARM, opt=None):
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
                   network.config.io_spec.loss_fn, opt)

    @classmethod
    def from_checkpoint(cls, checkpoint: "Checkpoint"):
        dataset, network = checkpoint.dataset, checkpoint.network
        train_cfg = checkpoint.training_config
        optimizer_state = checkpoint.optimizer_state
        dataloader = cls.get_dataloader(dataset, network, train_cfg)
        opt = cls.get_optimizer(network, dataloader, train_cfg)
        if optimizer_state is not None:
            opt[0][0].load_state_dict(optimizer_state)
        loop = cls(ARMHP(training=train_cfg, network=network.config, dataset=dataset.config),
                   dataset, dataloader, network,
                   network.config.io_spec.loss_fn, opt)
        loop.trainer_state = checkpoint.trainer_state
        return loop

    @property
    def config(self) -> ARMHP:
        return self._config

    def __init__(self,
                 hp: ARMHP,
                 dataset: h5m.TypedFile,
                 loader: torch.utils.data.DataLoader,
                 net: ARM,
                 loss_fn,
                 opt=None,
                 ):
        super().__init__()
        self._config = hp
        self.train_cfg = hp.training
        self.root_dir, self.hash_, self.output_template = self.get_os_paths(hp)
        self.dataset = dataset
        self.loader = loader
        self.loss_fn = loss_fn
        self.tbptt_len = self.train_cfg.tbptt_chunk_length
        if self.tbptt_len is not None:
            self.tbptt_len //= self.train_cfg.batch_length
        self.net = net
        self.callbacks = self.get_callbacks(
            self.net, self.dataset, self.root_dir, self.output_template,
            self.train_cfg
        )
        self.opt = opt
        self.trainer_state = None

    def configure_optimizers(self):
        if self.opt is None:
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

    def training_step_(self, batch, batch_idx):
        batch, target = batch
        L = {"loss": 0}
        for n in range(2):
            output = self.net.forward(batch)
            if not isinstance(output, tuple):
                output = output,
            target = (target[0][:, -output[0].size(1):], )
            L["loss"] += self.loss_fn(output, target)["loss"] * (1/(n+1))
            batch = (output[0].detach(),)
        return L

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
            logger=False,
            enable_checkpointing=False,
            num_sanity_val_steps=0,
            accelerator=default_device(),
            devices=torch.cuda.device_count() if torch.cuda.is_available() else 1,
            **self.config.training.trainer_kwargs
        )
        if self.trainer_state is not None:
            self.trainer.fit_loop.load_state_dict(self.trainer_state['fit_loop'])
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
