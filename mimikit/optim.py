import torch
from pytorch_lightning import LightningModule
import dataclasses as dtc
from typing import Tuple, Optional

__all__ = [
    'Adam',
    'SuperAdam'
]


class Adam(LightningModule):
    def __init__(self,
                 max_lr=1e-3,
                 betas=(.9, .9)
                 ):
        super(LightningModule, self).__init__()
        self.max_lr = max_lr
        self.betas = betas

    def configure_optimizers(self):
        self.opt = torch.optim.Adam(self.parameters(), lr=self.max_lr, betas=self.betas)
        return self.opt


class ManyOneCycleLR(torch.optim.lr_scheduler.OneCycleLR):
    def step(self, epoch=None):
        try:
            super(ManyOneCycleLR, self).step(epoch=epoch)
        except ValueError:
            self.last_epoch = 0
            self._step_count = 1
            super(ManyOneCycleLR, self).step(epoch=epoch)


@dtc.dataclass(repr=False, eq=False)
class SuperAdam(LightningModule):

    max_lr: float = 1e-3
    betas: Tuple[float, float] = (.9, .9)
    div_factor: float = 3.
    final_div_factor: float = 1.
    pct_start: float = 0.
    cycle_momentum: bool = False
    total_steps: Optional[int] = None

    def __post_init__(self):
        super(LightningModule, self).__init__()
        # those are set in setup when we know what the trainer and datamodule got
        self.steps_per_epoch = None
        self.max_epochs = None
        self.opt, self.sched = None, None

    def configure_optimizers(self):
        self.opt = torch.optim.Adam(self.parameters(), lr=self.max_lr, betas=self.betas)
        steps = dict(total_steps=self.total_steps) if self.total_steps is not None \
            else dict(steps_per_epoch=self.steps_per_epoch, epochs=self.max_epochs)
        self.sched = ManyOneCycleLR(self.opt,
                                    **steps,
                                    max_lr=self.max_lr, div_factor=self.div_factor,
                                    final_div_factor=self.final_div_factor,
                                    pct_start=self.pct_start,
                                    cycle_momentum=self.cycle_momentum
                                    )
        return [self.opt], [{"scheduler": self.sched, "interval": "step", "frequency": 1}]

    def setup(self, stage: str):
        if stage == "fit":
            self.max_epochs = self.trainer.max_epochs
            self.steps_per_epoch = len(self.train_dataloader())


class RMS(LightningModule):
    def __init__(self,
                 lr=1e-2,
                 alpha=0.99,
                 eps=1e-8,
                 weight_decay=0,
                 momentum=0,
                 centered=False
                 ):
        super(LightningModule, self).__init__()
        self.args = dict(lr=lr, alpha=alpha, eps=eps, weight_decay=weight_decay, momentum=momentum, centered=centered)

    def configure_optimizers(self):
        self.opt = torch.optim.RMSprop(self.parameters(), **self.args)
        return self.opt