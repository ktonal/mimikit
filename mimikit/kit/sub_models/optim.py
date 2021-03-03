import torch
from pytorch_lightning import LightningModule


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
            super(ManyOneCycleLR, self).step(epoch=epoch)


class SuperAdam(LightningModule):
    """
    Here the method ``configure_optimizers()`` returns an ``Adam`` optimizer and a slightly modified
    ``OneCycleLR`` scheduler (the only modification being that it won't raise a ``ValueError`` if you use for more
    steps than what it expects and will instead restarts its cycle)
    """

    def __init__(self,
                 max_lr=1e-3,
                 betas=(.9, .9),
                 div_factor=3.,
                 final_div_factor=1.,
                 pct_start=.25,
                 cycle_momentum=False):
        super(LightningModule, self).__init__()
        self.max_lr = max_lr
        self.betas = betas
        self.div_factor = div_factor
        self.final_div_factor = final_div_factor
        self.pct_start = pct_start
        self.cycle_momentum = cycle_momentum
        # those are set in setup when we know what the trainer and datamodule got
        self.steps_per_epoch = None
        self.max_epochs = None

    def configure_optimizers(self):
        self.opt = torch.optim.Adam(self.parameters(), lr=self.max_lr, betas=self.betas)
        self.sched = ManyOneCycleLR(self.opt,
                                    steps_per_epoch=self.steps_per_epoch,
                                    epochs=self.max_epochs,
                                    max_lr=self.max_lr, div_factor=self.div_factor,
                                    final_div_factor=self.final_div_factor,
                                    pct_start=self.pct_start,
                                    cycle_momentum=self.cycle_momentum
                                    )
        return [self.opt], [{"scheduler": self.sched, "interval": "step", "frequency": 1}]

    def setup(self, stage: str):
        if stage == "fit":
            self.max_epochs = self.trainer.max_epochs
            self.steps_per_epoch = len(self.datamodule.train_ds)
