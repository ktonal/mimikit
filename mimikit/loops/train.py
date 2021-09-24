from pytorch_lightning import LightningModule

from mimikit.loops.logger import LoggingHooks
from .get_trainer import get_trainer


class TrainLoop(LoggingHooks,
                LightningModule):

    def __init__(self, loader, net, loss_fn, optim):
        super().__init__()
        self.loader = loader
        self.net = net
        self.loss_fn = loss_fn
        self.optim = optim

    def forward(self, inputs):
        return self.net(inputs)

    def configure_optimizers(self):
        return self.optim

    def train_dataloader(self):
        return self.loader

    def training_step(self, batch, batch_idx):
        batch, target = batch
        output = self.forward(batch)
        return self.loss_fn(output, target)

    def run(self, **trainer_kwargs):
        self.trainer = get_trainer(**trainer_kwargs)
        self.trainer.fit(self)
        return self
