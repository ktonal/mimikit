import torch
from pytorch_lightning import LightningModule, Trainer

from .logger import LoggingHooks
from .callbacks import EpochProgressBarCallback


__all__ = [
    "TrainLoop"
]


class TrainLoop(LoggingHooks,
                LightningModule):

    def __init__(self,
                 loader, net, loss_fn, optim,
                 # number of batches before reset_hidden is called
                 tbptt_len=None,
                 ):
        super().__init__()
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
