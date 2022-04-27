from pytorch_lightning import LightningModule

from mimikit.loops.logger import LoggingHooks
from .get_trainer import get_trainer

__all__ = [
    "TrainLoop"
]


class TrainLoop(LoggingHooks,
                LightningModule):

    def __init__(self,
                 loader, net, loss_fn, optim,
                 # number of batches before reset_hidden is called
                 tbptt_len=None,
                 reset_optim=False,
                 ):
        super().__init__()
        self.loader = loader
        self.net = net
        self.loss_fn = loss_fn
        self.optim = optim
        self.tbptt_len = tbptt_len
        self.reset_optim = reset_optim

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

    def on_epoch_end(self) -> None:
        super(TrainLoop, self).on_epoch_end()
        if self.reset_optim:
            opt = self.optim[0][0]
            for p_group in opt.param_groups:
                for p in p_group["params"]:
                    if not opt.state[p]:
                        print("No states:", p.shape)
                        continue
                    opt.state[p]["exp_avg"].zero_()
                    opt.state[p]["exp_avg_sq"].zero_()
                    opt.state[p]["step"] = 10000

    def run(self, **trainer_kwargs):
        self.trainer = get_trainer(**trainer_kwargs)
        self.trainer.fit(self)
        return self
