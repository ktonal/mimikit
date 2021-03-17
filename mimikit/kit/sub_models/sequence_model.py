import torch
from pytorch_lightning import LightningModule
import numpy as np
from abc import ABC

from .utils import MMKHooks, LoggingHooks, tqdm


class SequenceModel(MMKHooks,
                    LoggingHooks,
                    LightningModule,
                    ABC):

    loss_fn = None
    db_class = None

    def __init__(self):
        super(LightningModule, self).__init__()
        MMKHooks.__init__(self)
        LoggingHooks.__init__(self)

    def training_step(self, batch, batch_idx):
        batch, target = batch
        output = self.forward(batch)
        L = self.loss_fn(output, target)
        return {"loss": L}

    def validation_step(self, batch, batch_idx):
        batch, target = batch
        output = self.forward(batch)
        L = self.loss_fn(output, target)
        return {"val_loss": L}

    def setup(self, stage: str):
        if stage == "fit" and getattr(self, "logger", None) is not None:
            self.logger.log_hyperparams(self.hparams)

    def batch_info(self, *args, **kwargs):
        raise NotImplementedError("subclasses of `SequenceModel` have to implement `batch_info`")

    def generate(self, prompt, n_steps, decode_outputs=False, **kwargs):
        raise NotImplementedError

    def encode_inputs(self, inputs: torch.Tensor):
        raise NotImplementedError

    def decode_outputs(self, outputs: torch.Tensor):
        raise NotImplementedError
