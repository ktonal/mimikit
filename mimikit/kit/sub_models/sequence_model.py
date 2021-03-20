import torch
from pytorch_lightning import LightningModule
from abc import ABC

from .utils import MMKHooks, LoggingHooks, tqdm


class SequenceModel(MMKHooks,
                    LoggingHooks,
                    LightningModule,
                    ABC):

    loss_fn = None

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

    def before_generate(self, *args, **kwargs):
        # prepare model
        self._was_training = self.training
        self._initial_device = self.device
        self.eval()
        self.to("cuda" if torch.cuda.is_available() else "cpu")
        torch.set_grad_enabled(False)

    def after_generate(self, *args, **kwargs):
        # reset model
        self.to(self._initial_device)
        self.train() if self._was_training else None
        torch.set_grad_enabled(True)

    def prepare_prompt(self, prompt, n_steps, at_least_nd=2):
        if not isinstance(prompt, torch.Tensor):
            prompt = torch.from_numpy(prompt)
        while len(prompt.shape) < at_least_nd:
            prompt = prompt.unsqueeze(0)
        prompt = prompt.to(self.device)
        return torch.cat((prompt, torch.zeros(prompt.size(0), n_steps).to(prompt)), dim=1)

    @staticmethod
    def generate_tqdm(rng):
        return tqdm(rng, desc="Generate", dynamic_ncols=True, leave=False, unit="step")

    def generate(self, prompt, n_steps, decode_outputs=False, **kwargs):
        raise NotImplementedError

    def encode_inputs(self, inputs: torch.Tensor):
        raise NotImplementedError

    def decode_outputs(self, outputs: torch.Tensor):
        raise NotImplementedError
