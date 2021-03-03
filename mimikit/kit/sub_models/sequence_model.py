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
        L = self.__class__.loss_fn(output, target)
        return {"val_loss": L}

    def setup(self, stage: str):
        if stage == "fit" and getattr(self, "logger", None) is not None:
            self.logger.log_hyperparams(self.hparams)

    def batch_info(self, *args, **kwargs):
        raise NotImplementedError("subclasses of `SequenceModel` have to implement `batch_info`")

    # Convenience Methods to generate audios (see also, `LoggingHooks.log_audio`):

    def generation_slices(self):
        raise NotImplementedError("subclasses of `SequenceModel` have to implement `generation_slices`")

    def generate_step(self, so_far_generated, step_idx):
        input_slice, output_slice = self.generation_slices()
        # let this function decide whether to use gradients or not (leaves room for adversarial gen_funcs)
        with torch.no_grad():
            out = self(so_far_generated[:, input_slice])

        return torch.cat((so_far_generated, out[:, output_slice]), dim=1)

    def generate(self, prompt, n_steps, step_func=None, afterwards=None, return_type=torch.Tensor):
        # prepare model
        was_training = self.training
        initial_device = self.device
        self.eval()
        self.to("cuda" if torch.cuda.is_available() else "cpu")

        # prepare prompt
        if not isinstance(prompt, torch.Tensor):
            prompt = torch.from_numpy(prompt)
        if len(prompt.shape) == 2:
            prompt = prompt.unsqueeze(0)
        generated = prompt.to(self.device)

        # main loop
        step_func = self.generate_step if step_func is None else step_func
        for step_idx in tqdm(range(n_steps), desc="Generate", dynamic_ncols=True, leave=False, unit="step"):
            generated = step_func(generated, step_idx)

        if afterwards is not None:
            generated = afterwards(generated)

        # reset model
        self.to(initial_device)
        self.train() if was_training else None

        # get the output right
        if return_type is np.ndarray and type(generated) is torch.Tensor:
            generated = generated.cpu().numpy()

        return generated
