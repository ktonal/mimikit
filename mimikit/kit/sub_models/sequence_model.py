import torch
from pytorch_lightning import LightningModule
import librosa
from abc import ABC

from .utils import MMKHooks, LoggingHooks, tqdm
from ..sub_models.data import DataSubModule


class SequenceModel(MMKHooks,
                    LoggingHooks,
                    DataSubModule,
                    LightningModule,
                    ABC):

    db_class = None
    loss_fn = None

    def __init__(self,
                 db=None,
                 files=None,
                 batch_size=64,
                 in_mem_data=True,
                 splits=[.8, .2],
                 **loaders_kwargs):
        super(LightningModule, self).__init__()
        MMKHooks.__init__(self)
        LoggingHooks.__init__(self)
        DataSubModule.__init__(self, db, files, in_mem_data, splits, batch_size=batch_size, **loaders_kwargs)
        self.save_hyperparameters()

    def training_step(self, batch, batch_idx):
        batch, target = batch
        output = self.forward(batch)
        recon = self.loss_fn(output, target)
        return {"loss": recon}

    def validation_step(self, batch, batch_idx):
        batch, target = batch
        output = self.forward(batch)
        recon = self.loss_fn(output, target)
        return {"val_loss": recon}

    def setup(self, stage: str):
        if stage == "fit":
            self.logger.log_hyperparams(self.hparams)

    def batch_info(self, *args, **kwargs):
        raise NotImplementedError("subclasses of `SequenceModel` have to implement `batch_info`")

    # Convenience Methods to generate audios (see also, `LoggingHooks.log_audio`):

    def generation_slices(self):
        raise NotImplementedError("subclasses of `SequenceModel` have to implement `generation_slices`")

    def on_generate_end(self, generated):
        pass

    def generate(self, prompt, n_steps, hop_length=512, time_domain=True):
        was_training = self.training
        self.eval()
        initial_device = self.device
        self.to("cuda" if torch.cuda.is_available() else "cpu")
        if not isinstance(prompt, torch.Tensor):
            prompt = torch.from_numpy(prompt)
        if len(prompt.shape) < 3:
            prompt = prompt.unsqueeze(0)
        input_slice, output_slice = self.generation_slices()
        generated = prompt.to(self.device)
        for _ in tqdm(range(n_steps), desc="Generate", dynamic_ncols=True, leave=False, unit="step"):
            with torch.no_grad():
                out = self(generated[:, input_slice])
                generated = torch.cat((generated, out[:, output_slice]), dim=1)
        if time_domain:
            generated = generated.transpose(1, 2).squeeze()
            generated = librosa.griffinlim(generated.cpu().numpy(), hop_length=hop_length, n_iter=64)
            generated = torch.from_numpy(generated)
        else:  # for consistency with time_domain :
            generated = generated.cpu()
        self.to(initial_device)
        self.train() if was_training else None
        return generated.unsqueeze(0)
