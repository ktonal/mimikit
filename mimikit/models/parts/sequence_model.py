import torch
import pytorch_lightning as pl
from abc import ABC
import matplotlib.pyplot as plt
import os

from .hooks import MMKHooks, LoggingHooks
from ...utils import audio

__all__ = [
    'SequenceModel',
    'GenerateCallBack'
]


class SequenceModel(MMKHooks,
                    LoggingHooks,
                    pl.LightningModule,
                    ABC):
    loss_fn = None

    def __init__(self):
        super(pl.LightningModule, self).__init__()
        MMKHooks.__init__(self)
        LoggingHooks.__init__(self)

    def training_step(self, batch, batch_idx):
        batch, target = batch
        output = self.forward(batch)
        return self.loss_fn(output, target)

    def validation_step(self, batch, batch_idx):
        batch, target = batch
        output = self.forward(batch)
        L = self.loss_fn(output, target)
        return {"val_loss" if k == "loss" else k: v for k, v in L.items()}

    def setup(self, stage: str):
        if stage == "fit" and getattr(self, "logger", None) is not None:
            self.logger.log_hyperparams(self.hparams)

    def encode_inputs(self, inputs: torch.Tensor):
        raise NotImplementedError

    def decode_outputs(self, outputs: torch.Tensor):
        raise NotImplementedError

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

    def generate(self, prompt, n_steps=16000,
                 encode_prompt=False, decode_outputs=False,
                 **kwargs):
        # prepare model
        self.before_generate()
        if encode_prompt:
            prompt = self.encode_inputs(prompt)
        output = self.generate_(prompt, n_steps, **kwargs)

        if decode_outputs:
            output = self.decode_outputs(output)

        self.after_generate()
        return output


class GenerateCallBack(pl.callbacks.Callback):

    def __init__(self, every_n_epochs=10, indices=3, n_steps=1000,
                 plot_audios=True, play_audios=True, log_audios=False,
                 log_dir=None,
                 **gen_kwargs):
        self.every_n_epochs = every_n_epochs
        self.indices = indices
        self.n_steps = n_steps
        self.kwargs = gen_kwargs
        self.log_audios = log_audios
        self.plot_audios = plot_audios
        self.play_audios = play_audios
        self.log_dir = log_dir

    def on_train_epoch_start(self, trainer, model):
        # TODO : avoid having to do that!
        dm = trainer.datamodule
        dm.full_ds = dm._init_ds(dm.full_ds, 'fit')

    def on_epoch_end(self, trainer: pl.Trainer, model: SequenceModel):
        if (trainer.current_epoch + 1) % self.every_n_epochs != 0:
            return
        dm = trainer.datamodule
        prompt = dm.get_prompts(self.indices)
        output = model.generate(prompt, self.n_steps, decode_outputs=True, **self.kwargs)
        sr = model.feature.params.get('sr', 22050)
        hop_length = model.feature.params.get('hop_length', 512)
        for i in range(output.size(0)):
            y = output[i].detach().cpu().numpy()
            if self.plot_audios:
                plt.figure(figsize=(20, 2))
                plt.plot(y)
                plt.show()
            if self.play_audios:
                audio(y, sr=sr, hop_length=hop_length)

        if self.log_audios:
            for i in range(output.size(0)):
                filename = "epoch=%i - prompt=%i" % (trainer.current_epoch, i)
                if self.log_dir is not None:
                    os.makedirs(self.log_dir, exist_ok=True)
                    filename = os.path.join(self.log_dir, filename)
                model.log_audio(filename, output[i].unsqueeze(0), sample_rate=sr)
