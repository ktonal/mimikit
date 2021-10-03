import os
from typing import Iterable
import torch
import pytorch_lightning as pl
from matplotlib import pyplot as plt
import soundfile as sf
from pytorch_lightning.trainer.states import TrainerState
from pytorch_lightning import Callback
from IPython import get_ipython

import h5mapper as h5m

__all__ = [
    'is_notebook',
    'EpochProgressBarCallback',
    'GradNormCallback',
    'MMKCheckpoint',
    'tqdm'
]


def is_notebook():
    shell = get_ipython().__class__.__name__
    if shell in ('ZMQInteractiveShell', "Shell"):
        # local and colab notebooks
        return True
    elif shell == 'TerminalInteractiveShell':
        return False
    else:
        return False


if is_notebook():
    from tqdm.notebook import tqdm
else:
    from tqdm import tqdm


class EpochProgressBarCallback(Callback):
    def on_train_start(self, trainer, pl_module):
        self.epoch_bar = tqdm(range(1, trainer.max_epochs), unit="epoch",
                              position=0, leave=False, dynamic_ncols=True)

    def on_train_epoch_end(self, trainer, pl_module, unused=None):
        if trainer.val_dataloaders is not None and any(trainer.val_dataloaders):
            return
        else:
            self.epoch_bar.update()

    def on_validation_epoch_end(self, *args):
        self.epoch_bar.update()


class GradNormCallback(Callback):

    def __init__(self):
        self.gradnorms = []

    def on_after_backward(self, trainer, pl_module) -> None:
        self.gradnorms += [pl_module.grad_norm(1.)]


class MMKCheckpoint(Callback):
    """
    checkpoints saved by this Callback can be loaded with :

        net = net_cls(**h5object.attrs)[.with_io(),
        ...].load_state_dict(h5object.get(id_str))

    """
    format = staticmethod(h5m.TensorDict.format)

    def __init__(self,
                 h5_tensor_dict,
                 id_template="epoch={epoch}-step={step}",
                 epochs=None,
                 ):
        super().__init__()
        self.h5_tensor_dict = h5_tensor_dict
        self.id_template = id_template
        self.epochs = epochs

    def on_pretrain_routine_start(self, trainer, pl_module) -> None:
        self.h5_tensor_dict.save_hp(pl_module.net.hp)
        self.h5_tensor_dict.flush()

    def on_train_epoch_end(self, trainer, pl_module, unused=None) -> None:
        epoch, global_step = trainer.current_epoch + 1, trainer.global_step
        if trainer.state == TrainerState.status.INTERRUPTED or \
                epoch == trainer.max_epochs or self.should_save(epoch, global_step):
            self.save_checkpoint(pl_module, epoch, global_step)

    def should_save(self, epoch, step):
        if type(self.epochs) is int:
            return epoch > 0 and (epoch % self.epochs) == 0
        elif type(self.epochs) is float:  # Doesn't work for now...
            return (step % int(self.epochs)) == 0
        elif isinstance(self.epochs, Iterable):
            return epoch in self.epochs
        return False

    def save_checkpoint(self, pl_module, epoch, global_step):
        id_str = self.format_id(epoch, global_step)
        self.h5_tensor_dict.add(id_str, self.format(pl_module.net.state_dict()))
        self.h5_tensor_dict.flush()

    def format_id(self, epoch, step):
        dct = {"epoch": epoch, "step": step}
        exec(f"out = f'{self.id_template}'", {}, dct)
        return dct["out"]

    def on_fit_end(self, trainer, pl_module) -> None:
        self.h5_tensor_dict.close()

    def teardown(self, trainer, pl_module, stage=None) -> None:
        self.h5_tensor_dict.close()


class GenerateCallback(pl.callbacks.Callback):

    def __init__(self,
                 generate_loop=None,
                 every_n_epochs=10,
                 plot_audios=True,
                 play_audios=True,
                 filename_template="",
                 h5_proxy=None):
        self.loop = generate_loop
        self.every_n_epochs = every_n_epochs
        self.plot_audios = plot_audios
        self.play_audios = play_audios
        self.filename_template = filename_template
        self.h5_proxy = h5_proxy

    def log_audios(self, outputs, epoch):
        for i in range(outputs.size(0)):
            filename = "epoch=%i - prompt=%i" % (epoch, i)
            if self.log_dir is not None:
                os.makedirs(self.log_dir, exist_ok=True)
                filename = os.path.join(self.log_dir, filename)
            model.log_audio(filename, output[i].unsqueeze(0), sample_rate=sr)

    def on_epoch_end(self, trainer: pl.Trainer, model):
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
                plt.show(block=False)
            # if self.play_audios:
            #     audio(y, sr=sr, hop_length=hop_length)

