import os
from typing import Iterable
from functools import partial
import pytorch_lightning as pl
from matplotlib import pyplot as plt
from pytorch_lightning.trainer.states import TrainerState
from pytorch_lightning import Callback
from IPython import get_ipython

import h5mapper as h5m
from ..utils import audio


__all__ = [
    'is_notebook',
    'EpochProgressBarCallback',
    'GradNormCallback',
    'MMKCheckpoint',
    'GenerateCallback',
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
                 h5_tensor_dict=None,
                 id_template="epoch={epoch}-step={step}",
                 epochs=None,
                 root_dir=''
                 ):
        super().__init__()
        self.h5_tensor_dict = h5_tensor_dict
        self.id_template = id_template
        self.epochs = epochs
        self.root_dir = root_dir
        self.tp_cls = None
        self.hp = {}

    def on_pretrain_routine_start(self, trainer, pl_module) -> None:
        if not self.root_dir:
            self.root_dir = trainer.default_root_dir
        self.hp = pl_module.net.hp
        self.hp.update({"cls": type(pl_module.net)})

        class CkptBank(h5m.TypedFile):
            ckpt = h5m.TensorDict(pl_module.net.state_dict())

        self.tp_cls = CkptBank

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
        new = self.tp_cls(os.path.join(self.root_dir, f"epoch={epoch}.h5"), "w")
        new.ckpt.save_hp(self.hp)
        new.ckpt.add("state_dict", self.format(pl_module.net.state_dict()))
        new.flush()
        new.close()
        # id_str = self.format_id(epoch, global_step)
        # self.h5_tensor_dict.add(id_str, self.format(pl_module.net.state_dict()))
        # self.h5_tensor_dict.flush()

    def format_id(self, epoch, step):
        dct = {"epoch": epoch, "step": step}
        exec(f"out = f'{self.id_template}'", {}, dct)
        return dct["out"]

    def on_fit_end(self, trainer, pl_module) -> None:
        # self.h5_tensor_dict.close()
        pass

    def teardown(self, trainer, pl_module, stage=None) -> None:
        # self.h5_tensor_dict.close()
        pass


class GenerateCallback(pl.callbacks.Callback):

    def __init__(self,
                 generate_loop=None,
                 every_n_epochs=10,
                 plot_audios=True,
                 play_audios=True,
                 output_features=tuple(),
                 audio_logger=None):
        sample_rate = {feat.sr for feat in output_features
                       if getattr(feat, 'sr', False)}
        sample_rate = sample_rate.pop() if len(sample_rate) > 0 else None
        if play_audios and sample_rate is None:
            raise ValueError("Cannot play audio files if no output_feature has a `sr` attribute")
        self.loop = generate_loop
        self.every_n_epochs = every_n_epochs
        self.plot_audios = plot_audios
        self.play_audios = play_audios
        self.output_features = output_features
        self.sample_rate = sample_rate
        self.logger = audio_logger
        self.indices = ()

    def get_prompt_idx(self, batch, output):
        idx = batch * self.loop.dataloader.batch_size + output
        return self.indices[idx]

    def process_outputs(self, outputs, batch_idx, epoch=0):
        for feature, output in zip(self.output_features, outputs):
            output = feature.inverse_transform(output)
            for i, out in enumerate(output):
                y = out.detach().cpu().numpy()
                if self.plot_audios:
                    plt.figure(figsize=(20, 2))
                    plt.plot(y)
                    plt.show(block=False)
                if self.play_audios:
                    audio(y, sr=getattr(feature, 'sr', 22050),
                          hop_length=getattr(feature, 'hop_length', 512))
                self.logger.write(y,
                                  epoch=epoch,
                                  prompt_idx=self.get_prompt_idx(batch_idx, i))

    def on_epoch_end(self, trainer: pl.Trainer, model):
        if (trainer.current_epoch + 1) % self.every_n_epochs != 0:
            return
        self.indices = self.loop.dataloader.sampler.indices
        self.loop.process_outputs = partial(self.process_outputs,
                                            epoch=trainer.current_epoch+1)
        self.loop.run()

