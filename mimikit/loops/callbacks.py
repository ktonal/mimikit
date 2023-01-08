import os
from typing import Iterable
import pytorch_lightning as pl
from pytorch_lightning.trainer.states import TrainerState
from pytorch_lightning import Callback
from IPython import get_ipython

from ..checkpoint import Checkpoint


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

    def __init__(self,
                 epochs=None,
                 root_dir=''
                 # todo: save_optimizer
                 ):
        super().__init__()
        self.epochs = epochs
        self.root_dir = root_dir
        self.config = None

    def on_fit_start(self, trainer, pl_module: "TrainLoop") -> None:
        config = pl_module.config
        # make sure that we can (de)serialize
        _class = type(config)
        yaml = config.serialize()
        _class.deserialize(yaml)
        self.config = config

    def on_train_epoch_end(self, trainer, pl_module, unused=None) -> None:
        epoch, global_step = trainer.current_epoch + 1, trainer.global_step
        if trainer.state == TrainerState.status.INTERRUPTED or \
                epoch == trainer.max_epochs or self.should_save(epoch, global_step):
            self.save_checkpoint(pl_module, epoch)

    def should_save(self, epoch, step):
        if type(self.epochs) is int:
            return epoch > 0 and (epoch % self.epochs) == 0
        elif type(self.epochs) is float:  # Doesn't work for now...
            return (step % int(self.epochs)) == 0
        elif isinstance(self.epochs, Iterable):
            return epoch in self.epochs
        return False

    def save_checkpoint(self, pl_module, epoch):
        root_dir, training_id = os.path.split(self.root_dir)
        Checkpoint(id=training_id, epoch=epoch, root_dir=root_dir)\
            .create(pl_module.net, self.config, optimizer=None)


class GenerateCallback(pl.callbacks.Callback):

    def __init__(self,
                 generate_loop=None,
                 every_n_epochs=10
                 ):
        self.loop = generate_loop
        self.every_n_epochs = every_n_epochs

    def on_train_epoch_end(self, trainer: pl.Trainer, model):
        if (trainer.current_epoch + 1) % self.every_n_epochs != 0:
            return
        self.loop.template_vars = dict(epoch=trainer.current_epoch+1)
        for _ in self.loop.run():
            continue

