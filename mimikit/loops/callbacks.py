import os
from time import time
from typing import Iterable
import pytorch_lightning as pl
from pytorch_lightning.trainer.states import TrainerState
from pytorch_lightning import Callback
from IPython import get_ipython

from ..checkpoint import Checkpoint


__all__ = [
    'is_notebook',
    'EpochProgressBarCallback',
    'TrainingProgressBar',
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
    from tqdm.auto import tqdm


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


class TrainingProgressBar(pl.callbacks.TQDMProgressBar):
    def init_train_tqdm(self):
        """Override this to customize the tqdm bar for training."""
        bar = tqdm(
            desc=self.train_description,
            initial=self.train_batch_idx,
            position=int(is_notebook())*2,
            disable=self.is_disabled,
            leave=True,
            dynamic_ncols=True,
            # file=sys.stdout,
            smoothing=0,
            mininterval=1.
        )
        return bar

    def _should_update(self, current: int, total: int) -> bool:
        if getattr(self, "_last_update", None) is None:
            self._last_update = time()
            self._last_current = 0
            return True
        now = time()
        should = (now - self._last_update) > 1. or current == total
        if should:
            self._last_update = now
        return should

    def on_train_epoch_start(self, trainer: "pl.Trainer", *_):
        super(TrainingProgressBar, self).on_train_epoch_start(trainer)
        self._last_current = 0

    def on_train_batch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", *_):
        current = self.train_batch_idx + self._val_processed
        if self._should_update(current, self.main_progress_bar.total):
            self.main_progress_bar.update(current - self._last_current)
            self.main_progress_bar.refresh()
            self.main_progress_bar.set_postfix(self.get_metrics(trainer, pl_module))
            self._last_current = current


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

    def on_fit_start(self, trainer, pl_module: "TrainARMLoop") -> None:
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

