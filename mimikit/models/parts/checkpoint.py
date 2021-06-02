from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning.trainer.states import TrainerState
from typing import Iterable
import os

__all__ = [
    'MMKCheckpoint'
]


class MMKCheckpoint(ModelCheckpoint):
    def __init__(self,
                 dirpath="./",
                 epochs=None,
                 ):
        filename = "{step}" if type(epochs) is float else "{epoch}"
        self.epochs = epochs

        super(MMKCheckpoint, self).__init__(
            monitor=None,
            verbose=False,
            save_last=None,
            save_top_k=None,
            save_weights_only=False,  # we just save optimizers separately
            mode="min",
            period=1,
            dirpath=os.path.join(dirpath, "states"),
            filename=filename
        )

    def should_save(self, epoch, step):
        if type(self.epochs) is int:
            return epoch > 0 and (epoch % self.epochs) == 0
        elif type(self.epochs) is float:  # Doesn't work for now...
            return (step % int(self.epochs)) == 0
        elif isinstance(self.epochs, Iterable):
            return epoch in self.epochs
        return False

    def save_checkpoint(self, trainer, pl_module):
        epoch, global_step = trainer.current_epoch + 1, trainer.global_step
        if trainer.state == TrainerState.INTERRUPTED or \
                epoch == trainer.max_epochs or self.should_save(epoch, global_step):
            # hacks the code from github
            self._add_backward_monitor_support(trainer)
            self._validate_monitor_key(trainer)

            # track epoch when ckpt was last checked
            self.last_global_step_saved = global_step

            filepath = self.format_checkpoint_name(epoch, global_step, {})

            self._save_model(filepath, trainer, pl_module)

    def on_save_checkpoint(self, trainer, model, checkpoint):
        """returns the state to be saved in the checkpoint"""
        return {"epochs": self.epochs}

    def on_load_checkpoint(self, state):
        """update own state according to checkpoint"""
        self.epochs = state["epochs"]
