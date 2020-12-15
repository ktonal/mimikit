from pytorch_lightning.utilities.cloud_io import load as pl_load
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from time import time, gmtime
from typing import Optional, Union, Callable, Dict, IO
import torch
import os
import warnings

from .. import __version__ as version


class EpochEndPrintHook:

    # unfortunately this can not be a callback since we need the training_step's output...

    def on_epoch_start(self):
        # convenience to have printing of the loss at the end of the epoch
        self._ep_metrics = {}
        self._batch_count = 0

    def training_step_end(self, out):
        for metric, val in out.items():
            if metric not in self._ep_metrics:
                self._ep_metrics.setdefault(metric, val.detach())
            else:
                self._ep_metrics[metric] += val.detach()
        self._batch_count += 1
        return out

    def on_epoch_end(self):
        to_print = "Epoch %i " % self.current_epoch
        for k, v in self._ep_metrics.items():
            to_print += "- %s : %.4f " % (k, v / self._batch_count)
        self.print(to_print)

    def on_fit_start(self):
        self._ep_time = time()

    def on_fit_end(self):
        total_time = gmtime(time() - self._ep_time)
        self.print("Training finished after "
                   "{0} days {1} hours {2} mins {3} seconds".format(total_time[2] - 1, total_time[3],
                                                                    total_time[4], total_time[5]))


def _check_version(other_v):
    if other_v.split(".")[0] < version.split(".")[0]:
        v = str(other_v)
        warnings.warn(("You are loading a checkpoint made by an earlier version of mmk (%s) as the one" % v) +
                      (" imported in this runtime (%s). If you encounter errors " % version) +
                      (", try to downgrade mmk with `pip install mmk==%s" % v))

class MMKHooks:

    @classmethod
    def load_from_checkpoint(
            cls,
            checkpoint_path: Union[str, IO],
            map_location: Optional[Union[Dict[str, str], str, torch.device, int, Callable]] = None,
            strict: bool = True,
            **kwargs):
        """
        This is NOT used when restoring a model from the trainer, but still practical in other cases!
        """
        if map_location is not None:
            checkpoint = pl_load(checkpoint_path, map_location=map_location)
        else:
            checkpoint = pl_load(checkpoint_path, map_location=lambda storage, loc: storage)

        hparams = torch.load(os.path.join(os.path.dirname(checkpoint_path), "hparams.pt"))
        checkpoint[cls.CHECKPOINT_HYPER_PARAMS_KEY] = hparams
        checkpoint[cls.CHECKPOINT_HYPER_PARAMS_KEY].update(kwargs)
        if checkpoint.get("version", None) is not None:
            _check_version(checkpoint["version"])
        model = cls._load_model_state(checkpoint, strict=strict, **kwargs)
        return model

    def on_save_checkpoint(self, checkpoint):
        ckpt_callbacks = [cb for cb in self.trainer.callbacks
                          if issubclass(type(cb), ModelCheckpoint)]
        if any(ckpt_callbacks):
            path = ckpt_callbacks[0].dirpath
        else:
            path = self.trainer.default_root_dir

        # overwrite previous optim state
        torch.save({"optimizer_states": checkpoint["optimizer_states"],
                    "lr_schedulers": checkpoint["lr_schedulers"]
                    }, os.path.join(path, "last_optim_state.ckpt"))

        # pop optim items from checkpoint
        checkpoint.pop("optimizer_states")
        checkpoint.pop("lr_schedulers")

        # save some trainer props
        if not os.path.exists(os.path.join(path, "trainer_props.pt")):
            trainer_props = {}
            for prop in [
                "max_epochs",
                "check_val_every_n_epoch",
                "reload_dataloaders_every_epoch"
            ]:
                trainer_props.update({prop: getattr(self.trainer, prop, None)})

            torch.save(trainer_props, os.path.join(path, "trainer_props.pt"))
        checkpoint["mmk_version"] = version
        return checkpoint

    def on_load_checkpoint(self, checkpoint):
        if checkpoint.get("version", None) is not None:
            _check_version(checkpoint["version"])
        # we restore the training state only if the model has a trainer...
        if getattr(self, "trainer", None) is not None:
            # ... and the trainer got a ckpt_path to resume from
            if self.trainer.resume_from_checkpoint is not None:

                ckpt_path = self.trainer.resume_from_checkpoint
                version_dir = os.path.abspath(os.path.dirname(ckpt_path))

                # make sure we'll find the optim-state or let the user know
                if "last_optim_state.ckpt" not in os.listdir(version_dir):
                    raise FileNotFoundError("Expected to find 'last_optim_state.ckpt' in the parent directory "
                                            "of the model's checkpoint's directory but no such file were found.")
                # update the checkpoint
                optim_state = torch.load(os.path.join(version_dir, "last_optim_state.ckpt"))
                checkpoint["optimizer_states"] = optim_state["optimizer_states"]
                checkpoint["lr_schedulers"] = optim_state["lr_schedulers"]

            # update the trainer
            trainer_props = torch.load(os.path.join(version_dir, "trainer_props.pt"))
            for prop in ["check_val_every_n_epoch",
                         "reload_dataloaders_every_epoch",
                         ]:
                setattr(self.trainer, prop, trainer_props[prop])
            if checkpoint["epoch"] < trainer_props["max_epochs"]:
                self.trainer.max_epochs = self.trainer.max_epochs - checkpoint["epoch"]
            else:
                self.trainer.max_epochs = self.trainer.max_epochs + checkpoint["epoch"]

        return checkpoint
