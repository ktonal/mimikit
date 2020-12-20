from pytorch_lightning.utilities.cloud_io import load as pl_load
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from time import time, gmtime
from typing import Optional, Union, Callable, Dict, IO
import torch
import os
import warnings
import torchaudio

try:
    from neptune.experiments import Experiment as NeptuneExperiment
    from neptunecontrib.api.audio import log_audio
except ImportError:
    NeptuneExperiment = type(None)
    warnings.warn("It seems neptune and/or neptunecontrib are not installed. "
                  "You won't be able to log checkpoints and data to neptune.")

from .. import __version__ as version
from ..data.transforms import SR

torchaudio.set_audio_backend("sox_io")


class LoggingHooks:

    # unfortunately this can not be a callback since we need the training_step's output...

    def on_epoch_start(self):
        # convenience to have printing of the loss at the end of the epoch
        self._ep_metrics = {}
        self._batch_count = {}

    def log_output(self, out):
        for metric, val in out.items():
            if metric not in self._ep_metrics:
                self._ep_metrics.setdefault(metric, val.detach())
                self._batch_count[metric] = 0
            else:
                self._ep_metrics[metric] += val.detach()
            self._batch_count[metric] += 1
        return out

    def training_step_end(self, out):
        return self.log_output(out)

    def validation_step_end(self, out):
        return self.log_output(out)

    def on_train_epoch_end(self, *args):
        to_print = "Epoch %i " % self.current_epoch
        to_log = {}
        for k, v in self._ep_metrics.items():
            to_print += "- %s : %.4f " % (k, v / self._batch_count[k])
            to_log[k] = (v / self._batch_count[k]).item()
        self.print(to_print)
        if getattr(self, "logger", None) is not None:
            self.logger.log_metrics(to_log)

    def on_fit_start(self):
        self._ep_time = time()

    def on_fit_end(self):
        duration = time() - self._ep_time
        total_time = gmtime(duration)
        self.print("Training finished after "
                   "{0} days {1} hours {2} mins {3} seconds".format(total_time[2] - 1, total_time[3],
                                                                    total_time[4], total_time[5]))
        if getattr(self, "logger", None) is not None:
            # accumulate if we resumed
            duration = self.hparams.get("training_time_sec", 0) + duration
            self.hparams.update({"training_time_sec": duration})
            self.logger.log_hyperparams(self.hparams)

    def log_audio(self, filename, audio_tensor, sample_rate=SR):
        # TensorBoards write their own "Event" file which is quite cumbersome to extract afterwards...
        # NeptuneLoggers need an audio file written on disk and store it afterwards as html on the server...
        # Just to be sure and even if it is then in 3 places : we add the audio in root_dir/audios of the model!
        path = filename
        if self.trainer is not None:
            if self.trainer.default_root_dir not in path:
                root = os.path.join(self.trainer.default_root_dir, "audios")
                if not os.path.exists(root):
                    os.mkdir(root)
                if ".wav" != os.path.splitext(path)[-1]:
                    path = os.path.join(root, path + ".wav")
        torchaudio.save(path, audio_tensor, sample_rate)
        for exp in self.logger.experiment:
            # Neptune and TestTube experiments have different APIs...
            if getattr(exp, "add_audio", False):
                exp.add_audio(filename + ".wav", audio_tensor, sample_rate=sample_rate)
                exp.save()
                print("Updated TensorBoard with", filename)
            elif isinstance(exp, NeptuneExperiment):
                log_audio(path, filename, exp)
                print("Updated neptune experiment", exp.id, "with", filename)
        return 1


def _check_version(other_v):
    if other_v.split(".")[0] != version.split(".")[0]:
        v = str(other_v)
        warnings.warn(("You are loading a checkpoint made by a different version of mmk (%s) as the one" % v) +
                      (" imported in this runtime (%s). If you encounter errors " % version) +
                      (", try to install the right version with `pip install mmk==%s" % v))


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

            self.trainer.max_epochs = self.trainer.max_epochs + checkpoint["epoch"]
        return checkpoint

    def upload_to_neptune(self, experiment=None):
        if experiment is None:
            if not any(isinstance(exp, NeptuneExperiment) for exp in self.logger.experiment):
                raise ValueError("`experiment` is None and this model isn't bound to any NeptuneExperiment...")
            experiment = [exp for exp in self.logger.experiment if isinstance(exp, NeptuneExperiment)][0]
        # log everything!
        experiment.log_artifact(self.trainer.default_root_dir)
        print("uploaded", self.trainer.default_root_dir, "to", experiment.id)
        return 1

