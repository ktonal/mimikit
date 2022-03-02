from time import time, gmtime
import dataclasses as dtc
from typing import Optional

import ffmpeg
import h5mapper
import soundfile as sf
import numpy as np
import os

import torch
from pytorch_lightning import LightningModule
from pytorch_lightning.utilities import rank_zero_only
from pytorch_lightning.loggers import LightningLoggerBase
from pytorch_lightning.loggers.base import rank_zero_experiment

__all__ = [
    "LoggingHooks",
    "LossLogger",
    'AudioLogger',
    'convert_to_mp3'
]


class LoggingHooks(LightningModule):

    def on_epoch_start(self):
        # convenience to have printing of the loss at the end of the epoch
        self._ep_metrics = {}
        self._batch_count = {}

    def log_output(self, out):
        for metric, val in out.items():
            if torch.isnan(val.detach()):
                raise KeyboardInterrupt(f"metric {metric} is nan")
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

    def _flush_ep_metrics(self):
        to_print = "Epoch %i " % self.current_epoch
        to_log = {}
        for k, v in self._ep_metrics.items():
            to_print += "- %s : %.4f " % (k, v / self._batch_count[k])
            to_log[k] = (v / self._batch_count[k]).item()
        self.print(to_print)
        if getattr(self, "logger", None) is not None:
            self.logger.log_metrics(to_log, self.current_epoch)

    def on_train_epoch_end(self, *args):
        if self.trainer.val_dataloaders is not None and any(self.trainer.val_dataloaders):
            # wait until validation end
            return
        else:
            self._flush_ep_metrics()

    def on_validation_epoch_end(self, *args):
        self._flush_ep_metrics()

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


class LossLogger(LightningLoggerBase):

    def __init__(self, logs_file):
        super(LossLogger, self).__init__()
        self.logs_file = logs_file

    @property
    def name(self):
        return 'LossLogger'

    @property
    @rank_zero_experiment
    def experiment(self):
        # Return the experiment object associated with this logger.
        return None

    @property
    def version(self):
        # Return the experiment version, int or str.
        return '0.1'

    @rank_zero_only
    def log_hyperparams(self, params):
        # params is an argparse.Namespace
        # your code to record hyperparameters goes here
        pass

    @rank_zero_only
    def log_metrics(self, metrics, step):
        # metrics is a dictionary of metric names and values
        # your code to record metrics goes here
        self.logs_file.add(str(step), {k: np.array([v]) for k, v in metrics.items()})

    @rank_zero_only
    def save(self):
        # Optional. Any code necessary to save logger data goes here
        # If you implement this, remember to call `super().save()`
        # at the start of the method (important for aggregation of metrics)
        super().save()
        self.logs_file.flush()

    @rank_zero_only
    def finalize(self, status):
        # Optional. Any code that needs to be run after training
        # finishes goes here
        pass


@dtc.dataclass
class AudioLogger:
    sr: int = 16000
    hop_length: int = 512
    filename_template: Optional[str] = None
    target_dir: Optional[str] = None
    id_template: Optional[str] = None
    proxy_template: Optional[str] = None
    target_bank: Optional[h5mapper.TypedFile] = None

    @staticmethod
    def format_template(template, **parameters):
        exec(f"out = f'{template}'", {}, parameters)
        return parameters["out"]

    def log_mp3(self, audio, **params):
        filename = self.format_template(self.filename_template, **params)
        if '.wav' not in filename:
            filename += '.wav'
        if self.target_dir:
            os.makedirs(self.target_dir, exist_ok=True)
            if self.target_dir not in filename:
                filename = os.path.join(self.target_dir, filename)
        if isinstance(audio, torch.Tensor):
            audio = audio.squeeze().detach().cpu().numpy()
        sf.write(filename, audio, self.sr, 'PCM_24')
        convert_to_mp3(filename)

    def log_h5(self, audio, **params):
        if isinstance(audio, torch.Tensor):
            audio = audio.squeeze().detach().cpu().numpy()
        src = self.format_template(self.id_template, **params)
        proxy = self.format_template(self.proxy_template, **params)
        self.target_bank.add(src, {proxy: audio})
        self.target_bank.flush()

    def write(self, audio, **params):
        if self.filename_template and self.target_dir:
            self.log_mp3(audio, **params)
        if self.id_template and self.proxy_template and self.target_bank:
            self.log_h5(audio, **params)


def convert_to_mp3(file_path, exists_ok=True):
    if exists_ok and os.path.isfile(os.path.splitext(file_path)[0] + ".mp3"):
        os.remove(os.path.splitext(file_path)[0] + ".mp3")
    stream = ffmpeg.input(file_path)
    stream.output(os.path.splitext(file_path)[0] + ".mp3").run()
    os.remove(file_path)