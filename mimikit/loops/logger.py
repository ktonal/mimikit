from time import time, gmtime
import dataclasses as dtc
from typing import Optional, Union
from matplotlib import pyplot as plt
import IPython.display as ipd

import pydub
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
]


class LoggingHooks(LightningModule):

    def on_train_epoch_start(self):
        # convenience to have printing of the loss at the end of the epoch
        self._ep_metrics = {}
        self._batch_count = {}

    def on_before_backward(self, loss: torch.Tensor) -> None:
        if torch.isnan(loss.detach()) or torch.isinf(loss.detach().abs()):
            raise RuntimeError(f"loss is {loss.detach()}")

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
        super(LoggingHooks, self).on_train_epoch_end(*args)
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
    file_template: Optional[str] = None  # file or title template
    title_template: Optional[str] = None

    figsize = (30, 4)

    def __post_init__(self):
        if self.file_template is not None:
            self.target_dir = os.path.dirname(self.file_template)
            self.format = os.path.splitext(self.file_template)[-1][1:]

    @staticmethod
    def format_template(template, **parameters):
        exec(f"__out__ = f'{template}'", {}, parameters)
        return parameters["__out__"]

    @staticmethod
    def to_numpy(audio: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
        if audio.ndim > 1:
            raise ValueError(f"Expected `audio` array to have a single dimension, got {audio.ndim}.")
        if isinstance(audio, torch.Tensor):
            audio = audio.squeeze().detach().cpu().numpy()
        return audio

    def write(self, audio: Union[np.ndarray, torch.Tensor], **template_params):
        audio = self.to_numpy(audio)
        filename = self.format_template(self.file_template, **template_params)
        os.makedirs(self.target_dir, exist_ok=True)
        # normalize
        y = np.int16(audio * 2 ** 15)
        segment = pydub.AudioSegment(y.tobytes(),
                                     frame_rate=self.sr,
                                     sample_width=2,
                                     channels=1)
        params = dict(bitrate="320k")
        with open(filename, "wb") as f:
            segment.export(f, format=self.format,
                           **(params if self.format in {"mp3", "mp4", "m4a"} else {}))

    def write_batch(self, audio, **template_params):
        pass

    def display(self, audio, **template_params):
        self.display_waveform(audio, **template_params)
        self.display_html(audio, **template_params)

    def display_batch(self, audio, **template_params):
        for y in audio:
            self.display(y, **template_params)

    def display_spectrogram(self, audio, **template_params):
        pass

    def display_waveform(self, audio, **template_params):
        audio = self.to_numpy(audio)
        plt.figure(figsize=self.figsize)
        plt.plot(audio)
        if template_params:
            plt.title(self.format_template(self.title_template, **template_params))
        plt.show(block=False)

    def display_html(self, audio, **template_params):
        audio = self.to_numpy(audio)
        ipd.display(ipd.Audio(audio, rate=self.sr))
