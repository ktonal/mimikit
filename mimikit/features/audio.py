import h5mapper
import numpy as np
import torch.nn as nn
import librosa
import dataclasses as dtc
import IPython.display as ipd
import soundfile as sf
import torch

from . import Feature
from . import audio_fmodules as T
from ..modules import mean_L1_prop, mean_2d_diff


__all__ = [
    'AudioSignal',
    'MuLawSignal',
    'ALawSignal',
    'Spectrogram',
    'MultiScale',
]


@dtc.dataclass(unsafe_hash=True)
class AudioSignal(Feature):
    """
    audio signal managers
    """
    __ext__ = 'audio'
    domain = "time"

    sr: int = 22050
    normalize: bool = False
    emphasis: float = 0.

    def __post_init__(self):
        self.base_feature = T.FileToSignal(self.sr)

    def transform(self, inputs):
        if self.emphasis:
            inputs = T.Emphasis(self.emphasis)(inputs)
        if self.normalize:
            inputs = T.Normalize()(inputs)
        return inputs

    def inverse_transform(self, inputs):
        if self.emphasis:
            inputs = T.Deemphasis(self.emphasis)(inputs)
        if self.normalize:
            inputs = T.Normalize()(inputs)
        return inputs

    def batch_item(self, data='snd', shift=0, length=1,
                   frame_size=None,
                   hop_length=None,
                   center=False,
                   pad_mode="reflect",
                   downsampling=1,
                   **kwargs):
        if frame_size is None:
            getter = h5mapper.AsSlice(shift=shift, length=length, downsampling=downsampling)
        else:
            getter = h5mapper.AsFramedSlice(shift=shift, length=length,
                                            frame_size=frame_size, hop_length=hop_length,
                                            center=center, pad_mode=pad_mode,
                                            downsampling=downsampling)
        inpt = h5mapper.Input(data=data, getter=getter, transform=self.transform)
        return inpt

    def loss_fn(self, output, target):
        return {"loss": nn.MSELoss()(output.squeeze(), target) * 100}

    def load(self, path):
        y = self.base_feature(path)
        y = self.transform(y)
        return y

    def display(self, inputs, **waveplot_kwargs):
        waveplot_kwargs.setdefault('sr', self.sr)
        return librosa.display.waveplot(inputs, **waveplot_kwargs)

    def play(self, inputs):
        return ipd.display(ipd.Audio(inputs, rate=self.sr))

    def write(self, filename, inputs):
        return sf.write(filename, inputs, self.sr, 'PCM_24')


@dtc.dataclass(unsafe_hash=True)
class MuLawSignal(AudioSignal):
    q_levels: int = 256
    pr_y: torch.tensor = None

    def __post_init__(self):
        self.base_feature = AudioSignal(self.sr, self.normalize, self.emphasis)
        self.transform_ = T.MuLawCompress(q_levels=self.q_levels)
        self.inverse_transform_ = T.MuLawExpand(self.q_levels)

    def transform(self, inputs):
        return self.transform_(inputs)

    def inverse_transform(self, inputs):
        return self.inverse_transform_(inputs)

    def loss_fn(self, output, target):
        # FFT
        # y_o, y_t = self.inverse_transform(output.argmax(dim=-1)), self.inverse_transform(target)
        # fft = Spectrogram(sr=self.sr, n_fft=512, hop_length=512, coordinate="mag", center=True)
        # S_o, S_t = fft.transform(y_o), fft.transform(y_t)
        # fft_loss = mean_L1_prop(S_o, S_t)

        # Gumbel softmax
        # output, target = output.view(-1, output.size(-1)), target.view(-1)
        # output = nn.functional.gumbel_softmax(output, tau=1.)
        # criterion = nn.NLLLoss(reduction='mean', weight=self.pr_y)
        # loss = criterion(torch.maximum(output, torch.tensor(1e-16).view(1, 1).to(output.device)).log_(), target)
        # diff = output.detach().argmax(dim=-1) != target
        # err = diff.sum() / output.size(0)

        # cross entropy
        criterion = nn.CrossEntropyLoss(reduction="none", weight=self.pr_y)
        output, target = output.view(-1, output.size(-1)), target.view(-1)
        loss = criterion(output, target).mean()
        diff = output.detach().argmax(dim=-1) != target
        err = diff.sum() / output.size(0)

        return {"loss": loss, "err": err}


@dtc.dataclass(unsafe_hash=True)
class ALawSignal(AudioSignal):
    A: float = 87.7
    q_levels: int = 256
    target_width: int = 1

    def __post_init__(self):
        self.base_feature = AudioSignal(self.sr, self.normalize, self.emphasis)
        self.transform_ = T.ALawCompress(A=self.A, q_levels=self.q_levels)
        self.inverse_transform_ = T.ALawExpand(A=self.A, q_levels=self.q_levels)

    def transform(self, inputs):
        return self.transform_(inputs)

    def inverse_transform(self, inputs):
        return self.inverse_transform_(inputs)

    def loss_fn(self, output, target):
        criterion = nn.CrossEntropyLoss(reduction="mean")
        L = criterion(output.view(-1, output.size(-1)), target.view(-1))
        return {"loss": L}


class MultiScale(Feature):

    def __init__(self, base, frame_sizes, hop_lengths):
        self.base = base
        self.frame_sizes = frame_sizes
        self.hop_lengths = hop_lengths

    def transform(self, inputs):
        return self.base.transform(inputs)

    def inverse_transform(self, inputs):
        return self.base.inverse_transform(inputs)

    def batch_item(self, data='snd', shift=0, length=1,
                   frame_size=None, hop_length=None, center=False, pad_mode="reflect",
                   training=True, **kwargs):
        if training:
            return tuple(
                self.base.batch_item(data, self.frame_sizes[0] - fs, length, frame_size=fs, hop_length=hop)
                for fs, hop in zip(self.frame_sizes, self.hop_lengths)
            )
        else:
            return self.base.batch_item(data, shift, length)

    def loss_fn(self, output, target):
        return self.base.loss_fn(output, target)


@dtc.dataclass(unsafe_hash=True)
class Spectrogram(AudioSignal):
    domain = "time-freq"

    n_fft: int = 2048
    hop_length: int = 512
    coordinate: str = 'pol'
    center: bool = True

    def __post_init__(self):
        self.base_feature = AudioSignal(self.sr, self.normalize, self.emphasis)
        if self.coordinate == 'mag':
            self.transform_ = T.MagSpec(self.n_fft, self.hop_length, center=self.center)
            self.inverse_transform_ = T.GLA(self.n_fft, self.hop_length, center=self.center, n_iter=32)
        else:
            self.transform_ = T.STFT(self.n_fft, self.hop_length, self.coordinate, center=self.center)
            self.inverse_transform_ = T.ISTFT(self.n_fft, self.hop_length, self.coordinate, center=self.center)

    def transform(self, inputs):
        return self.transform_(inputs)

    def inverse_transform(self, inputs):
        return self.inverse_transform_(inputs)

    def batch_item(self, data='snd', shift=0, length=1, downsampling=1, **kwargs):
        getter = h5mapper.AsSlice(shift=shift, length=length, downsampling=downsampling)
        getter.shift, getter.length = getter.shift_and_length_to_samples(
            self.n_fft, self.hop_length, self.center
        )
        return h5mapper.Input(data=data, getter=getter, transform=self.transform)

    def loss_fn(self, output, target):
        if self.coordinate == 'mag':
            return {"loss": mean_L1_prop(output, target)}
        elif self.coordinate == 'pol':
            mag = mean_L1_prop(output[..., 0], target[..., 0])
            phs = mean_2d_diff(output[..., 1], target[..., 1])
            return {"loss": mag + phs, 'mag': mag.detach(), "phs": phs.detach()}
        else:
            raise NotImplementedError(f"no default loss function for coordinate '{self.coordinate}'")
