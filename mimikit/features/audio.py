from typing import Optional

import torch.nn as nn
import librosa
import dataclasses as dtc
import IPython.display as ipd
import soundfile as sf

from . import Feature
from . import audio_fmodules as T
from ..modules import mean_L1_prop
from ..networks import SingleClassMLP


__all__ = [
    'AudioSignal',
    'MuLawSignal',
    'Spectrogram'
]


@dtc.dataclass(unsafe_hash=True)
class AudioSignal(Feature):
    """
    audio signal managers
    """
    __ext__ = 'audio'

    sr: int = 22050
    normalize: bool = True
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

    def input_module(self, net_dim):
        return nn.Identity()

    def output_module(self, net_dim):
        return nn.Identity()

    def loss_fn(self, output, target):
        return nn.MSELoss()(output, target)

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
    input_mod_dim: Optional[int] = None
    output_mod_dim: Optional[int] = None

    def __post_init__(self):
        self.base_feature = AudioSignal(self.sr, self.normalize, self.emphasis)
        self.transform_ = T.MuLawCompress(q_levels=self.q_levels)
        self.inverse_transform_ = T.MuLawExpand(self.q_levels)

    def transform(self, inputs):
        return self.transform_(inputs)

    def inverse_transform(self, inputs):
        return self.inverse_transform_(inputs)

    def input_module(self, net_dim):
        return nn.Embedding(self.q_levels, net_dim or self.input_mod_dim)

    def output_module(self, net_dim):
        return SingleClassMLP(net_dim, net_dim or self.output_mod_dim, self.q_levels, nn.ReLU())

    def loss_fn(self, output, target):
        criterion = nn.CrossEntropyLoss(reduction="mean")
        return criterion(output.view(-1, output.size(-1)), target.view(-1))


@dtc.dataclass(unsafe_hash=True)
class Spectrogram(AudioSignal):
    n_fft: int = 2048
    hop_length: int = 512
    coordinate: str = 'car'
    input_mod_dim: Optional[int] = None
    output_mod_dim: Optional[int] = None

    def __post_init__(self):
        self.base_feature = AudioSignal(self.sr, self.normalize, self.emphasis)
        if self.coordinate == 'mag':
            self.transform_ = T.MagSpec(self.n_fft, self.hop_length)
            self.inverse_transform_ = T.GLA(self.n_fft, self.hop_length, n_iter=32)
        else:
            self.transform_ = T.STFT(self.n_fft, self.hop_length, self.coordinate)
            self.inverse_transform_ = T.ISTFT(self.n_fft, self.hop_length, self.coordinate)

    def transform(self, inputs):
        return self.transform_(inputs)

    def inverse_transform(self, inputs):
        return self.inverse_transform_(inputs)

    def input_module(self, net_dim):
        pass

    def output_module(self, net_dim):
        pass

    def loss_fn(self, output, target):
        if self.coordinate == 'mag':
            return mean_L1_prop(output, target)


