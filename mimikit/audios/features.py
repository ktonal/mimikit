import torch
import numpy as np
import librosa
import dataclasses as dtc
import IPython.display as ipd
import soundfile as sf

from ..data import Feature
from . import fmodules as T

__all__ = [
    'AudioSignal',
    'MuLawSignal',
    'Spectrogram'
]


@dtc.dataclass
class AudioSignal(Feature):
    """
    audio signal managers
    """
    __ext__ = 'audio'

    sr: int = 22050
    normalize: bool = True
    emphasis: float = 0.

    @property
    def dim(self):
        return 1

    @property
    def encoders(self):
        def func(inputs):
            if self.emphasis:
                inputs = T.Emphasis(self.emphasis)(inputs)
            if self.normalize:
                inputs = T.Normalize()(inputs)
            return inputs

        return {
            np.ndarray: func,
            torch.Tensor: func
        }

    @property
    def decoders(self):
        def func(inputs):
            if self.emphasis:
                inputs = T.Deemphasis(self.emphasis)(inputs)
            if self.normalize:
                inputs = T.Normalize()(inputs)
            return inputs

        return {
            np.ndarray: func,
            torch.Tensor: func
        }

    def load(self, path):
        y = T.FileToSignal(self.sr)(path)
        y = self.encode(y)
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

    @property
    def dim(self):
        return self.q_levels

    @property
    def encoders(self):
        return {
            np.ndarray: T.MuLawCompress(q_levels=self.q_levels),
            torch.Tensor: T.MuLawCompress(q_levels=self.q_levels)
        }

    @property
    def decoders(self):
        return {
            np.ndarray: T.MuLawExpand(self.q_levels),
            torch.Tensor: T.MuLawExpand(self.q_levels)
        }


@dtc.dataclass(unsafe_hash=True)
class Spectrogram(AudioSignal):
    n_fft: int = 2048
    hop_length: int = 512
    magspec: bool = False

    @property
    def dim(self):
        return self.n_fft // 2 + 1

    @property
    def encoders(self):
        if self.magspec:
            return {
                np.ndarray: T.MagSpec(self.n_fft, self.hop_length),
                torch.Tensor: T.MagSpec(self.n_fft, self.hop_length)
            }
        return {
            np.ndarray: T.STFT(self.n_fft, self.hop_length),
            torch.Tensor: T.STFT(self.n_fft, self.hop_length)
        }

    @property
    def decoders(self):
        if self.magspec:
            return {
                np.ndarray: T.GLA(self.n_fft, self.hop_length, n_iter=32),
                torch.Tensor: T.GLA(self.n_fft, self.hop_length, n_iter=32)
            }
        return {
            np.ndarray: T.ISTFT(self.n_fft, self.hop_length),
            torch.Tensor: T.ISTFT(self.n_fft, self.hop_length)
        }

    def display(self, inputs, **waveplot_kwargs):
        y = self.decode(inputs)
        return super(Spectrogram, self).display(y, **waveplot_kwargs)

    def play(self, inputs):
        y = self.decode(inputs)
        return super(Spectrogram, self).play(y)

    def write(self, filename, inputs):
        y = self.decode(inputs)
        return super(Spectrogram, self).write(filename, y)
