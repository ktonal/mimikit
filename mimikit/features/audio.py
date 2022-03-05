import h5mapper
import torch.nn as nn
import librosa
import dataclasses as dtc
import IPython.display as ipd
import soundfile as sf
import torch
from mimikit.modules.misc import Chunk, Flatten, Abs

from . import Feature
from . import audio_fmodules as T
from ..modules import mean_L1_prop, mean_2d_diff, HOM, ScaledTanh, ScaledSigmoid, ScaledAbs, Maybe
from ..networks import SingleClassMLP

__all__ = [
    'AudioSignal',
    'MuLawSignal',
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
        return h5mapper.Input(data=data, getter=getter, transform=self.transform)

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
    target_width: int = 1

    def __post_init__(self):
        self.base_feature = AudioSignal(self.sr, self.normalize, self.emphasis)
        self.transform_ = T.MuLawCompress(q_levels=self.q_levels)
        self.inverse_transform_ = T.MuLawExpand(self.q_levels)

    def transform(self, inputs):
        return self.transform_(inputs)

    def inverse_transform(self, inputs):
        return self.inverse_transform_(inputs)

    def loss_fn(self, output, target):
        criterion = nn.CrossEntropyLoss(reduction="mean")
        L = criterion(output.view(-1, output.size(-1)), target.view(-1))
        C = output.size(-1)
        Ct, Zerot = torch.tensor([C - 1]).to(target), torch.tensor([0]).to(target)
        for w in range(2, self.target_width + 1):
            d = .5 * 1 / w
            L += d * criterion(output.view(-1, C),
                               torch.minimum(target.view(-1) + w, Ct))
            L += d * criterion(output.view(-1, C),
                               torch.maximum(target.view(-1) - w, Zerot))
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
