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
from ..modules import mean_L1_prop, mean_2d_diff, HOM, ScaledTanh, ScaledSigmoid, ScaledAbs
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
                   frame_size=None, hop_length=None, center=False, pad_mode="reflect",
                   downsampling=1, **kwargs):
        if frame_size is None:
            getter = h5mapper.AsSlice(shift=shift, length=length, downsampling=downsampling)
        else:
            getter = h5mapper.AsFramedSlice(shift=shift, length=length,
                                            frame_size=frame_size, hop_length=hop_length,
                                            center=center, pad_mode=pad_mode,
                                            downsampling=downsampling)
        return h5mapper.Input(data=data, getter=getter, transform=self.transform)

    def input_module(self, *args, **kwargs):
        return nn.Identity()

    def output_module(self, *args, **kwargs):
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

    def __post_init__(self):
        self.base_feature = AudioSignal(self.sr, self.normalize, self.emphasis)
        self.transform_ = T.MuLawCompress(q_levels=self.q_levels)
        self.inverse_transform_ = T.MuLawExpand(self.q_levels)

    def transform(self, inputs):
        return self.transform_(inputs)

    def inverse_transform(self, inputs):
        return self.inverse_transform_(inputs)

    def input_module(self, net_in_dim, **kwargs):
        return nn.Embedding(self.q_levels, net_in_dim, **kwargs)

    def output_module(self, net_dim, mlp_dim):
        return SingleClassMLP(net_dim, mlp_dim, self.q_levels, nn.ReLU())

    def loss_fn(self, output, target):
        criterion = nn.CrossEntropyLoss(reduction="mean")
        return {"loss": criterion(output.view(-1, output.size(-1)), target.view(-1))}


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
                self.base.batch_item(data, self.frame_sizes[0]-fs, length, frame_size=fs, hop_length=hop)
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

    def input_module(self, in_dim, net_dim, n_chunks):
        if self.coordinate == 'mag':
            return Chunk(nn.Linear(in_dim, net_dim * n_chunks), n_chunks, sum_out=True)
        elif self.coordinate == 'pol':
            return HOM('x -> x',
                       (Flatten(2), 'x -> x'),
                       (Chunk(nn.Linear(in_dim * 2, net_dim * n_chunks), n_chunks, sum_out=True), 'x -> x'))
        else:
            raise NotImplementedError(f"no input module for coordinate '{self.coordinate}'")

    def output_module(self, net_dim, out_dim, n_chunks, scaled_activation=False):
        if self.coordinate == 'mag':
            return nn.Sequential(Chunk(nn.Linear(net_dim, out_dim * n_chunks), n_chunks, sum_out=True),
                                 ScaledSigmoid(out_dim, with_range=False) if scaled_activation else Abs())
        elif self.coordinate == 'pol':
            pi = torch.acos(torch.zeros(1)).item()
            act_phs = ScaledTanh(out_dim, with_range=False) if scaled_activation else nn.Tanh()
            act_mag = ScaledSigmoid(out_dim, with_range=False) if scaled_activation else Abs()

            class ScaledPhase(HOM):
                def __init__(self):
                    super(ScaledPhase, self).__init__(
                        "x -> phs",
                        (nn.Sequential(Chunk(nn.Linear(net_dim, out_dim * n_chunks), n_chunks, sum_out=True), act_phs),
                         'x -> phs'),
                        (lambda self, phs: torch.cos(
                            phs * self.psis.to(phs).view(*([1] * (len(phs.shape) - 1)), -1)) * pi,
                         'self, phs -> phs'),
                    )
                    self.psis = nn.Parameter(torch.ones(out_dim))

            return HOM("x -> y",
                       # phase module
                       (ScaledPhase(), 'x -> phs'),
                       # magnitude module
                       (nn.Sequential(Chunk(nn.Linear(net_dim, out_dim * n_chunks), n_chunks, sum_out=True), act_mag),
                        'x -> mag'),
                       (lambda mag, phs: torch.stack((mag, phs), dim=-1), "mag, phs -> y")
                       )
        else:
            raise NotImplementedError(f"no default output module for coordinate '{self.coordinate}'")

    def loss_fn(self, output, target):
        if self.coordinate == 'mag':
            return {"loss": mean_L1_prop(output, target)}
        elif self.coordinate == 'pol':
            mag = mean_L1_prop(output[..., 0], target[..., 0])
            phs = mean_2d_diff(output[..., 1], target[..., 1])
            return {"loss": mag + phs, 'mag': mag.detach(), "phs": phs.detach()}
        else:
            raise NotImplementedError(f"no default loss function for coordinate '{self.coordinate}'")
