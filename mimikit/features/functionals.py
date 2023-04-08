from typing import Optional, Tuple, Union

import librosa
import torch
import torchaudio.functional as F
import torchaudio.transforms as T
import numpy as np
from scipy.signal import lfilter
from scipy.interpolate import interp1d
from sklearn.decomposition import PCA as skPCA,\
    FactorAnalysis as skFactorAnalysis, NMF as skNMF
from sklearn.preprocessing import StandardScaler
from numba import njit, prange, float32, intp
import dataclasses as dtc
import abc

from .item_spec import Sample, Frame, Unit
from ..config import Config

__all__ = [
    'Continuous',
    'Discrete',
    'Functional',
    'Identity',
    'get_metadata',
    'Compose',
    'FileToSignal',
    'RemoveDC',
    'Normalize',
    'Emphasis',
    'Deemphasis',
    'Resample',
    'MuLawCompress',
    'MuLawExpand',
    'ALawCompress',
    'ALawExpand',
    'STFT',
    'ISTFT',
    'MagSpec',
    'GLA',
    "MelSpec",
    "MFCC",
    "Chroma",
    "HarmonicSource",
    "PercussiveSource",
    'Envelop',
    'EnvelopBank',
    'Interpolate',
    'derivative_np',
    'derivative_torch',
    'Derivative',
    "AutoConvolve",
    "F0Filter",
    "NearestNeighborFilter",
    "PCA",
    "NMF",
    "FactorAnalysis",
]

N_FFT = 2048
HOP_LENGTH = 512
SR = 22050
Q_LEVELS = 256


@dtc.dataclass
class Continuous:
    min_value: Union[float, int]
    max_value: Union[float, int]
    size: int


@dtc.dataclass
class Discrete:
    size: int


EventType = Union[Continuous, Discrete]


@dtc.dataclass
class Functional(abc.ABC, Config):

    @property
    def unit(self) -> Optional[Unit]:
        """output's time unit"""
        return None

    @property
    def elem_type(self) -> Optional[EventType]:
        return None

    @abc.abstractmethod
    def np_func(self, inputs):
        raise NotImplementedError

    @abc.abstractmethod
    def torch_func(self, inputs):
        raise NotImplementedError

    @property
    def functions(self) -> dict:
        return {np.ndarray: self.np_func, torch.Tensor: self.torch_func}

    def __call__(self, inputs):
        return self.functions[type(inputs)](inputs)

    @property
    @abc.abstractmethod
    def inv(self) -> "Functional":
        ...


@dtc.dataclass
class Identity(Functional):

    def np_func(self, inputs):
        return inputs

    def torch_func(self, inputs):
        return inputs

    @property
    def inv(self) -> "Functional":
        return Identity()


def _to_dict(value):
    return {} if value is None else value


def _add_metadata(x: Union[np.ndarray, torch.Tensor], **metadata):
    if isinstance(x, np.ndarray):
        prev = _to_dict(x.dtype.metadata)
        prev.update(metadata)
        dtype = np.dtype(x.dtype, metadata=prev)
        return x.view(dtype)
    for k, v in metadata.items():
        setattr(x, k, v)
    return x


def get_metadata(x, key: str, default=None):
    if isinstance(x, np.ndarray):
        meta = _to_dict(x.dtype.metadata)
        return meta.get(key, default)
    return getattr(x, key, default)


@dtc.dataclass
class FileToSignal(Functional):
    sr: int = SR
    offset: float = 0.
    duration: Optional[float] = None

    @property
    def unit(self) -> Optional[Unit]:
        return Sample(self.sr)

    @property
    def elem_type(self) -> Optional[EventType]:
        return Continuous(-float("inf"), float("inf"), 1)

    def np_func(self, path):
        y = librosa.load(path, sr=self.sr,
                         offset=self.offset,
                         duration=self.duration,
                         mono=True, res_type='soxr_vhq')[0]
        return _add_metadata(y, sr=self.sr)

    def torch_func(self, path):
        return torch.from_numpy(self.np_func(path))

    def __call__(self, path):
        return self.np_func(path)

    @property
    def inv(self):
        return Identity()


@dtc.dataclass
class Compose(Functional):
    functionals: Tuple[Functional, ...]

    def __init__(self, *funcs: Functional, functionals=()):
        self.functionals = funcs or functionals

    @property
    def unit(self) -> Optional[Unit]:
        u = tuple(f.unit for f in self.functionals if f.unit is not None)
        return u[-1] if any(u) else None

    @property
    def elem_type(self) -> Optional[EventType]:
        ev = tuple(f.elem_type for f in self.functionals if f.elem_type is not None)
        return ev[-1] if any(ev) else None

    def np_func(self, inputs):
        raise NotImplementedError

    def torch_func(self, inputs):
        raise NotImplementedError

    def __call__(self, inputs):
        x = inputs
        for f in self.functionals:
            x = f(x)
        return x

    @property
    def inv(self):
        return Compose(*(f.inv for f in reversed(self.functionals)))


@dtc.dataclass
class RemoveDC(Functional):

    @property
    def elem_type(self) -> Optional[EventType]:
        return None

    def np_func(self, inputs):
        return lfilter([1.0, -1.0], [1.0, -0.99], inputs, axis=-1).astype(inputs.dtype)

    def torch_func(self, inputs):
        return F.lfilter(torch.tensor([1.0, -0.99]).to(inputs),
                         torch.tensor([1.0, -1.0]).to(inputs),
                         inputs, clamp=False).view(inputs.dtype)

    @property
    def inv(self) -> "Functional":
        return Identity()


@dtc.dataclass
class Normalize(Functional):
    p: float = float('inf')
    dim: int = -1

    @property
    def elem_type(self) -> Optional[EventType]:
        return Continuous(-1., 1., 1)

    def np_func(self, inputs):
        return librosa.util.normalize(inputs, norm=self.p, axis=self.dim).astype(inputs.dtype)

    def torch_func(self, inputs):
        return torch.nn.functional.normalize(inputs, p=self.p, dim=self.dim)

    @property
    def inv(self):
        return Identity()


@dtc.dataclass
class Emphasis(Functional):
    emphasis: float = 0.

    def np_func(self, inputs):
        return lfilter([1, -self.emphasis], [1], inputs).astype(inputs.dtype)

    def torch_func(self, inputs):
        return F.lfilter(inputs,
                         torch.tensor([1, 0]).to(inputs),  # a0, a1
                         torch.tensor([1, -self.emphasis]).to(inputs),  # b0, b1
                         clamp=False)

    @property
    def inv(self):
        return Deemphasis(self.emphasis)


@dtc.dataclass
class Deemphasis(Functional):
    emphasis: float = 0.

    def np_func(self, inputs):
        return lfilter([1 - self.emphasis], [1, -self.emphasis], inputs).astype(inputs.dtype)

    def torch_func(self, inputs):
        return F.lfilter(inputs,
                         torch.tensor([1, - self.emphasis]).to(inputs),  # a0, a1
                         torch.tensor([1 - self.emphasis, 0]).to(inputs), clamp=False)  # b0, b1

    @property
    def inv(self):
        return Emphasis(self.emphasis)


@dtc.dataclass
class Resample(Functional):
    orig_sr: int = 22050
    target_sr: int = 16000

    @property
    def unit(self) -> Optional[Unit]:
        return Sample(self.target_sr)

    def np_func(self, inputs):
        y = librosa.resample(inputs, orig_sr=self.orig_sr, target_sr=self.target_sr,
                             res_type='soxr_vhq')
        return _add_metadata(y, sr=self.target_sr)

    def torch_func(self, inputs):
        return F.resample(inputs, self.orig_sr, self.target_sr)

    @property
    def inv(self):
        return Resample(self.target_sr, self.orig_sr)


@dtc.dataclass
class MuLawCompress(Functional):
    q_levels: int = Q_LEVELS
    compression: float = 1.

    @property
    def elem_type(self) -> Optional[EventType]:
        return Discrete(self.q_levels)

    def np_func(self, inputs):
        # librosa's mu_compress is not correctly centered...
        mu = self.q_levels - 1.0
        x_mu = np.sign(inputs) * np.log1p(mu * np.abs(inputs) * self.compression) \
               / np.log1p(mu * self.compression)
        x_mu = ((x_mu + 1) / 2 * mu + 0.5).astype(np.int64)
        return _add_metadata(x_mu, **_to_dict(inputs.dtype.metadata))

    def torch_func(self, inputs):
        mu = self.q_levels - 1.0
        if not inputs.is_floating_point():
            inputs = inputs.to(torch.float)
        mu = torch.tensor(mu, dtype=inputs.dtype)
        C = torch.tensor(self.compression, dtype=inputs.dtype)
        x_mu = torch.sign(inputs) * torch.log1p(mu * torch.abs(inputs) * C) / torch.log1p(mu * C)
        x_mu = ((x_mu + 1) / 2 * mu + 0.5).to(torch.int64)
        return x_mu

    @property
    def inv(self):
        return MuLawExpand(self.q_levels, self.compression)


@dtc.dataclass
class MuLawExpand(Functional):
    q_levels: int = Q_LEVELS
    compression: float = 1.

    @property
    def elem_type(self) -> Optional[EventType]:
        return Continuous(-1., 1., 1)

    def np_func(self, inputs):
        mu = self.q_levels - 1.0
        x = (inputs / mu) * 2 - 1.0
        x = np.sign(x) * (np.exp(np.abs(x) * np.log1p(mu * self.compression)) - 1.0) / \
            (mu * self.compression)
        return _add_metadata(x, **_to_dict(inputs.dtype.metadata))

    def torch_func(self, inputs):
        mu = self.q_levels - 1.0
        if not inputs.is_floating_point():
            inputs = inputs.to(torch.float)
        mu = torch.tensor(mu, dtype=inputs.dtype)
        C = torch.tensor(self.compression, dtype=inputs.dtype)
        x = (inputs / mu) * 2 - 1.0
        x = torch.sign(x) * (torch.exp(torch.abs(x) * torch.log1p(mu * C)) - 1.0) / (mu * C)
        return x

    @property
    def inv(self):
        return MuLawCompress(self.q_levels, self.compression)


def _quantize_np(x_comp, q):
    return (
        np.digitize(
            x_comp, np.linspace(-1, 1, num=q, endpoint=True), right=True
        ))


def _linearize_np(x, mu):
    return x * 2.0 / mu


def alaw_compress(x, A=87.6):
    mask = np.abs(x) < (1 / A)
    y = np.sign(x)
    y[mask] *= (A * np.abs(x[mask])) / (1 + np.log(A))
    y[~mask] *= (1 + np.log(A) * np.abs(x[~mask])) / (1 + np.log(A))
    return y


def alaw_expand(y, A=87.6):
    x = np.sign(y)
    ln_A = (1 + np.log(A))
    mask = np.abs(y) < (1 / ln_A)
    x[mask] *= (np.abs(y[mask]) * ln_A) / A
    x[~mask] *= np.exp(-1 + np.abs(y[~mask]) * ln_A) / A
    return x


@dtc.dataclass
class ALawCompress(Functional):
    A: float = 87.6
    q_levels: int = Q_LEVELS

    @property
    def elem_type(self) -> Optional[EventType]:
        return Discrete(self.q_levels)

    def np_func(self, inputs):
        if np.any(inputs < -1) or np.any(inputs > 1):
            inputs = Normalize()(inputs)
        qx = alaw_compress(inputs, A=self.A)
        qx = _quantize_np(qx, self.q_levels)
        return qx

    def torch_func(self, inputs):
        # there are inconsistencies between librosa and torchaudio for MuLaw Stuff...
        return torch.from_numpy(self.np_func(inputs.detach().cpu().numpy())).to(inputs.device)

    @property
    def inv(self):
        return ALawExpand(self.A, self.q_levels)


@dtc.dataclass
class ALawExpand(Functional):
    A: float = 87.6
    q_levels: int = Q_LEVELS

    @property
    def elem_type(self) -> Optional[EventType]:
        return Continuous(-1., 1., 1)

    def np_func(self, inputs):
        return alaw_expand(_linearize_np(inputs, self.q_levels), A=self.A)

    def torch_func(self, inputs):
        # there are inconsistencies between librosa and torchaudio for MuLaw Stuff...
        return torch.from_numpy(self.np_func(inputs.detach().cpu().numpy())).to(inputs.device)

    @property
    def inv(self):
        return ALawCompress(self.A, self.q_levels)


@dtc.dataclass
class STFT(Functional):
    n_fft: int = N_FFT
    hop_length: int = HOP_LENGTH
    coordinate: str = 'pol'
    center: bool = True
    window: Optional[str] = "hann"
    pad_mode: str = "constant"

    @property
    def unit(self) -> Optional[Unit]:
        return Frame(self.n_fft, self.hop_length, padding=self.center)

    @property
    def elem_type(self) -> Optional[EventType]:
        return Continuous(0., float("inf"), 1 + self.n_fft // 2)

    def np_func(self, inputs):
        # returned shape is (time x freq)
        S = librosa.stft(inputs, n_fft=self.n_fft, hop_length=self.hop_length,
                         center=self.center,
                         window=self.window if self.window is not None else 1.,
                         pad_mode=self.pad_mode
                         ).transpose(-1, -2)
        if self.coordinate == 'pol':
            S = np.stack((abs(S), np.angle(S)), axis=-1)
        elif self.coordinate == 'car':
            S = np.stack(S.real, S.imag, axis=-1)
        elif self.coordinate == 'mag':
            S = abs(S)
        elif self.coordinate == 'angle':
            S = np.angle(S)
        S = _add_metadata(S, n_samples=inputs.shape[0], **_to_dict(inputs.dtype.metadata))
        return S

    def torch_func(self, inputs):
        S = torch.stft(inputs, self.n_fft, hop_length=self.hop_length, return_complex=True,
                       center=self.center,
                       window=torch.hann_window(self.n_fft, device=inputs.device),
                       pad_mode=self.pad_mode)
        S = S.transpose(-1, -2).contiguous()
        if self.coordinate == 'pol':
            S = torch.stack((abs(S), torch.angle(S)), dim=-1)
        elif self.coordinate == 'car':
            S = torch.stack((S.real, S.imag), dim=-1)
        elif self.coordinate == 'mag':
            S = S.abs()
        elif self.coordinate == 'angle':
            S = torch.angle(S)
        return S

    @property
    def inv(self):
        return ISTFT(self.n_fft, self.hop_length, self.coordinate, self.center, self.window)


@dtc.dataclass
class ISTFT(Functional):
    n_fft: int = N_FFT
    hop_length: int = HOP_LENGTH
    coordinate: str = 'pol'
    center: bool = True
    window: Optional[str] = None
    pad_mode: str = "constant"

    @property
    def unit(self) -> Optional[Unit]:
        return Sample(None)

    @property
    def elem_type(self) -> Optional[EventType]:
        return Continuous(-1., 1., 1)

    def np_func(self, inputs):
        # inputs is of shape (time x freq)
        if self.coordinate == 'pol':
            inputs = inputs[..., 0] * np.exp(1j * inputs[..., 1])
        elif self.coordinate == 'car':
            inputs = inputs[..., 0] * (1j * inputs[..., 1])
        y = librosa.istft(inputs.T, n_fft=self.n_fft, hop_length=self.hop_length, center=self.center,
                          window=self.window if self.window is not None else 1.)
        return y

    def torch_func(self, inputs):
        if self.coordinate == 'pol':
            inputs = inputs[..., 0] * torch.exp(1j * inputs[..., 1])
        elif self.coordinate == 'car':
            inputs = inputs[..., 0] * (1j * inputs[..., 1])
        y = torch.istft(inputs.transpose(1, 2).contiguous(),
                        n_fft=self.n_fft, hop_length=self.hop_length,
                        # center=self.center,
                        window=torch.hann_window(self.n_fft, device=inputs.device))
        return y

    @property
    def inv(self):
        return STFT(self.n_fft, self.hop_length, self.coordinate, self.center, self.window, self.pad_mode)


@dtc.dataclass
class MagSpec(Functional):
    n_fft: int = N_FFT
    hop_length: int = HOP_LENGTH
    center: bool = True
    window: Optional[str] = "hann"
    pad_mode: str = "constant"

    @property
    def unit(self) -> Optional[Unit]:
        return Frame(self.n_fft, self.hop_length, padding=self.center)

    @property
    def elem_type(self) -> Optional[EventType]:
        return Continuous(0., float("inf"), 1 + self.n_fft // 2)

    def __post_init__(self):
        self.stft = STFT(self.n_fft, self.hop_length, "mag",
                         self.center, self.window, self.pad_mode)

    def np_func(self, inputs):
        return self.stft.np_func(inputs)

    def torch_func(self, inputs):
        return self.stft.torch_func(inputs)

    @property
    def functions(self):
        return self.stft.functions

    @property
    def inv(self):
        return GLA(self.n_fft, self.hop_length, self.center, self.window, self.pad_mode)


@dtc.dataclass
class GLA(Functional):
    n_fft: int = N_FFT
    hop_length: int = HOP_LENGTH
    center: bool = True
    window: Optional[str] = None
    pad_mode: str = "constant"
    n_iter: int = 32

    @property
    def unit(self) -> Optional[Unit]:
        return Sample(None)

    @property
    def elem_type(self) -> Optional[EventType]:
        return Continuous(-1., 1., 1)

    def np_func(self, inputs):
        # inputs is of shape (time x freq)
        return librosa.griffinlim(inputs.T, hop_length=self.hop_length, n_iter=self.n_iter, center=self.center)

    def torch_func(self, inputs):
        # TODO : pull request for center=False support?
        gla = T.GriffinLim(n_fft=self.n_fft, hop_length=self.hop_length, power=1.,
                           wkwargs=dict(device=inputs.device))
        # inputs is of shape (time x freq)
        return gla(inputs.transpose(-1, -2).contiguous())

    @property
    def inv(self):
        return MagSpec(self.n_fft, self.hop_length, self.center, self.window, self.pad_mode)


@dtc.dataclass
class MelSpec(Functional):
    """expects a MagSpec as inputs"""
    n_mels: int = 128
    fmin: float = 0.
    fmax: Optional[float] = None
    htk: bool = False

    @property
    def unit(self) -> Optional[Unit]:
        return None

    @property
    def elem_type(self) -> Optional[EventType]:
        return Continuous(0., float("inf"), self.n_mels)

    def np_func(self, inputs):
        return librosa.feature.melspectrogram(
            S=inputs.T, n_mels=self.n_mels, fmin=self.fmin, fmax=self.fmax, htk=self.htk
        ).T

    def torch_func(self, inputs):
        # TODO
        pass

    @property
    def inv(self) -> "Functional":
        return Identity()


@dtc.dataclass
class MFCC(Functional):
    """expects a MelSpec as inputs"""
    n_mfcc: int = 20
    dct_type: int = 2
    norm: Optional[str] = "ortho"
    lifter: int = 0

    @property
    def unit(self) -> Optional[Unit]:
        return None

    @property
    def elem_type(self) -> Optional[EventType]:
        return Continuous(0., float("inf"), self.n_mfcc)

    def np_func(self, inputs):
        return librosa.feature.mfcc(
            S=inputs.T, n_mfcc=self.n_mfcc, dct_type=self.dct_type,
            norm=self.norm, lifter=self.lifter
        ).T

    def torch_func(self, inputs):
        # TODO
        pass

    @property
    def inv(self) -> "Functional":
        return Identity()


@dtc.dataclass
class Chroma(Functional):
    n_chroma: int = 12

    @property
    def unit(self) -> Optional[Unit]:
        return None

    @property
    def elem_type(self) -> Optional[EventType]:
        return Continuous(0., float("inf"), self.n_chroma)

    def np_func(self, inputs):
        return librosa.feature.chroma_stft(
            S=inputs.T, n_chroma=self.n_chroma
        ).T

    def torch_func(self, inputs):
        # TODO
        pass

    @property
    def inv(self) -> "Functional":
        return Identity()


@dtc.dataclass
class HarmonicSource(Functional):
    kernel_size: int = 31
    power: float = 1.
    margin: float = 1.

    @property
    def unit(self) -> Optional[Unit]:
        return None

    @property
    def elem_type(self) -> Optional[EventType]:
        return None

    def np_func(self, inputs):
        return librosa.decompose.hpss(
            S=inputs.T, kernel_size=self.kernel_size,
            power=self.power, margin=self.margin
        )[0].T

    def torch_func(self, inputs):
        # TODO
        pass

    @property
    def inv(self) -> "Functional":
        return Identity()


@dtc.dataclass
class PercussiveSource(Functional):
    kernel_size: int = 31
    power: float = 1.
    margin: float = 1.

    @property
    def unit(self) -> Optional[Unit]:
        return None

    @property
    def elem_type(self) -> Optional[EventType]:
        return None

    def np_func(self, inputs):
        return librosa.decompose.hpss(
            S=inputs.T, kernel_size=self.kernel_size,
            power=self.power, margin=self.margin
        )[1].T

    def torch_func(self, inputs):
        # TODO
        pass

    @property
    def inv(self) -> "Functional":
        return Identity()


@dtc.dataclass
class Envelop(Functional):
    n_fft: int = N_FFT
    hop_length: int = HOP_LENGTH
    normalize: bool = True
    interp_to_time_domain: bool = True

    def __post_init__(self):
        self.fft = MagSpec(self.n_fft, self.hop_length, center=True,
                           window="hann", pad_mode="reflect")

    @property
    def unit(self) -> Optional[Unit]:
        return Sample(None) if self.interp_to_time_domain else self.fft.unit

    @property
    def elem_type(self) -> Optional[EventType]:
        mx = 1. if self.normalize else float('inf')
        return Continuous(0., mx, 1)

    def np_func(self, inputs):
        S = self.fft(inputs)
        e = S.sum(axis=1)
        if self.interp_to_time_domain:
            e = Interpolate(length=inputs.shape[0])(e)
        if self.normalize:
            e /= e.max()
        return e.astype(np.float32)

    def torch_func(self, inputs):
        return torch.from_numpy(self.np_func(inputs.detach().cpu().numpy()))

    @property
    def inv(self):
        return Identity()


@dtc.dataclass
class EnvelopBank(Functional):
    n_fft: Tuple[int] = (N_FFT,)
    hop_length: Tuple[int] = (HOP_LENGTH,)
    normalize: bool = True

    def __post_init__(self):
        # always interp to time domain!
        self.envelops = tuple(
            Envelop(n_fft, hop, self.normalize, True)
            for n_fft, hop in zip(self.n_fft, self.hop_length)
        )

    @property
    def unit(self) -> Optional[Unit]:
        return Sample(None)

    @property
    def elem_type(self) -> Optional[EventType]:
        mx = 1. if self.normalize else float('inf')
        return Continuous(0., mx, len(self.envelops))

    def np_func(self, inputs):
        return np.hstack([e(inputs) for e in self.envelops])

    def torch_func(self, inputs):
        return torch.from_numpy(self.np_func(inputs.detach().cpu().numpy()))

    @property
    def inv(self):
        return Identity()


@dtc.dataclass
class Interpolate(Functional):
    axis: int = -1
    mode: str = 'linear'
    length: Optional[int] = None
    factor: Optional[int] = None
    metadata_key: str = "n_samples"

    @property
    def unit(self) -> Optional[Unit]:
        return None

    @property
    def elem_type(self) -> Optional[EventType]:
        return Continuous(-float("inf"), float("inf"), 1)

    @property
    def inv(self) -> "Functional":
        return Identity()

    def _get_target_length(self, x):
        if self.length is None:
            if self.factor is None:
                N = get_metadata(x, self.metadata_key)
                if N is None:
                    raise ValueError("No target length provided. "
                                     "One of length or factor must not be None,"
                                     f" or inputs must have the metadata key {self.metadata_key}")
            else:
                N = self.factor * x.shape[self.axis]
        else:
            N = self.length
        return N

    def np_func(self, inputs):
        x = inputs
        input_N = x.shape[self.axis]
        xp = np.arange(input_N)
        f = interp1d(xp, x, kind=self.mode, axis=self.axis,
                     assume_sorted=True, copy=False)
        N = self._get_target_length(x)
        return f(np.linspace(0, input_N - 1, N)).astype(x.dtype)

    def torch_func(self, inputs):
        x = inputs
        if self.mode != "linear":
            return torch.from_numpy(self.np_func(x.detach().cpu().numpy())).to(x)
        N = self._get_target_length(x)
        return torch.nn.functional.interpolate(x.view(*((1,) * (3 - x.ndim)), *x.shape),
                                               N, mode="linear").squeeze()


@njit(float32[:](float32[:], intp), fastmath=True, cache=True)
def odd_reflect_pad_1d(x, k):
    """1d version of calling np.pad with **{mode='reflect', reflect_type='odd'}"""
    k_half = k // 2
    y = np.zeros((*x.shape[:-1], x.shape[-1] + k_half * 2),
                 dtype=np.float32)
    y[k_half:-k_half] = x
    y[:k_half] = x[0] + (x[0] - x[1:1 + k_half])[::-1]
    y[-k_half:] = x[-1] + (x[-1] - x[-k_half - 1:-1])[::-1]
    return y


@njit(float32[:](float32[:], intp), fastmath=True, cache=True, parallel=False)
def derivative_np_1d(y, max_lag):
    grads = np.zeros(y.shape, dtype=np.float32)
    for lag in prange(1, max_lag + 1):
        k = lag * 2 + 1
        y_p = odd_reflect_pad_1d(y, k)
        a, b = y_p[:-k + 1], y_p[k - 1:]
        g = (1 / lag) * ((b - y) + (y - a)) / 2
        grads += g / max_lag
    return grads


@njit(float32[:, :](float32[:, :], intp), fastmath=True, cache=True, parallel=True)
def derivative_np_2d(y, max_lag):
    grads = np.zeros(y.shape, dtype=np.float32)
    for i in prange(y.shape[0]):
        grads[i] = derivative_np_1d(y[i], max_lag)
    return grads


def derivative_np(y: np.ndarray, max_lag: int):
    if y.ndim == 1:
        return derivative_np_1d(y, max_lag)
    elif y.ndim == 2:
        return derivative_np_2d(y, max_lag)
    else:
        raise ValueError(f"Expected input array to have 1 or 2 dimensions. Got {y.ndim}")


def derivative_torch(y, max_lag):
    grads = torch.zeros(*y.shape, dtype=torch.float32, device=y.device)
    for n, delay in enumerate(range(1, max_lag + 1)):
        k = delay * 2 + 1
        # odd reflect pad:
        k_half = k // 2
        y_p = torch.zeros(*y.shape[:-1], y.shape[-1] + k - 1, dtype=y.dtype, device=y.device)
        y_p[..., k_half:-k_half] = y
        y_p[..., :k_half] = y[..., 0] + (y[..., 0] - y[..., 1:1 + k_half]).flip(-1)
        y_p[..., -k_half:] = y[..., -1] + (y[..., -1] - y[..., -k_half - 1:-1]).flip(-1)
        ##
        a, b = y_p[..., :-k + 1], y_p[..., k - 1:]
        g = (1 / delay) * ((b - y) + (y - a)) / 2
        grads += g / max_lag
    return grads


@dtc.dataclass
class Derivative(Functional):
    max_lag: int = 3
    normalize: bool = False

    @property
    def unit(self) -> Optional[Unit]:
        return None

    @property
    def elem_type(self) -> Optional[EventType]:
        return Continuous(-float("inf"), float("inf"), 1)

    def np_func(self, inputs):
        g = derivative_np(inputs, self.max_lag)
        if self.normalize:
            g /= abs(g).max(axis=-1, keepdims=True)
        return g

    def torch_func(self, inputs):
        g = derivative_torch(inputs, self.max_lag)
        if self.normalize:
            g /= abs(g).max(dim=-1, keepdims=True)
        return g

    @property
    def inv(self) -> "Functional":
        return Identity()


@dtc.dataclass
class AutoConvolve(Functional):
    window_size: int = 3

    @property
    def unit(self) -> Optional[Unit]:
        return None

    @property
    def elem_type(self) -> Optional[EventType]:
        return None

    def np_func(self, inputs):
        S = inputs
        k = self.window_size
        x_padded = np.pad(S.T, ((0, 0), (k // 2, 0)), constant_values=1)
        x_win = librosa.feature.stack_memory(
            x_padded, n_steps=k, delay=-1, constant_values=1
        )[:, :-(k // 2)].reshape(k, *S.shape[::-1])
        z = np.log(1 + np.prod(x_win.astype(np.float64), axis=0)).T
        z = z / (z.sum(axis=1, keepdims=True) + 1e-8)
        return z * S

    def torch_func(self, inputs):
        # TODO
        pass

    @property
    def inv(self) -> "Functional":
        return Identity()


@dtc.dataclass
class F0Filter(Functional):
    n_overtone: int = 4
    n_undertone: int = 4
    soft: bool = True
    normalize: bool = True

    @property
    def unit(self) -> Optional[Unit]:
        return None

    @property
    def elem_type(self) -> Optional[EventType]:
        return None

    def np_func(self, inputs):
        z = inputs.T
        # sum of overtones above bi
        sl = librosa.interp_harmonics(z, librosa.fft_frequencies(),
                                      list(range(1, self.n_overtone))
                                      ).sum(axis=0)
        # sum of undertones under bi
        sl2 = librosa.interp_harmonics(z, librosa.fft_frequencies(),
                                       [1 / x for x in list(range(2, self.n_undertone))]
                                       ).sum(axis=0)
        y = (sl - sl2)
        if self.soft:
            y = y * (y > 0)
        else:
            y = y > 0

        if self.normalize:
            y = y / (y.sum(axis=0) + 1e-8)
        return inputs * y.T

    def torch_func(self, inputs):
        # TODO
        pass

    @property
    def inv(self) -> "Functional":
        return Identity()


@dtc.dataclass
class NearestNeighborFilter(Functional):
    n_neighbors: int = 16
    metric: str = "cosine"
    aggregate: str = "median"

    @property
    def unit(self) -> Optional[Unit]:
        return None

    @property
    def elem_type(self) -> Optional[EventType]:
        return None

    def np_func(self, inputs):
        return librosa.decompose.nn_filter(
            inputs,
            aggregate=getattr(np, self.aggregate),
            metric=self.metric, sym=True, sparse=True,
            k=self.n_neighbors, axis=0
        )

    def torch_func(self, inputs):
        # TODO
        pass

    @property
    def inv(self) -> "Functional":
        return Identity()


@dtc.dataclass
class PCA(Functional):
    n_components: int = 16
    random_seed: int = 42

    @property
    def unit(self) -> Optional[Unit]:
        return None

    @property
    def elem_type(self) -> Optional[EventType]:
        return None

    def np_func(self, inputs):
        x_h = StandardScaler().fit_transform(inputs)
        return skPCA(n_components=self.n_components, random_state=self.random_seed, copy=False
                     ).fit_transform(x_h)

    def torch_func(self, inputs):
        # TODO
        pass

    @property
    def inv(self) -> "Functional":
        return Identity()


@dtc.dataclass
class NMF(Functional):
    n_components: int = 16
    tol: float = 1e-4
    max_iter: int = 200
    random_seed: int = 42

    @property
    def unit(self) -> Optional[Unit]:
        return None

    @property
    def elem_type(self) -> Optional[EventType]:
        return None

    def np_func(self, inputs):
        return skNMF(
            n_components=self.n_components,
            tol=self.tol,
            max_iter=self.max_iter,
            random_state=self.random_seed
        ).fit_transform(inputs)

    def torch_func(self, inputs):
        # TODO
        pass

    @property
    def inv(self) -> "Functional":
        return Identity()


@dtc.dataclass
class FactorAnalysis(Functional):
    n_components: int = 16
    tol: float = 1e-2
    max_iter: int = 1000
    random_seed: int = 42

    @property
    def unit(self) -> Optional[Unit]:
        return None

    @property
    def elem_type(self) -> Optional[EventType]:
        return None

    def np_func(self, inputs):
        return skFactorAnalysis(
            n_components=self.n_components,
            tol=self.tol,
            max_iter=self.max_iter,
            random_state=self.random_seed,
            copy=False
        ).fit_transform(inputs)

    def torch_func(self, inputs):
        # TODO
        pass

    @property
    def inv(self) -> "Functional":
        return Identity()
