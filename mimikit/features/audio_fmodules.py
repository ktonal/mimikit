import librosa
import torch
import torchaudio.functional as F
import torchaudio.transforms as T
import numpy as np
from scipy.signal import lfilter
import dataclasses as dtc
import abc


__all__ = [
    'FileToSignal',
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
    'GLA'
]

N_FFT = 2048
HOP_LENGTH = 512
SR = 22050
Q_LEVELS = 256


class FModule(abc.ABC):

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


@dtc.dataclass
class FileToSignal:

    sr: int = SR

    def __call__(self, path):
        return librosa.load(path, sr=self.sr, mono=True, res_type='soxr_vhq')[0]


@dtc.dataclass
class Normalize(FModule):
    p: float = float('inf')
    dim: int = -1

    def np_func(self, inputs):
        return librosa.util.normalize(inputs, norm=self.p, axis=self.dim)

    def torch_func(self, inputs):
        return torch.nn.functional.normalize(inputs, p=self.p, dim=self.dim)


@dtc.dataclass
class Emphasis(FModule):
    emphasis: float = 0.

    def np_func(self, inputs):
        return lfilter([1, -self.emphasis], [1], inputs)

    def torch_func(self, inputs):
        return F.lfilter(inputs,
                         torch.tensor([1, 0]).to(inputs),  # a0, a1
                         torch.tensor([1, -self.emphasis]).to(inputs))  # b0, b1


@dtc.dataclass
class Deemphasis(FModule):
    emphasis: float = 0.

    def np_func(self, inputs):
        return lfilter([1 - self.emphasis], [1, -self.emphasis], inputs)

    def torch_func(self, inputs):
        return F.lfilter(inputs,
                         torch.tensor([1, - self.emphasis]).to(inputs),  # a0, a1
                         torch.tensor([1 - self.emphasis, 0]).to(inputs))  # b0, b1


@dtc.dataclass
class Resample(FModule):
    orig_sr: int = 22050
    target_sr: int = 16000

    def np_func(self, inputs):
        return librosa.resample(inputs, orig_sr=self.orig_sr, target_sr=self.target_sr,
                                res_type='soxr_vhq')

    def torch_func(self, inputs):
        return F.resample(inputs, self.orig_sr, self.target_sr)


@dtc.dataclass
class MuLawCompress(FModule):
    q_levels: int = Q_LEVELS

    def np_func(self, inputs):
        if np.any(inputs < -1) or np.any(inputs > 1):
            inputs = Normalize()(inputs)
        qx = librosa.mu_compress(inputs, mu=self.q_levels - 1, quantize=True)
        qx = qx + self.q_levels // 2
        return qx

    def torch_func(self, inputs):
        # there are inconsistencies between librosa and torchaudio for MuLaw Stuff...
        # return torch.from_numpy(np_func(inputs.detach().cpu().numpy())).to(inputs.device)
        return F.mu_law_encoding(inputs, self.q_levels)


@dtc.dataclass
class MuLawExpand(FModule):
    q_levels: int = Q_LEVELS

    def np_func(self, inputs):
        return librosa.mu_expand(inputs - self.q_levels // 2, mu=self.q_levels - 1, quantize=True)

    def torch_func(self, inputs):
        # there are inconsistencies between librosa and torchaudio for MuLaw Stuff...
        # return torch.from_numpy(np_func(inputs.detach().cpu().numpy())).to(inputs.device)
        return F.mu_law_decoding(inputs, self.q_levels)


def _quantize_np(x_comp, q):
    return (
            np.digitize(
                x_comp, np.linspace(-1, 1, num=q, endpoint=True), right=True
            ))


def _linearize_np(x, mu):
    return x * 2.0 / mu


def alaw_compress(x, A=87.6):
    mask = np.abs(x) < (1/A)
    y = np.sign(x)
    y[mask] *= (A*np.abs(x[mask])) / (1+np.log(A))
    y[~mask] *= (1 + np.log(A) * np.abs(x[~mask])) / (1 + np.log(A))
    return y


def alaw_expand(y, A=87.6):
    x = np.sign(y)
    ln_A = (1 + np.log(A))
    mask = np.abs(y) < (1 / ln_A)
    x[mask] *= (np.abs(y[mask]) * ln_A) / A
    x[~mask] *= np.exp(-1+np.abs(y[~mask])*ln_A) / A
    return x


@dtc.dataclass
class ALawCompress(FModule):
    A: float = 87.6
    q_levels: int = Q_LEVELS

    def np_func(self, inputs):
        if np.any(inputs < -1) or np.any(inputs > 1):
            inputs = Normalize()(inputs)
        qx = alaw_compress(inputs, A=self.A)
        qx = _quantize_np(qx, self.q_levels)
        return qx

    def torch_func(self, inputs):
        # there are inconsistencies between librosa and torchaudio for MuLaw Stuff...
        return torch.from_numpy(self.np_func(inputs.detach().cpu().numpy())).to(inputs.device)


@dtc.dataclass
class ALawExpand(FModule):
    A: float = 87.6
    q_levels: int = Q_LEVELS

    def np_func(self, inputs):
        return alaw_expand(_linearize_np(inputs, self.q_levels), A=self.A)

    def torch_func(self, inputs):
        # there are inconsistencies between librosa and torchaudio for MuLaw Stuff...
        return torch.from_numpy(self.np_func(inputs.detach().cpu().numpy())).to(inputs.device)


@dtc.dataclass
class STFT(FModule):
    n_fft: int = N_FFT
    hop_length: int = HOP_LENGTH
    coordinate: str = 'pol'
    center: bool = True

    def np_func(self, inputs):
        # returned shape is (time x freq)
        S = librosa.stft(inputs, n_fft=self.n_fft, hop_length=self.hop_length,
                         center=self.center,).T
        if self.coordinate == 'pol':
            S = np.stack((abs(S), np.angle(S)), axis=-1)
        elif self.coordinate == 'car':
            S = np.stack(S.real, S.imag, axis=-1)
        return S

    def torch_func(self, inputs):
        S = torch.stft(inputs, self.n_fft, hop_length=self.hop_length, return_complex=True,
                       center=self.center,
                       window=torch.hann_window(self.n_fft, device=inputs.device)).transpose(-1, -2).contiguous()
        if self.coordinate == 'pol':
            S = torch.stack((abs(S), torch.angle(S)), dim=-1)
        elif self.coordinate == 'car':
            S = torch.stack((S.real, S.imag), dim=-1)
        return S


@dtc.dataclass
class ISTFT(FModule):
    n_fft: int = N_FFT
    hop_length: int = HOP_LENGTH
    coordinate: str = 'pol'
    center: bool = True

    def np_func(self, inputs):
        # inputs is of shape (time x freq)
        if self.coordinate == 'pol':
            inputs = inputs[..., 0] * np.exp(1j * inputs[..., 1])
        elif self.coordinate == 'car':
            inputs = inputs[..., 0] * (1j * inputs[..., 1])
        y = librosa.istft(inputs.T, n_fft=self.n_fft, hop_length=self.hop_length, center=self.center)
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


@dtc.dataclass
class MagSpec(STFT):

    @property
    def functions(self):
        sup_f = super(MagSpec, self).functions
        # dict comprehension would result in a single function for
        # all types, so we declare the dict manually...
        return {
            np.ndarray: lambda x: abs(sup_f[np.ndarray](x)[..., 0]),
            torch.Tensor: lambda x: abs(sup_f[torch.Tensor](x)[..., 0])
        }


@dtc.dataclass
class GLA(FModule):
    n_fft: int = N_FFT
    hop_length: int = HOP_LENGTH
    n_iter: int = 32
    center: bool = True

    def np_func(self, inputs):
        # inputs is of shape (time x freq)
        return librosa.griffinlim(inputs.T, hop_length=self.hop_length, n_iter=self.n_iter, center=self.center)

    def torch_func(self, inputs):
        # TODO : pull request for center=False support?
        gla = T.GriffinLim(n_fft=self.n_fft, hop_length=self.hop_length, power=1.,
                           wkwargs=dict(device=inputs.device))
        # inputs is of shape (time x freq)
        return gla(inputs.transpose(-1, -2).contiguous())

