import librosa
import torch
import torchaudio.functional as F
import torchaudio.transforms as T
import numpy as np
from scipy.signal import lfilter
import dataclasses as dtc

__all__ = [
    'FileToSignal',
    'Normalize',
    'Emphasis',
    'Deemphasis',
    'MuLawCompress',
    'MuLawExpand',
    'STFT',
    'ISTFT',
    'MagSpec',
    'GLA'
]

N_FFT = 2048
HOP_LENGTH = 512
SR = 22050
Q_LEVELS = 256


class FModule:
    """
    base class for implementing message passing in callable objects
    """

    @property
    def functions(self) -> dict:
        raise NotImplementedError

    def __call__(self, inputs):
        return self.functions[type(inputs)](inputs)


@dtc.dataclass
class FileToSignal(FModule):
    sr: int = SR

    @property
    def functions(self):
        return {
            str: lambda path: librosa.load(path, sr=self.sr)[0]
        }


@dtc.dataclass
class Normalize(FModule):
    p: int = 1
    dim: int = -1

    @property
    def functions(self):
        def np_func(inputs):
            return librosa.util.normalize(inputs, norm=self.p, axis=self.dim)

        def torch_func(inputs):
            return torch.nn.functional.normalize(inputs, p=self.p, dim=self.dim)

        return {
            np.ndarray: np_func,
            torch.Tensor: torch_func
        }


@dtc.dataclass
class Emphasis(FModule):
    emphasis: float = 0.

    @property
    def functions(self):
        def np_func(inputs):
            return lfilter([1, -self.emphasis], [1], inputs)

        def torch_func(inputs):
            return F.lfilter(inputs,
                             torch.tensor([1, 0]).to(inputs),  # a0, a1
                             torch.tensor([1, -self.emphasis]).to(inputs))  # b0, b1

        return {
            np.ndarray: np_func,
            torch.Tensor: torch_func
        }


@dtc.dataclass
class Deemphasis(FModule):
    emphasis: float = 0.

    @property
    def functions(self):
        def np_func(inputs):
            return lfilter([1 - self.emphasis], [1, -self.emphasis], inputs)

        def torch_func(inputs):
            return F.lfilter(inputs,
                             torch.tensor([1, - self.emphasis]).to(inputs),  # a0, a1
                             torch.tensor([1 - self.emphasis, 0]).to(inputs))  # b0, b1

        return {
            np.ndarray: np_func,
            torch.Tensor: torch_func
        }


@dtc.dataclass
class MuLawCompress(FModule):
    q_levels: int = Q_LEVELS

    @property
    def functions(self):
        def np_func(inputs):
            qx = librosa.mu_compress(inputs, self.q_levels - 1, quantize=True)
            qx = qx + self.q_levels // 2
            return qx

        def torch_func(inputs):
            mod = T.MuLawEncoding(self.q_levels)
            return mod(inputs)

        return {
            np.ndarray: np_func,
            torch.Tensor: torch_func
        }


@dtc.dataclass
class MuLawExpand(FModule):
    q_levels: int = Q_LEVELS

    @property
    def functions(self):
        def np_func(inputs):
            return librosa.mu_expand(inputs - self.q_levels // 2, self.q_levels - 1, quantize=True)

        def torch_func(inputs):
            mod = T.MuLawDecoding(self.q_levels)
            return mod(inputs)

        return {
            np.ndarray: np_func,
            torch.Tensor: torch_func
        }


@dtc.dataclass
class STFT(FModule):
    n_fft: int = N_FFT
    hop_length: int = HOP_LENGTH

    @property
    def functions(self):
        def np_func(inputs):
            # returned shape is (time x freq)
            return librosa.stft(inputs, n_fft=self.n_fft, hop_length=self.hop_length).T

        def torch_func(inputs):
            mod = T.Spectrogram(self.n_fft, hop_length=self.hop_length, power=1.,
                                wkwargs=dict(device=inputs.device))
            # returned shape is (..., time x freq)
            return mod(inputs).transpose(-1, -2).contiguous()

        return {
            np.ndarray: np_func,
            torch.Tensor: torch_func
        }


@dtc.dataclass
class ISTFT(FModule):
    n_fft: int = N_FFT
    hop_length: int = HOP_LENGTH

    @property
    def functions(self):
        def np_func(inputs):
            # inputs is of shape (time x freq)
            return librosa.istft(inputs.T, n_fft=self.n_fft, hop_length=self.hop_length, )

        def torch_func(inputs):
            # inputs is of shape (time x freq)
            y = torch.istft(inputs.transpose(-1, -2).contiguous(),
                            n_fft=self.n_fft, hop_length=self.hop_length,
                            window=torch.hann_window(self.n_fft, device=inputs.device))
            return y

        return {
            np.ndarray: np_func,
            torch.Tensor: torch_func
        }


@dtc.dataclass
class MagSpec(STFT):

    @property
    def functions(self):
        sup_f = super(MagSpec, self).functions
        # dict comprehension would result in a single function for
        # all types, so we declare the dict manually...
        return {
            np.ndarray: lambda x: abs(sup_f[np.ndarray](x)),
            torch.Tensor: lambda x: abs(sup_f[torch.Tensor](x))
        }


@dtc.dataclass
class GLA(FModule):
    n_fft: int = N_FFT
    hop_length: int = HOP_LENGTH
    n_iter: int = 32

    @property
    def functions(self):
        def np_func(inputs):
            # inputs is of shape (time x freq)
            return librosa.griffinlim(inputs.T, hop_length=self.hop_length, n_iter=self.n_iter)

        def torch_func(inputs):
            gla = T.GriffinLim(n_fft=self.n_fft, hop_length=self.hop_length, power=1.,
                               wkwargs=dict(device=inputs.device))
            # inputs is of shape (time x freq)
            return gla(inputs.transpose(-1, -2).contiguous())

        return {
            np.ndarray: np_func,
            torch.Tensor: torch_func
        }
