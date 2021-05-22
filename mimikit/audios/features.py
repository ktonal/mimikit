import torchaudio.functional as F
import torchaudio.transforms as T
import torch
import numpy as np
from abc import ABC
import librosa
import dataclasses as dtc
import IPython.display as ipd
import soundfile as sf
from functools import partial

from ..audios import transforms as A
from ..h5data.write import write_feature


class Feature(ABC):
    @staticmethod
    def extract(path, **kwargs):
        pass

    @staticmethod
    def after_make(db):
        pass

    @staticmethod
    def encode(inputs: torch.Tensor, **kwargs):
        pass

    @staticmethod
    def decode(outputs: torch.Tensor, **kwargs):
        pass


class QuantizedSignal(Feature):

    @staticmethod
    def extract(path, sr=16000, q_levels=255, emphasis=0.):
        signal = A.FileTo.signal(path, sr)
        if emphasis:
            signal = A.emphasize(signal, emphasis)
        signal = A.normalize(signal)
        signal = A.SignalTo.mu_law_compress(signal, q_levels=q_levels)
        return dict(qx=(dict(sr=sr, q_levels=q_levels, emphasis=emphasis),
                        signal.reshape(-1, 1), None))

    @staticmethod
    def encode(inputs: torch.Tensor, q_levels=256, emphasis=0.):
        if emphasis:
            inputs = F.lfilter(inputs,
                               torch.tensor([1, 0]).to(inputs),  # a0, a1
                               torch.tensor([1, -emphasis]).to(inputs))  # b0, b1
        inputs = inputs / torch.norm(inputs, p=float("inf"))
        return F.mu_law_encoding(inputs, q_levels)

    @staticmethod
    def decode(outputs: torch.Tensor, q_levels=256, emphasis=0.):
        signal = F.mu_law_decoding(outputs, q_levels)
        if emphasis:
            signal = F.lfilter(signal,
                               torch.tensor([1, -emphasis]).to(signal),  # a0, a1
                               torch.tensor([1 - emphasis, 0]).to(signal))  # b0, b1
        return signal


class MagSpec(Feature):

    @staticmethod
    def extract(path, n_fft=2048, hop_length=512, sr=22050):
        y = A.FileTo.signal(path, sr)
        y = A.normalize(y)
        fft = A.SignalTo.mag_spec(y, n_fft, hop_length)
        params = dict(n_fft=n_fft, hop_length=hop_length, sr=sr)
        return dict(fft=(params, fft.T, None))

    @staticmethod
    def encode(inputs: torch.Tensor, n_fft=2048, hop_length=512):
        stft = T.Spectrogram(n_fft=n_fft, hop_length=hop_length, power=1.,
                             wkwargs=dict(device=inputs.device))
        return stft(inputs).transpose(-1, -2).contiguous()

    @staticmethod
    def decode(outputs: torch.Tensor, n_fft=2048, hop_length=512):
        gla = T.GriffinLim(n_fft=n_fft, hop_length=hop_length, power=1.,
                           wkwargs=dict(device=outputs.device))
        return gla(outputs.transpose(-1, -2).contiguous())


class SegmentLabels(Feature):

    @staticmethod
    def extract(path, n_fft=2048, hop_length=512, sr=22050):
        feat_dict = MagSpec.extract(path, n_fft, hop_length, sr)
        regions = A.MagSpecTo.regions(feat_dict["fft"][1].T)
        return dict(fft=(dict(n_fft=n_fft, hop_length=hop_length, sr=sr),
                         feat_dict["fft"][1],
                         regions))

    @staticmethod
    def after_make(db):
        labels = np.hstack([np.ones((tp.duration,), dtype=np.int) * tp.Index
                            for tp in db.fft.regions.itertuples()])
        write_feature(db.h5_file,
                      "labels", dict(n_classes=len(db.fft.regions)), labels)
        files_labels = np.hstack([np.ones((tp.duration,), dtype=np.int) * tp.Index
                                  for tp in db.fft.files.itertuples()])
        write_feature(db.h5_file,
                      "files_labels", dict(n_classes=len(db.fft.files)), files_labels)


@dtc.dataclass
class AudioSignal:
    __ext__ = 'audio'

    sr: int = 22050
    normalize: bool = True
    emphasis: float = 0.

    @property
    def params(self):
        return dtc.asdict(self)

    @property
    def encoders(self):
        def np_encode(inputs):
            if self.emphasis:
                inputs = A.emphasize(inputs, self.emphasis)
            if self.normalize:
                inputs = A.normalize(inputs)
            return inputs

        def torch_encode(inputs):
            if self.emphasis:
                inputs = F.lfilter(inputs,
                                   torch.tensor([1, 0]).to(inputs),  # a0, a1
                                   torch.tensor([1, -self.emphasis]).to(inputs))  # b0, b1
            if self.normalize:
                inputs = inputs / torch.norm(inputs, p=float("inf"))
            return inputs

        return {
            np.ndarray: np_encode,
            torch.Tensor: torch_encode
        }

    @property
    def decoders(self):
        def np_decode(inputs):
            if self.emphasis:
                inputs = A.deemphasize(inputs, self.emphasis)
            if self.normalize:
                inputs = A.normalize(inputs)
            return inputs

        def torch_decode(inputs):
            if self.emphasis:
                inputs = F.lfilter(inputs,
                                   torch.tensor([1, -self.emphasis]).to(inputs),  # a0, a1
                                   torch.tensor([1 - self.emphasis, 0]).to(inputs))  # b0, b1
            if self.normalize:
                inputs = inputs / torch.norm(inputs, p=float("inf"))
            return inputs

        return {
            np.ndarray: np_decode,
            torch.Tensor: torch_decode
        }

    def load(self, path):
        y = A.FileTo.signal(path, self.sr)
        y = self.encode(y)
        return y

    def display(self, inputs, decode=True, **waveplot_kwargs):
        if decode:
            inputs = self.decode(inputs)
        waveplot_kwargs.setdefault('sr', self.sr)
        librosa.display.waveplot(inputs, **waveplot_kwargs)

    def play(self, inputs, decode=True):
        if decode:
            inputs = self.decode(inputs)
        ipd.display(ipd.Audio(inputs, rate=self.sr))

    def write(self, filename, inputs, decode=True):
        if decode:
            inputs = self.decode(inputs)
        sf.write(filename, inputs, self.sr, 'PCM_24')

    def encode(self, inputs):
        return self.encoders[type(inputs)](inputs)

    def decode(self, inputs):
        return self.decoders[type(inputs)](inputs)


@dtc.dataclass
class MuLawSignal(AudioSignal):
    q_levels: int = 256

    @property
    def encoders(self):
        return {
            np.ndarray: partial(A.SignalTo.mu_law_compress, q_levels=self.q_levels),
            torch.Tensor: partial(F.mu_law_encoding, quantization_channels=self.q_levels)
        }

    @property
    def decoders(self):
        return {
            np.ndarray: partial(A.SignalFrom.mu_law_compressed, q_levels=self.q_levels),
            torch.Tensor: partial(F.mu_law_decoding, quantization_channels=self.q_levels)
        }

    def encode(self, inputs):
        return self.encoders[type(inputs)](inputs)

    def decode(self, inputs):
        return self.decoders[type(inputs)](inputs)

    def load(self, path):
        y = super().load(path)  # apply emphasis etc.
        return self.encode(y)

    def display(self, inputs, decode=True, **waveplot_kwargs):
        if decode:
            inputs = self.decode(inputs)
        super().display(inputs, decode, **waveplot_kwargs)

    def play(self, inputs, decode=True):
        if decode:
            inputs = self.decode(inputs)
        super().play(inputs, decode)

    def write(self, filename, inputs, decode=True):
        if decode:
            inputs = self.decode(inputs)
        super(MuLawSignal, self).write(filename, inputs, decode)
