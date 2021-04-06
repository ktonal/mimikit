import torchaudio.functional as F
import torchaudio.transforms as T
import torch
import numpy as np
from abc import ABC

from ..audios import transforms as A
from ..h5data.write import write_feature
from ..kit.modules.interpolate import Interp1d


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
    def extract(path, sr=16000, q_levels=255, emphasis=0., sample_encoding='mu_law', normalize=True):
        signal = A.FileTo.signal(path, sr)
        if emphasis:
            signal = A.emphasize(signal, emphasis)
        if sample_encoding == 'mu_law':
            shaper = 'mu_law'
            signal = A.SignalTo.mu_law_compress(signal, q_levels=q_levels, normalize=normalize)
        elif sample_encoding == 'adapted':
            signal, shaper = A.SignalTo.adapted_uniform(signal, 
                                                        q_levels=q_levels,
                                                        normalize=normalize)
        elif sample_encoding == 'pcm':
            shaper = 'pcm'
            signal = A.SignalTo.pcm_unsigned(signal, q_levels=q_levels, normalize=normalize)
        else:
            raise ValueError("sample_encoding has to 'mu_law', 'adapted', or 'pcm'")
        return dict(qx=(dict(sr=sr,
                             q_levels=q_levels,
                             emphasis=emphasis,
                             shaper=shaper,
                             sample_encoding=sample_encoding),
                        signal.reshape(-1, 1), None))

    # Maybe normalize should be False by default, since this could be used on small signal snippets
    @staticmethod
    def encode(inputs: torch.Tensor, q_levels=256, emphasis=0., sample_encoding='mu_law', normalize=True, shaper=None):
        if emphasis:
            inputs = F.lfilter(inputs,
                               torch.tensor([1, 0]).to(inputs),  # a0, a1
                               torch.tensor([1, -emphasis]).to(inputs))  # b0, b1
        if normalize:
            inputs = inputs / torch.norm(inputs, p=float("inf"))
        if sample_encoding == 'mu_law':
            return F.mu_law_encoding(inputs, q_levels)
        elif sample_encoding == 'adapted':
            ids = torch.from_numpy(shaper[1]).float()
            # unfortunately Interp1d does not let you select an axis - so process columns one by one
            return ((inputs + 1.0) * 0.5 * (q_levels - 1)).astype(np.int)
        elif sample_encoding == 'pcm':
            return ((inputs + 1.0) * 0.5 * (q_levels - 1)).astype(np.int)
        else:
            raise ValueError("sample_encoding has to 'mu_law', 'adapted', or 'pcm'")

    @staticmethod
    def decode(outputs: torch.Tensor, q_levels=256, emphasis=0., sample_encoding='mu_law', shaper=None):
        if sample_encoding == 'mu_law':
            signal = F.mu_law_decoding(outputs, q_levels)
        elif sample_encoding == 'adapted':
            outputs = outputs.float()
            ids = torch.from_numpy(shaper[1]).float().to(outputs)
            xvals = 2 * ids / ids[-1] - 1.0
            # unfortunately Interp1d does not let you select an axis - so process columns one by one
            signal = torch.stack([Interp1d()(xvals, 
                                             torch.from_numpy(shaper[0]).float().to(outputs), 
                                             2.0 * outputs[:,k] / (q_levels - 1) - 1.0) 
                                  for k in range(outputs.shape[1])]).T
        elif sample_encoding == 'pcm':
            signal = 2.0 * outputs.float() / (q_levels - 1) - 1.0
        else:
            raise ValueError("sample_encoding has to 'mu_law', 'adapted', or 'pcm'")
        if emphasis:
            signal = F.lfilter(signal,
                               torch.tensor([1, -emphasis]).to(signal),  # a0, a1
                               torch.tensor([1 - emphasis, 0]).to(signal))  # b0, b1
        return signal


class MagSpec(Feature):

    @staticmethod
    def extract(path, n_fft=2048, hop_length=512, sr=22050):
        y = A.FileTo.signal(path, sr)
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


class PolarSpec(Feature):

    @staticmethod
    def extract(path, n_fft=2048, hop_length=512, sr=22050, normalize=False):
        y = A.FileTo.signal(path, sr)
        fft = A.SignalTo.polar_spec(y, n_fft, hop_length, normalize)
        params = dict(n_fft=n_fft, hop_length=hop_length, sr=sr)
        return dict(fft=(params, fft.transpose((0,2,1)), None))

    @staticmethod
    def encode(inputs: torch.Tensor, n_fft=2048, hop_length=512):
        stft = T.Spectrogram(n_fft=n_fft, hop_length=hop_length, power=1.,
                             wkwargs=dict(device=inputs.device))
        S = stft(inputs).transpose(-1, -2).contiguous()
        return torch.stack([abs(S), torch.angle(S)], 2)

    # there are multiple ways to decode.  Best might be to use gla with initial phase from the estimates
    @staticmethod
    def decode(outputs: torch.Tensor, n_fft=2048, hop_length=512):
        raise NotImplementedError


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
        labels = np.hstack([np.ones((tp.duration, ), dtype=np.int) * tp.Index
                         for tp in db.fft.regions.itertuples()])
        write_feature(db.h5_file,
                      "labels", dict(n_classes=len(db.fft.regions)), labels)
        files_labels = np.hstack([np.ones((tp.duration, ), dtype=np.int) * tp.Index
                                 for tp in db.fft.files.itertuples()])
        write_feature(db.h5_file,
                      "files_labels", dict(n_classes=len(db.fft.files)), files_labels)
