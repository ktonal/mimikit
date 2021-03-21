import torchaudio.functional as F
import torchaudio.transforms as T
import torch
import numpy as np
from abc import ABC

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
<<<<<<< HEAD
        if normalize:
            inputs = inputs / torch.norm(inputs, p=float("inf"))
        if sample_encoding == 'mu_law':
            return F.mu_law_encoding(inputs, q_levels)
        elif sample_encoding == 'adapted':
            indices = torch.from_numpy(shaper[1]).float()
            xvals = 2 * indices / indices[-1] - 1.0
            # unfortunately Interp1d does not let you select an axis - so process columns one by one
            signal = torch.stack([Interp1d()(torch.from_numpy(shaper[0]).float().to(inputs),
                                             xvals,
                                             2.0 * inputs[:, k] / (q_levels - 1) - 1.0)
                                  for k in range(inputs.shape[1])]).T
            return ((signal + 1.0) * 0.5 * (q_levels - 1)).astype(np.int)
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
            indices = torch.from_numpy(shaper[1]).float().to(outputs)
            xvals = 2 * indices / indices[-1] - 1.0
            # unfortunately Interp1d does not let you select an axis - so process columns one by one
            signal = torch.stack([Interp1d()(xvals,
                                             torch.from_numpy(shaper[0]).float().to(outputs),
                                             2.0 * outputs[:, k] / (q_levels - 1) - 1.0)
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
