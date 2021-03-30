import librosa
from scipy.signal import lfilter
from ..extract.segment import from_recurrence_matrix
import numpy as np
from scipy.interpolate import interp1d


N_FFT = 2048
HOP_LENGTH = 512
SR = 22050
Q_LEVELS = 256


def _make_unidist_shaper(x):
    x = np.sort(x, axis=0)
    indices = _make_unidist_shaper_indices(x)
    yvals = 2 * indices / indices[-1] - 1.0
    encode_shaper = interp1d(x[indices], yvals, bounds_error=False)
    return encode_shaper, list(x[indices]), indices


def _make_unidist_shaper_indices(x):
    ids = np.linspace(0, len(x) - 1, 22).astype(np.int)
    ids = np.concatenate([ids[:1], [ids[1] + (ids[0] - ids[1]) // 2],
                          ids[1:-1], [ids[-2] + (ids[-1] - ids[-2]) // 2], ids[-1:]])
    return ids


class SignalTo:

    @staticmethod
    def stft(y, n_fft=N_FFT, hop_length=HOP_LENGTH, **kwargs):
        S = librosa.stft(y, n_fft=n_fft, hop_length=hop_length, **kwargs)
        return S

    @staticmethod
    def mag_spec(y, n_fft=N_FFT, hop_length=HOP_LENGTH, **kwargs):
        S = SignalTo.stft(y, n_fft, hop_length, **kwargs)
        return abs(S)

    @staticmethod
    def polar_spec(y, n_fft=N_FFT, hop_length=HOP_LENGTH, normalize=False, **kwargs):
        S = SignalTo.stft(y, n_fft, hop_length, **kwargs)
        mags = abs(S)
        if normalize:
            max_mag = np.max(mags)
            mags = mags / max_mag
        angles = np.angle(S)
        return np.stack([mags, angles], 0)

    @staticmethod
    def mu_law_compress(y, q_levels=Q_LEVELS):
        y = librosa.util.normalize(y)
        qx = librosa.mu_compress(y, q_levels-1, quantize=True)
        qx = qx + q_levels // 2
        return qx

    @staticmethod
    def pcm_unsigned(y, q_levels=Q_LEVELS, normalize=True):
        if normalize:
            y = librosa.util.normalize(y)
        qx = ((y + 1.0) * (q_levels - 1)).astype(np.int)
        return qx

    @staticmethod
    def adapted_uniform(y, q_levels=Q_LEVELS, normalize=True):
        if normalize:
            y = librosa.util.normalize(y)
        encode_shaper, shaper_vals, shaper_indices = _make_unidist_shaper(y)
        y = encode_shaper(y)
        qx = ((y + 1.0) * 0.5 * (q_levels - 1)).astype(np.int)
        return qx, (shaper_vals, shaper_indices)


class SignalFrom:

    @staticmethod
    def gla(mag_spec, **kwargs):
        return librosa.griffinlim(mag_spec, **kwargs)

    @staticmethod
    def mu_law_compressed(qx, mu=Q_LEVELS):
        return librosa.mu_expand(qx, mu, quantize=True)


def normalize(y, **kwargs):
    return librosa.util.normalize(y, **kwargs)


def emphasize(y, emphasis):
    return lfilter([1, -emphasis], [1], y)


def deemphasize(y, emphasis):
    return lfilter([1-emphasis], [1, -emphasis], y)


class MagSpecTo:

    @staticmethod
    def regions(S, **kwargs):
        return from_recurrence_matrix(S, **kwargs)


class FileTo:

    @staticmethod
    def signal(file_path, sr=SR):
        y, _ = librosa.load(file_path, sr=sr)
        return y


default_extract_func = FileTo.signal
