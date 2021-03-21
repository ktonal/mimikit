import librosa
from scipy.signal import lfilter
from ..extract.segment import from_recurrence_matrix


N_FFT = 2048
HOP_LENGTH = 512
SR = 22050
Q_LEVELS = 256


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
    def mu_law_compress(y, q_levels=Q_LEVELS):
        y = librosa.util.normalize(y)
        qx = librosa.mu_compress(y, q_levels-1, quantize=True)
        qx = qx + q_levels // 2
        return qx


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
