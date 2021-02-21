import librosa
from ..extract.segment import from_recurrence_matrix

N_FFT = 2048
HOP_LENGTH = 512
SR = 22050
MU = 255


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
    def mu_law_compress(y, mu=MU):
        y = librosa.util.normalize(y)
        qx = librosa.mu_compress(y, mu, quantize=True)
        qx = qx + (MU + 1) // 2
        return qx


class SignalFrom:

    @staticmethod
    def gla(mag_spec, **kwargs):
        return librosa.griffinlim(mag_spec, **kwargs)

    @staticmethod
    def mu_law_compressed(qx, mu=MU):
        return librosa.mu_expand(qx, mu, quantize=True)


class MagSpecTo:

    @staticmethod
    def regions(S, **kwargs):
        return from_recurrence_matrix(S, **kwargs)


class FileTo:

    @staticmethod
    def signal(file_path, sr=SR):
        y, _ = librosa.load(file_path, sr=sr)
        return y

    @staticmethod
    def stft(file_path, sr=SR, n_fft=N_FFT, hop_length=HOP_LENGTH, **kwargs):
        y = FileTo.signal(file_path, sr)
        S = SignalTo.stft(y, n_fft, hop_length, **kwargs)
        return S

    @staticmethod
    def mag_spec(file_path, sr=SR, n_fft=N_FFT, hop_length=HOP_LENGTH, **kwargs):
        y = FileTo.signal(file_path, sr)
        S = SignalTo.mag_spec(y, n_fft, hop_length, **kwargs)
        return S

    @staticmethod
    def mu_law_compress(file_path, sr=SR, mu=MU):
        y = FileTo.signal(file_path, sr)
        y = librosa.util.normalize(y)
        qx = librosa.mu_compress(y, mu, quantize=True)
        qx = qx + (MU + 1) // 2
        return qx


default_extract_func = FileTo.signal
