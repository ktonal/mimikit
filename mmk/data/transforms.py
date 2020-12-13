import librosa
from .metadata import Metadata


N_FFT = 2048
HOP_LENGTH = 512
SR = 22050


def stft(file, n_fft=N_FFT, hop_length=HOP_LENGTH, sr=SR):
    y, sr = librosa.load(file, sr=sr)
    return librosa.stft(y, n_fft=n_fft, hop_length=hop_length)


def file_to_fft(abs_path):
    fft = abs(stft(abs_path))
    metadata = Metadata.from_duration([fft.shape[1]])
    return dict(fft=(dict(time_axis=0), fft.T), metadata=({}, metadata))


default_extract_func = file_to_fft

