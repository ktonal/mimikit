import librosa
from .metadata import Metadata

N_FFT = 2048
HOP_LENGTH = 512
SR = 22050


def stft(file, n_fft=N_FFT, hop_length=HOP_LENGTH, sr=SR):
    y, sr = librosa.load(file, sr=sr)
    fft = librosa.stft(y, n_fft=n_fft, hop_length=hop_length)
    # returns the feature and its attributes
    return fft, dict(n_fft=n_fft, hop_length=hop_length, sr=sr)


def file_to_fft(abs_path, n_fft=N_FFT, hop_length=HOP_LENGTH, sr=SR):
    fft, params = stft(abs_path, n_fft, hop_length, sr)
    fft = abs(fft)
    metadata = Metadata.from_duration([fft.shape[1]])
    params.update(dict(time_axis=0))
    return dict(fft=(params, fft.T), metadata=({}, metadata))


default_extract_func = file_to_fft
