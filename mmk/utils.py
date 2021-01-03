import librosa
import numpy as np
from librosa.display import specshow
import matplotlib.pyplot as plt
import IPython.display as ipd
import torch
from .data.transforms import HOP_LENGTH, SR

# Conversion

normalize = librosa.util.normalize
a2db = lambda S: librosa.amplitude_to_db(abs(S), ref=S.max())
s2f = librosa.samples_to_frames
s2t = librosa.samples_to_time
f2s = librosa.frames_to_samples
f2t = librosa.frames_to_time
t2f = librosa.time_to_frames
t2s = librosa.time_to_samples
hz2m = librosa.hz_to_midi
m2hz = librosa.midi_to_hz


# Debugging utils

def to_db(S):
    if S.dtype == np.complex64:
        S_hat = a2db(abs(S)) + 40
    elif S.min() >= 0 and S.dtype in (np.float, np.float32, np.float64, np.float_):
        S_hat = a2db(S) + 40
    else:
        S_hat = a2db(S) + 40
    return S_hat


def signal(S, hop_length=HOP_LENGTH):
    if S.dtype in (np.complex64, np.complex128):
        return librosa.istft(S, hop_length=hop_length)
    else:
        return librosa.griffinlim(S, hop_length=hop_length, n_iter=64)


def audio(S, hop_length=HOP_LENGTH, sr=SR):
    if len(S.shape) > 1:
        y = signal(S, hop_length)
        if y.size > 0:
            return ipd.display(ipd.Audio(y, rate=sr))
        else:
            return ipd.display(ipd.Audio(np.zeros(hop_length*2), rate=sr))
    else:
        return ipd.display(ipd.Audio(S, rate=sr))


def playlist(iterable):
    for seg in iterable:
        audio(seg)
    return


def playthrough(iterable, axis=1):
    rv = np.concatenate(iterable, axis=axis)
    return audio(rv)


def show(S, figsize=(), db_scale=True, title="", **kwargs):
    S_hat = to_db(S) if db_scale else S
    if figsize:
        plt.figure(figsize=figsize)
    if "x_axis" not in kwargs:
        kwargs["x_axis"] = "frames"
    if "y_axis" not in kwargs:
        kwargs["y_axis"] = "frames"
    ax = specshow(S_hat, sr=SR, **kwargs)
    plt.colorbar()
    plt.tight_layout()
    plt.title(title)
    return ax


# Array <==> Tensor ops :

def numcpu(y):
    if isinstance(y, torch.Tensor):
        if y.requires_grad:
            return y.detach().cpu().numpy()
        return y.cpu().numpy()
    else:  # tuples
        return tuple(numcpu(x) for x in y)


def cuda_if_available():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def to_torch(x, device=cuda_if_available()):
    if isinstance(x, np.ndarray):
        return torch.from_numpy(x).float().to(device)
    elif isinstance(x, torch.Tensor):
        if x.device == device:
            return x.float()
        else:
            return x.float().to(device)
    else:  # tuples
        return tuple(to_torch(y) for y in x)
