from typing import List

import numpy as np
from librosa.util import peak_pick
from librosa.beat import tempo
from librosa import griffinlim
from librosa.sequence import dtw
from scipy.interpolate import RectBivariateSpline as RBS
from sklearn.metrics import pairwise_distances as pwd
from joblib import Parallel, delayed
import json
import matplotlib.pyplot as plt
import click
import os
import soundfile as sf
import shutil

__all__ = [
    'from_recurrence_matrix'
]


def optimal_path(x, y):
    """
    Parameters
    ----------
    x: np.ndarray
        shape: (Time x Dim)
    y: np.ndarray
        shape: (Time x Dim)

    Returns
    -------
    path: np.ndarray
    """
    return dtw(C=pwd(abs(x), abs(y), metric='cosine'), subseq=True)[1][::-1]


def etl(input_file, sr, n_fft, hop_length):
    """STFT Extract-Transform-Load"""
    from mimikit import FileToSignal, MagSpec

    y = FileToSignal(sr)(input_file)
    S = MagSpec(n_fft, hop_length)(y)
    return y, S


def checker(N, normalize=True):
    """
    checker(2)
    =>  [[-1, -1, 0,  1,  1],
         [-1, -1, 0,  1,  1],
         [ 0,  0, 0,  0,  0],
         [ 1,  1, 0, -1, -1],
         [ 1,  1, 0, -1, -1]]
    """
    block = np.zeros((N * 2 + 1, N * 2 + 1), dtype=np.int32)
    for k in range(-N, N + 1):
        for l in range(-N, N + 1):
            block[k + N, l + N] = - np.sign(k) * np.sign(l)
    if normalize:
        block = block / np.abs(block).sum()
    return block


def from_recurrence_matrix(X,
                           kernel_sizes=[6],
                           min_dur=4,
                           plot=False):
    """

    Parameters
    ----------
    X: np.ndarray
        shape: T x D
    kernel_size
    min_dur
    plot

    Returns
    -------
    indices of the segments
    """
    R = pwd(X, metric='cosine', n_jobs=1)
    diagonals = np.zeros((len(kernel_sizes), R.shape[0]))
    for scale, kernel_size in enumerate(kernel_sizes):
        # intensify checker-board-like entries
        K = checker(kernel_size)
        # convolve R with K manually is more efficient
        # since we only need to convolve along the diag
        R_hat = np.pad(R, kernel_size, mode="constant")
        R_hat = np.stack([R_hat[i:i + 2 * kernel_size + 1, i:i + 2 * kernel_size + 1].flat[:]
                          for i in range(X.shape[0])])
        dg = (R_hat @ K.flat[:].reshape(-1, 1)).reshape(-1)
        dg = dg - dg.min()
        diagonals[scale] = dg
    dg = diagonals.mean(axis=0)
    print(R.shape, diagonals.shape, dg.shape)
    mx = peak_pick(dg, min_dur // 2, min_dur // 2, min_dur // 2, min_dur // 2, 0., min_dur)
    mx = mx[(mx > min_dur) & (mx < R.shape[0] - min_dur)]
    if plot:
        plt.figure(figsize=(dg.size, 10))
        for k, d in zip(kernel_sizes, diagonals):
            plt.plot(d, label=f'kernel_size={k}', linestyle='--', alpha=0.75)
        plt.plot(dg, label=f'mean diagonal')
        plt.legend()
        plt.vlines(mx, dg.min(), dg.max(), linestyles='-')
        plt.show()
    return mx


def export(S, target_path, sr, n_fft, hop_length):
    from mimikit import GLA
    gla = GLA(n_fft, hop_length)
    if S.shape[0] <= 1:
        print("!!!!!", target_path)
        return
    y = gla(S)
    sf.write(f"{target_path}.wav", y, sr, 'PCM_24')
    return


@click.command()
@click.option("-f", "--input-file", help="file to be segmented")
@click.option("--sr", "-r", default=22050, help="the sample rate used for loading the file (default=22050)")
@click.option("--n-fft", "-d", default=2048, help="size of the fft (default=2048)")
@click.option("--hop-length", "-h", default=512, help="hop length of the stft (default=512)")
@click.option("--kernel-size", "-s", default=None, type=int,
              help="half size in frames of the kernel "
                   "(if not specified, tempo of the file is estimated and kernel-size is set to 1 beat)")
@click.option("--factor", "-r", multiple=True, type=float,
              help="factor for the kernel size "
                   "(default to 1., can be specified multiple times)")
@click.option("--min-dur", "-m", default=None, type=int,
              help="minimum number of frames per segment "
              "(if not specified, default to kernel-size)")
@click.option("--export-durations", "-x", is_flag=True, help="whether to write the durations as a text file")
@click.option("--plot", "-p", is_flag=True, help="whether to plot the results")
def segment(input_file: str,
            sr: int = 22050,
            n_fft=2048,
            hop_length=512,
            kernel_size=None,
            factor=[],
            min_dur=None,
            export_durations=False,
            plot=False
            ):
    """extract segments from an audio file"""

    y, S = etl(input_file, sr, n_fft, hop_length)
    S = S[:2000]
    if kernel_size is None:
        tmp = tempo(y, sr)[0]
        kernel_size = int((60 / tmp) * sr) // hop_length
    else:
        tmp = None
    if min_dur is None:
        min_dur = kernel_size
    if not factor:
        factor = (1.,)
    stops = from_recurrence_matrix(S,
                                   kernel_sizes=[int(f*kernel_size) for f in factor],
                                   min_dur=min_dur, plot=plot)
    segments = np.split(S, stops, axis=0)
    target_dir = os.path.splitext(input_file)[0]
    print(f"writing segments to target directory '{target_dir}/'")
    if os.path.exists(target_dir):
        shutil.rmtree(target_dir)
    os.makedirs(target_dir, exist_ok=True)
    Parallel(n_jobs=-1, backend='multiprocessing') \
        (delayed(export)(s, f"{target_dir}/{i}", sr, n_fft, hop_length)
         for i, s in enumerate(segments))
    durs, counts = np.unique([s.shape[0] for s in segments], return_counts=True)
    distrib_str = "\n    ".join([f"{d}  :  {c}"
                                 for d, c in zip(durs[(-counts).argsort()[:10]],
                                                 counts[(-counts).argsort()[:10]])]) + \
                  "\n    [  ...  ]\n"
    print(
        f"""
    Original data shape       = {S.shape}
    {f'estimated tempo           = {tmp:.3f} BPM' if tmp is not None else ""}
    1 frame                   = {hop_length / sr:.3f} sec
    kernel size               = {kernel_size} frames ({kernel_size * hop_length / sr:.3f} sec)
    min dur                   = {min_dur} frames ({min_dur * hop_length / sr:.3f} sec)

    number of segments        = {len(segments)}

    MIN  segment duration     = {min(durs)} frames ({np.min(durs) * hop_length / sr:.3f} sec)
    MAX  segment duration     = {max(durs)} frames ({np.max(durs) * hop_length / sr:.3f} sec)
    MEAN segment duration     = {np.mean(durs):.1f} frames ({np.mean(durs) * hop_length / sr:.3f} sec)
    STD  segment duration     = {np.std(durs):.3f} frames ({np.std(durs) * hop_length / sr:.3f} sec)

    Dur : Count
    """ + distrib_str
    )
    if export_durations:
        with open(os.path.join(target_dir, "durations.json"), 'w') as f:
            f.write(json.dumps({i: {"o_dur": s.shape[0], "target": s.shape[0]} for i, s in enumerate(segments)}, indent=1))
    if plot:
        plt.figure(figsize=(y.shape[0]//sr, 10))
        plt.plot(y)
        plt.vlines(np.cumsum([s.shape[0] * hop_length for s in segments]), -1, 1, linestyles='--',
                   alpha=.5, colors='green')
        plt.show()
    return


def _stretch_rbs(S, ratio):
    if S.shape[1] <= 1 or S.shape[0] <= 1:
        return S
    time_indices = np.linspace(0, S.shape[1]-1, int(np.rint(S.shape[1] * ratio)))
    if S.dtype in (np.complex64, np.complex128):
        mag, phase = abs(S), np.imag(S)
    else:
        mag, phase = S, None
    spline = RBS(np.arange(S.shape[0]), np.arange(S.shape[1]), mag)
    interp = spline.ev(np.arange(S.shape[0])[:, None], time_indices)
    if S.dtype in (np.complex64, np.complex128):
        return griffinlim(interp, n_iter=32)
    else:
        return interp


def _stretch(S, target_dur):
    r = target_dur / S.shape[0]
    Shat = _stretch_rbs(S.T, r).T
    return Shat


def nearest_multiple(x, mu):
    return np.maximum(np.round(x / mu) * mu, mu).astype(x.dtype)


def x_pand(x, ga):
    loc, scale = np.mean(x), np.std(x)
    xh = (x - loc) / scale
    y = np.round(loc + (xh * ga) * scale).astype(x.dtype)
    return np.maximum(np.minimum(y, x.max()), 2)


@click.command()
@click.option("-s", "--source-dir", help="file to be segmented")
@click.option("--sr", "-r", default=22050, help="the sample rate used for loading the file")
@click.option("--n-fft", "-d", default=2048, help="size of the fft")
@click.option("--hop-length", "-h", default=512, help="hop length of the stft")
@click.option("--nmm", "-n", default=None, type=int,
              help="stretch segments to the nearest multiple of the specified value")
@click.option("--xpand", "-x", default=None, type=float,
              help="expand (>1.) or compress (<1.) durations around their mean")
@click.option("--manual", "-m", is_flag=True,
              help="if specified, expects `source-dir` to contain a `durations.json` file "
                   "containing target durations for each segment")
@click.option("--verbose", "-v", is_flag=True,
              help="whether to print out the transformation durations")
def re_stretch(source_dir: str,
               sr=22050,
               n_fft=2048,
               hop_length=512,
               nmm=None,
               xpand=None,
               manual=False,
               verbose=False):
    from h5mapper import FileWalker

    files = sorted(FileWalker(r'(.*/[0-9]*\.wav$)', source_dir),
                   key=lambda f: int(os.path.splitext(os.path.split(f)[1])[0]))
    segments = Parallel(n_jobs=-1, backend="multiprocessing")(
        delayed(etl)(f, sr, n_fft, hop_length)
        for f in files
    )
    segments = [s[1] for s in segments]
    durations = np.r_[[s.shape[0] for s in segments]]
    if nmm is not None:
        targets = nearest_multiple(durations, nmm)
    elif xpand is not None:
        targets = x_pand(durations, xpand)
    elif manual and os.path.isfile(os.path.join(source_dir, "durations.json")):
        dct = json.load(open(os.path.join(source_dir, "durations.json"), 'r'))
        targets = np.zeros((durations.size,), dtype=np.int)
        for i, data in dct.items():
            targets[int(i)] = data['target']
    else:
        raise ValueError("`nmm` and `xpand` are None, `manual` is False. Cannot stretch without targets.")
    if verbose:
        for i, (o, t) in enumerate(zip(durations, targets)):
            print(f"{i}: {o} -> {t}")
    stretched = Parallel(n_jobs=-1, backend="multiprocessing")(
        delayed(_stretch)(s, target)
        for s, target in zip(segments, targets)
    )
    stretched = np.concatenate(stretched, axis=0)
    export(stretched, os.path.join(source_dir, "stretched"), sr, n_fft, hop_length)
    return
