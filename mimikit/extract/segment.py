import numpy as np
from librosa.util import peak_pick, localmax
from librosa.beat import tempo
from librosa import griffinlim
from librosa.sequence import dtw
from scipy.interpolate import RectBivariateSpline as RBS
from scipy.ndimage.filters import minimum_filter1d, median_filter
from sklearn.metrics import pairwise_distances as pwd
from joblib import Parallel, delayed
from typing import List
from numba import njit, prange, float64, intp
import json
import matplotlib.pyplot as plt
import click
import os
import soundfile as sf
import shutil
from time import time
import matplotlib as mpl
mpl.rcParams['agg.path.chunksize'] = 10000

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


@njit(float64[:, :](float64[:, :], intp), fastmath=True, cache=True, parallel=True)
def pwdk_cosine(X, k):
    """
    pairwise distance within a kernel size - cosine version

    Parameters
    ----------
    X: np.ndarray (shape = T x D)
        Matrix of observations in time
    k: int
        kernel size - must be odd and > 0

    Returns
    -------
    diags: np.ndarray (shape = T x k)
        cosine distance between observations t
        and t-(k//2) to t+(k//2)
    """
    T = X.shape[0]
    dist = np.zeros((T, 2 * k - 1))
    kdiv2 = k
    # for angular distance:
    # has_negatives = int(np.any(A < 0))
    for i in prange(X.shape[0]):
        for j in prange(max(i - kdiv2, 0), min(i + kdiv2 + 1, T)):
            if i == j:
                continue
            Xi, Xj = np.ascontiguousarray(X[i]), np.ascontiguousarray(X[j])
            # compute cosine distance
            dij = np.dot(Xi, Xj)
            denom = (np.linalg.norm(Xi) * np.linalg.norm(Xj))
            if denom == 0:
                dij = 1.
            else:
                dij = 1 - (dij / denom)
            # for angular distance:
            # dij = (1 + has_negatives) * np.arccos(dij) / np.pi
            dist[i, kdiv2 + (j - i)] = dij
    return dist


@njit(float64[:](float64[:, :], float64[:, :]), fastmath=True, cache=True, parallel=True)
def convolve_diagonals(diagonals, kernel):
    """
    convolve `diagonals` with `kernel`

    Parameters
    ----------
    diagonals: np.ndarray (T x k)
    kernel: np.ndarray (k x k)

    Returns
    -------
    c: np.ndarray (T, )

    """
    K = kernel.shape[0]
    N = diagonals.shape[0] - K + 1
    out = np.zeros((N,), dtype=np.float64)
    # outer loop is parallel, inner sequential
    for i in prange(N):
        s = 0.
        for j in range(K):
            xi = diagonals[i + j:i + j + 1, K - j - 1:(2 * K) - j - 1]
            xi = np.ascontiguousarray(xi)
            kj = np.ascontiguousarray(kernel[j])
            s = s + np.dot(xi, kj)[0]
        out[i] = s
    return out


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
    return block.astype(np.float64)


def discontinuity_scores(
        X: np.ndarray,
        kernel_sizes: List[int],
):
    # make the kernels odd
    kernel_sizes = [(k*2)+1 for k in kernel_sizes]
    max_kernel = max(kernel_sizes)
    X = np.ascontiguousarray(X, dtype=np.float)
    N = X.shape[0]
    scores = np.zeros((len(kernel_sizes), N))
    diagonals = pwdk_cosine(X, max_kernel)
    for i, k in enumerate(kernel_sizes):
        kd2 = k // 2
        if k < max_kernel:
            extra = max_kernel - k
            dk = diagonals[:, extra:-extra]
        else:
            dk = diagonals.copy()
        dk = np.pad(dk, ((kd2, kd2), (0, 0)))
        kernel = checker(kd2, normalize=True)
        scr = convolve_diagonals(dk, kernel)
        scores[i] = scr - scr.min()
    return scores


def pick_globally_sorted_maxes(x, wait_before, wait_after, min_strength=0.02):
    mn = minimum_filter1d(
        x, wait_before + wait_after, mode='constant', cval=x.min()
    )
    glob_rg = x.max() - x.min()
    strength = (x - mn) / glob_rg
    # filter out peaks with too few contrasts
    mx = localmax(x) & (strength >= min_strength)

    mx_indices = mx.nonzero()[0][np.argsort(-x[mx])]

    final_maxes = np.zeros_like(x, dtype=bool)

    for m in mx_indices:
        i, j = max(0, m - wait_before), min(x.shape[0], m + wait_after)
        if np.any(final_maxes[i:j]):
            continue
        else:
            # make sure the max dominates left and right
            # aka we are not globally increasing/decreasing around it
            mu_l = x[i:m].mean()
            mu_r = x[m:j].mean()
            mx = x[m]
            if mx > mu_l and mx > mu_r:
                final_maxes[m] = True
    return final_maxes.nonzero()[0]


def from_recurrence_matrix(X,
                           kernel_sizes=[6],
                           min_dur=4,
                           min_strength=0.03,
                           plot=False):
    N = X.shape[0]
    diagonals = discontinuity_scores(X, kernel_sizes)
    dg = diagonals.mean(axis=0)
    mx2 = peak_pick(dg, min_dur // 2, min_dur // 2, min_dur // 2, min_dur // 2, 0., min_dur)
    mx = pick_globally_sorted_maxes(dg, min_dur, min_dur, min_strength)
    mx = mx[(mx > min_dur) & (mx < (N - min_dur))]
    if plot:
        def plot_diagonals():
            plt.figure(figsize=(dg.size // 500, 10))
            for k, d in zip(kernel_sizes, diagonals):
                plt.plot(d, label=f'kernel_size={k}', linestyle='--', alpha=0.75)
            plt.plot(dg, label=f'mean diagonal')
            plt.legend()
            plt.vlines(mx, dg.min(), dg.max(), linestyles='-', alpha=.5, colors='green', label='new method')
            plt.vlines(mx2, dg.min(), dg.max(), linestyles='dotted', alpha=.75, colors='blue', label='old method')
            plt.legend()
            plt.show()
    else:
        def plot_diagonals():
            return
    return mx, plot_diagonals


def export(S, target_path, sr, n_fft, hop_length):
    from mimikit import GLA
    gla = GLA(n_fft, hop_length)
    if S.shape[0] <= 1:
        print("!!!!!", target_path)
        return
    # if S is too long, joblib sends it as np.memmap
    # which messes with gla()...
    y = gla(np.asarray(S))
    sf.write(f"{target_path}.wav", y, sr, 'PCM_24')
    return


@click.command()
@click.option("-f", "--input-file", help="file to be segmented")
@click.option("--sr", "-r", default=22050, help="the sample rate used for loading the file (default=22050)")
@click.option("--n-fft", "-d", default=2048, help="size of the fft (default=2048)")
@click.option("--hop-length", "-h", default=512, help="hop length of the stft (default=512)")
@click.option("--kernel-size", "-k", default=None, type=int,
              help="half size in frames of the kernel "
                   "(if not specified, tempo of the file is estimated and kernel-size is set to 1 beat)")
@click.option("--factor", "-r", multiple=True, type=float,
              help="factor for the kernel size "
                   "(default to 1., can be specified multiple times)")
@click.option("--min-dur", "-m", default=None, type=int,
              help="minimum number of frames per segment "
                   "(if not specified, default to kernel-size)")
@click.option("--min-strength", "-s", default=0.03, type=float,
              help="minimum strength for a peak to be selected "
                   "(default=0.03, must be between 0. and 1.)")
@click.option("--export-durations", "-x", is_flag=True, help="whether to write the durations as a text file")
@click.option("--plot", "-p", is_flag=True, help="whether to plot the results")
def segment(input_file: str,
            sr: int = 22050,
            n_fft=2048,
            hop_length=512,
            kernel_size=None,
            factor=[],
            min_dur=None,
            min_strength=0.03,
            export_durations=False,
            plot=False
            ):
    """extract segments from an audio file"""

    y, S = etl(input_file, sr, n_fft, hop_length)
    if kernel_size is None:
        tmp = tempo(y, sr)[0]
        kernel_size = int((60 / tmp) * sr) // hop_length
    else:
        tmp = None
    if min_dur is None:
        min_dur = kernel_size
    if not factor:
        factor = (1.,)
    start = time()
    stops, plot_diagonals = from_recurrence_matrix(
        S,
        kernel_sizes=[int(f * kernel_size) for f in factor],
        min_dur=min_dur, min_strength=min_strength, plot=plot
    )
    duration = time() - start
    segments = np.split(S, stops, axis=0)
    target_dir = os.path.splitext(input_file)[0]
    durs, counts = np.unique([s.shape[0] for s in segments], return_counts=True)
    distrib_str = "\n    ".join([f"{d}  :  {c}"
                                 for d, c in zip(durs[(-counts).argsort()[:10]],
                                                 counts[(-counts).argsort()[:10]])]) + \
                  "\n    [  ...  ]\n"
    print(
        f"""
    Segmented '{input_file}' in {duration:.3f} seconds

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
    print(f"writing segments to target directory '{target_dir}/'")
    if os.path.exists(target_dir):
        shutil.rmtree(target_dir)
    os.makedirs(target_dir, exist_ok=True)
    Parallel(n_jobs=-1, backend='multiprocessing', batch_size=16) \
        (delayed(export)(s, f"{target_dir}/{i}", sr, n_fft, hop_length)
         for i, s in enumerate(segments))

    params = dict(k=kernel_size, r=factor, m=min_dur)
    with open(os.path.join(target_dir, "params.json"), 'w') as f:
        f.write(json.dumps(params))
    if export_durations:
        with open(os.path.join(target_dir, "durations.json"), 'w') as f:
            f.write(json.dumps({i: {"o_dur": s.shape[0], "target": s.shape[0]} for i, s in enumerate(segments)}, indent=1))

    if plot:
        plot_diagonals()
        plt.show()
        plt.figure(figsize=(60, 10))
        plt.plot(y)
        plt.vlines(np.cumsum([s.shape[0] * hop_length for s in segments]), -1, 1, linestyles='--',
                   alpha=.5, colors='green')
        plt.show()
    return


def _stretch_rbs(S, ratio):
    if S.shape[1] <= 1 or S.shape[0] <= 1:
        return S
    time_indices = np.linspace(0, S.shape[1] - 1, int(np.rint(S.shape[1] * ratio)))
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


def _kde_stretch(S, target_dur, n_components=32, grid_size=16, smoother_size=2):
    from sklearn.neighbors import KernelDensity
    from sklearn.decomposition import PCA
    from sklearn.model_selection import GridSearchCV

    # project the data to a lower dimension
    pca = PCA(n_components=min(n_components, S.shape[0]), whiten=False)
    data = pca.fit_transform(S)

    # use grid search cross-validation to optimize the bandwidth
    params = {"bandwidth": np.logspace(-1, 1, grid_size)}
    grid = GridSearchCV(KernelDensity(), params)
    grid.fit(data)

    print("best bandwidth: {0}".format(grid.best_estimator_.bandwidth))

    # use the best estimator to compute the kernel density estimate
    kde = grid.best_estimator_

    # sample 44 new points from the data
    new_data = kde.sample(target_dur)
    d = pwd(new_data, data, metric='cosine')
    idx = np.argsort(d.argmin(axis=1))
    new_data = new_data[idx]
    # new_data = median_filter(new_data, size=(smoother_size, smoother_size))
    return pca.inverse_transform(new_data)


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
@click.option("--stretcher", "-t", type=str, help="method to use for stretching. One of 'rbs', 'kde'",
              default='rbs')
@click.option("--verbose", "-v", is_flag=True,
              help="whether to print out the transformation durations")
def re_stretch(source_dir: str,
               sr=22050,
               n_fft=2048,
               hop_length=512,
               nmm=None,
               xpand=None,
               manual=False,
               stretcher='rbs',
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
    with open(os.path.join(source_dir, "params.json"), 'r') as f:
        arg_str = "_".join([k+str(v) for k, v in json.loads(f.read()).items()])
    if nmm is not None:
        targets = nearest_multiple(durations, nmm)
        arg_str += f"_n{nmm}"
    elif xpand is not None:
        targets = x_pand(durations, xpand)
        arg_str += f"_x{xpand}"
    elif manual and os.path.isfile(os.path.join(source_dir, "durations.json")):
        dct = json.load(open(os.path.join(source_dir, "durations.json"), 'r'))
        targets = np.zeros((durations.size,), dtype=np.int)
        for i, data in dct.items():
            targets[int(i)] = data['target']
        arg_str += f"_manual"
    else:
        raise ValueError("`nmm` and `xpand` are None, `manual` is False. Cannot stretch without targets.")
    if stretcher == 'rbs':
        stretch_func = _stretch
    else:
        stretch_func = _kde_stretch
    if verbose:
        for i, (o, t) in enumerate(zip(durations, targets)):
            print(f"{i}: {o} -> {t}")
    stretched = Parallel(n_jobs=-1, backend="multiprocessing")(
        delayed(stretch_func)(s, target)
        for s, target in zip(segments, targets)
    )
    stretched = np.concatenate(stretched, axis=0)
    export(stretched, os.path.join(source_dir, f"stretched_{arg_str}"), sr, n_fft, hop_length)
    return
