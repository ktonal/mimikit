import numpy as np
from librosa.util import peak_pick, localmax
from librosa.sequence import dtw
from scipy.ndimage.filters import minimum_filter1d
from sklearn.metrics import pairwise_distances as pwd
from typing import List
from numba import njit, prange, float64, intp
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['agg.path.chunksize'] = 10000

__all__ = [
    'from_recurrence_matrix'
]


def optimal_path(x, y):
    return dtw(C=pwd(abs(x), abs(y), metric='cosine'), subseq=True)[1][::-1]


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
                           min_strength=0.03
                           ):
    N = X.shape[0]
    diagonals = discontinuity_scores(X, kernel_sizes)
    dg = diagonals.mean(axis=0)
    mx2 = peak_pick(dg, min_dur // 2, min_dur // 2, min_dur // 2, min_dur // 2, 0., min_dur)
    mx = pick_globally_sorted_maxes(dg, min_dur, min_dur, min_strength)
    mx = mx[(mx > min_dur) & (mx < (N - min_dur))]
    return mx, mx2, diagonals


class CutsFromRecurrenceMatrix:
    def __init__(self,
                 kernel_size: int = 6,
                 factors: List[float] = [1.],
                 min_dur: int = 4,
                 min_strength: float = 0.03,
                 ):
        self.kernel_sizes = [int(f * kernel_size) for f in factors]
        self.min_dur = min_dur
        self.min_strength = min_strength

    def __call__(self, X):
        """
        X; 2d np.ndarray, e.g. STFT with shape (Time x Freq)
        """
        self.mx, self.mx2, self.diagonals = \
            from_recurrence_matrix(X, self.kernel_sizes, self.min_dur, self.min_strength)
        return self.mx

    def plot_diagonals(self):
        dg = self.diagonals.mean(axis=0)
        plt.figure(figsize=(dg.size // 500, 10))
        for k, d in zip(self.kernel_sizes, self.diagonals):
            plt.plot(d, label=f'kernel_size={k}', linestyle='--', alpha=0.75)
        plt.plot(dg, label=f'mean diagonal')
        plt.legend()
        plt.vlines(self.mx, dg.min(), dg.max(), linestyles='-', alpha=.5, colors='green', label='new method')
        plt.vlines(self.mx2, dg.min(), dg.max(), linestyles='dotted', alpha=.75, colors='blue', label='old method')
        plt.legend()
        plt.show()