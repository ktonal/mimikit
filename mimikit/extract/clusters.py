import numpy as np
from scipy.sparse.csgraph import connected_components
from sklearn.metrics import pairwise_distances as pwd
from sklearn.neighbors import KNeighborsTransformer, NearestNeighbors
import sklearn.cluster as C
import torch
import torch.nn as nn
from joblib import Parallel, delayed
import click
import os
import soundfile as sf
import shutil
from functools import partial

from ..modules import angular_distance


class QCluster:

    def __init__(self, qe=.5, n_neighbs=None, k=None, metric="euclidean"):
        self.qe = qe
        self.n_neighbs = n_neighbs
        self.k = k
        self.metric = metric
        self.is_core_ = None
        self.labels_ = None
        self.K_ = None

    def fit(self, x):
        N = x.shape[0]
        if self.n_neighbs is None:
            self.n_neighbs = int(np.sqrt(N))
        if self.k is None:
            self.k = int(self.qe * self.n_neighbs)

        D = pwd(x, x, metric=self.metric)
        Dk = D.copy()
        Dk[Dk == 0] = np.inf

        kn = KNeighborsTransformer(mode="distance", n_neighbors=self.n_neighbs,
                                   metric="precomputed")
        adj = kn.fit_transform(D)
        adj_sub = kn.kneighbors_graph(None, self.k, mode="distance")
        Kx = (0 < adj.A).sum(axis=0).reshape(-1)
        is_core = Kx >= np.quantile(Kx, self.qe)

        adj_cores = (adj_sub.A.astype(np.bool)) & (is_core & is_core[:, None])
        cores_idx = is_core.nonzero()[0]
        print(f"found {cores_idx.shape[0]} cores")
        for i in range(adj.shape[0]):
            if not np.any(adj_cores[i] & is_core):
                nearest_core = Dk[i, is_core].argmin()
                nearest_core = cores_idx[nearest_core]
                adj_cores[i, nearest_core] = True

        K, labels = connected_components(adj_cores)
        self.K_, self.labels_, self.is_core_ = K, labels, is_core
        return self


class GCluster:
    def __init__(self,
                 n_means,
                 n_iter=128,
                 lr=0.025,
                 betas=(0.05, 0.05),
                 metric="cosine",
                 eps=1e-6):
        self.n_means = n_means
        self.n_iter = n_iter
        self.lr = lr
        self.betas = betas
        self.metric = metric
        self.d_func = dict(euclidean=torch.cdist,
                           cosine=partial(angular_distance, eps=eps))[metric]
        self.K_ = None
        self.labels_ = None
        self.losses_ = None

    def fit(self, x):
        X = torch.from_numpy(x)
        H = nn.Parameter(X[torch.randint(0, X.shape[0], (self.n_means,))].clone())
        opt = torch.optim.Adam([H], lr=self.lr, betas=self.betas)
        losses = []
        for _ in range(self.n_iter):
            opt.zero_grad()
            L = self.d_func(H, X).mean()
            L = L - .5 * self.d_func(H, H).mean()
            L.backward()
            opt.step()
            losses += [L.item()]
        x = X.detach().cpu().numpy()
        h = H.detach().cpu().numpy()
        DXH = pwd(h, x, self.metric)
        hi, xi = np.unravel_index(DXH.argsort(None), DXH.shape)
        labels = np.zeros(x.shape[0], dtype=np.int)
        got = set()
        for label, i in zip(hi.flat[:], xi.flat[:]):
            if i not in got:
                labels[i] = label
                got.add(i)
        self.losses_ = losses
        self.K_ = self.n_means
        self.labels_ = labels
        return self


class HCluster:
    def __init__(self, max_iter=32, metric="cosine"):
        self.max_iter = max_iter
        self.metric = metric
        self.K_ = None
        self.labels_ = None

    def fit(self, x):
        Da = pwd(x, x, self.metric)
        xa = x.copy()
        Da[Da == 0] = np.inf
        LBS = np.zeros((x.shape[0], self.max_iter), dtype=np.int)

        for i in range(self.max_iter):
            Adj_nearest = np.zeros_like(Da, dtype=np.bool)
            nearest = Da.argmin(axis=1)
            Adj_nearest[np.arange(Da.shape[0]), nearest] = True

            K, labels = connected_components(Adj_nearest)
            if i == 0:
                LBS[:, 0] = labels
            else:
                LBS[:, i] = np.r_[[labels[LBS[n, i - 1]] for n in range(x.shape[0])]]

            xa = np.stack([xa[labels == k].mean(axis=0) for k in range(K)])
            Da = pwd(xa, xa, metric=self.metric)
            Da[Da == 0] = np.inf
            if K == 1:
                LBS = LBS[:, :i + 1]
                self.K_ = i + 1
                break

        self.labels_ = LBS
        return self


def distance_matrices(X, metric="euclidean", n_neighbors=1, radius=1e-3):
    Dx = pwd(X, X, metric=metric)
    NN = NearestNeighbors(n_neighbors=n_neighbors, radius=radius, metric="precomputed")
    NN.fit(Dx)
    Kx = NN.kneighbors_graph(n_neighbors=n_neighbors, mode='connectivity')
    Rx = NN.radius_neighbors_graph(radius=radius, mode='connectivity')
    return Dx, Kx, Rx


class ArgMax(object):
    def __init__(self):
        self.labels_ = None
        self.K_ = None

    def fit(self, X):
        maxes = np.argmax(X, axis=1)
        uniques, self.labels_ = np.unique(maxes, return_inverse=True)
        self.K_ = len(uniques)
        return self


def sk_cluster(X, Dx=None, n_clusters=128, metric="euclidean", estimator="argmax"):
    estimators = {

        "argmax": ArgMax(),

        "kmeans": C.KMeans(n_clusters=n_clusters,
                           n_init=4,
                           max_iter=200,
                           ),

        "spectral": C.SpectralClustering(n_clusters=n_clusters,
                                         affinity="nearest_neighbors",
                                         n_neighbors=32,
                                         assign_labels="discretize",
                                         ),

        "agglo_ward": C.AgglomerativeClustering(
            n_clusters=n_clusters,
            affinity="euclidean",
            compute_full_tree='auto',
            linkage='ward',
            distance_threshold=None, ),

        "agglo_single": C.AgglomerativeClustering(
            n_clusters=n_clusters,
            affinity="precomputed",
            compute_full_tree='auto',
            linkage='single',
            distance_threshold=None, ),

        "agglo_complete": C.AgglomerativeClustering(
            n_clusters=n_clusters,
            affinity="precomputed",
            compute_full_tree='auto',
            linkage='complete',
            distance_threshold=None, )

    }

    needs_distances = estimator in {"agglo_single", "agglo_complete"}

    if needs_distances:
        if Dx is None:
            Dx, _, _ = distance_matrices(X, metric=metric)
        X_ = Dx
    else:
        X_ = X

    cls = estimators[estimator]
    cls.fit(X_)

    return cls


def etl(input_file, sr, n_fft, hop_length):
    """STFT Extract-Transform-Load"""
    from mimikit import FileToSignal, MagSpec

    y = FileToSignal(sr)(input_file)
    S = MagSpec(n_fft, hop_length)(y)
    return y, S


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
@click.option("--n-neighbs", "-n", default=None, type=int,
              help="size of the neighborhood used to estimate point-density")
@click.option("--qe", "-q", default=0.5, type=float,
              help="proportion of edge-points")
@click.option("--k", "-k", default=1, type=int,
              help="proportion of connecting core points")
def cluster(input_file: str,
            sr: int = 22050,
            n_fft: int = 2048,
            hop_length: int = 512,
            n_neighbs: int = None,
            qe: float = 0.5,
            k: int = 1
            ):
    y, S = etl(input_file, sr, n_fft, hop_length)
    est = QCluster(qe=qe, n_neighbs=n_neighbs, k=k, metric="euclidean")
    est.fit(S)

    clusters = [S[est.labels_ == i] for i in range(est.K_)]
    items, counts = np.unique(est.labels_, return_counts=True)
    srtd = (-counts).argsort()
    items, counts = items[srtd[:10]], counts[srtd[:10]]
    distrib_str = "\n    ".join([f"{d}  :  {c}"
                                 for d, c in zip(items, counts)]) + \
                  "\n    [  ...  ]\n"
    print(f"""
Data shape : {S.shape}
Parameters : n_neighbs={est.n_neighbs} ; k={est.k} ; qe={qe}
========>   found {est.K_} clusters
    """ + distrib_str
          )

    target_dir = os.path.splitext(input_file)[0]
    if os.path.exists(target_dir):
        shutil.rmtree(target_dir)
    os.makedirs(target_dir, exist_ok=True)
    Parallel(backend='multiprocessing') \
        (delayed(export)(s, f"{target_dir}/c{i}", sr, n_fft, hop_length)
         for i, s in enumerate(clusters))
    return
