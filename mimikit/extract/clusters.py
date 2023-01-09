from functools import partial

import numpy as np
from scipy.sparse.csgraph import connected_components
from scipy.sparse import csc_matrix
from sklearn.metrics import pairwise_distances as pwd
from sklearn.neighbors import KNeighborsTransformer, NearestNeighbors
import sklearn.cluster as C
import torch
import torch.nn as nn

from mimikit.modules.loss_functions import AngularDistance


class QCluster:

    def __init__(self,
                 cores_prop=.5,
                 n_neighbors=None,
                 core_neighborhood_size=None,
                 metric="euclidean"):
        self.qe = 1 - cores_prop
        self.n_neighbs = n_neighbors
        self.k = core_neighborhood_size
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

        n_neighbs = self.n_neighbs
        kn = KNeighborsTransformer(mode="distance", n_neighbors=n_neighbs,
                                   metric=self.metric)
        adj_o = kn.fit_transform(x)
        adj = adj_o.tolil()
        rg = np.arange(adj.shape[0])
        adj[rg, rg] = 0.
        in_degree = (0 < adj.tocsc()).sum(axis=0).A.reshape(-1)
        is_core = in_degree >= np.quantile(in_degree, self.qe)
        cores_idx = is_core.nonzero()[0]
        k = self.k
        # print(f"found {cores_idx.shape[0]} cores ({cores_idx.shape[0] / N:.3f})-- will do {k} connections")
        adj_sub_o = kn.kneighbors_graph(x[is_core], n_neighbors=k + 1, mode='distance')
        asub = adj_sub_o.tocoo()
        adj_c = csc_matrix((np.r_[[x in cores_idx for x in asub.col]],
                            (cores_idx[asub.row], asub.col)),
                           shape=adj.shape)
        adj_c = (adj_c > 0).tolil()

        disconnected = (adj_c.tocsc()[:, cores_idx].tocsr().sum(axis=1).A.reshape(-1) == 0)

        cores_est = KNeighborsTransformer(mode="distance", n_neighbors=2,
                                          metric=self.metric).fit(x[is_core])

        nearest_cores = cores_est.kneighbors(x[disconnected], return_distance=False)
        nearest_cores = cores_idx[nearest_cores]
        dis_idx = rg[disconnected]
        # print("n disconnected = ", len(dis_idx))
        for i, cores in zip(dis_idx, nearest_cores):
            # nearest core can equal i!
            nearest_core = next(n for n in cores if n != i)
            adj_c[i, nearest_core] = True

        K, labels = connected_components(adj_c)
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
                           cosine=AngularDistance(eps=eps))[metric]
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
        labels = np.zeros(x.shape[0], dtype=int)
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
        LBS = np.zeros((x.shape[0], self.max_iter), dtype=int)

        for i in range(self.max_iter):
            Adj_nearest = np.zeros_like(Da, dtype=bool)
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


def cluster(X, estimator="argmax", **parameters):

    estimators = {

        "argmax": partial(ArgMax),

        "kmeans": partial(C.KMeans),

        "qcores": partial(QCluster),

        "spectral": partial(C.SpectralClustering,
                            affinity="nearest_neighbors",
                            eigen_solver="amg",
                            assign_labels="discretize",
                            ),

        "agglo_ward": partial(C.AgglomerativeClustering,
                              affinity="euclidean",
                              compute_full_tree='auto',
                              linkage='ward',
                              distance_threshold=None, ),

        "agglo_single": partial(C.AgglomerativeClustering,
                                affinity="precomputed",
                                compute_full_tree='auto',
                                linkage='single',
                                distance_threshold=None, ),

        "agglo_complete": partial(C.AgglomerativeClustering,
                                  affinity="precomputed",
                                  compute_full_tree='auto',
                                  linkage='complete',
                                  distance_threshold=None, )

    }

    if estimator in {"agglo_single", "agglo_complete"}:
        X_, _, _ = distance_matrices(X, metric=parameters.get("metric", "euclidean"))
    else:
        X_ = X

    cls = estimators[estimator](**parameters)
    cls.fit(X_)

    return cls
