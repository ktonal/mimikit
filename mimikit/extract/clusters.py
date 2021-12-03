import numpy as np
from sklearn.metrics import pairwise_distances as pwd
from sklearn.neighbors import KNeighborsTransformer


def k_neighbors_pack(X, k, metric="euclidean"):
    D = pwd(X, X, metric=metric, n_jobs=-1)
    kn = KNeighborsTransformer(mode="connectivity", n_neighbors=k,
                               metric="precomputed", n_jobs=-1)
    adj = kn.fit_transform(D)
    dists, idx = kn.kneighbors(X, k, return_distance=True)
    return D, adj, dists, idx
