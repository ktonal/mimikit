import torch
import numpy as np

from ..modules import angular_distance


def nearest_neighbor(X, Y):
    """
    computes nearest neighbor by angular distance
    """
    D_xy = angular_distance(X, Y)
    dists, nn = torch.min(D_xy, dim=-1)
    return dists, nn


def torch_frame(x, frame_size, hop_length):
    """
    helper to reshape an array into frames
    """
    N = x.size(-1)
    org_size = x.size()[:-1]
    tmp_0 = np.prod(tuple(org_size))
    new_dims = (1 + int((N - frame_size) / hop_length), frame_size)
    framed = torch.as_strided(x.reshape(-1, N), (int(tmp_0), *new_dims), (N, hop_length, 1))
    return framed.reshape(*org_size, *new_dims)


def repeat_rate(x, frame_size, hop_length):
    """
    frames x and compute repeat-rate per frame
    """
    framed = torch_frame(x, frame_size, hop_length)
    uniques = torch.tensor([torch.unique(row).size(0) for row in framed.reshape(-1, framed.size(-1))])
    return (1 - (uniques-1) / (frame_size-1)).reshape(framed.size()[:-1], -1)


# for scoring outputs
def cum_entropy(neighbors, reduce="sum", neg_diff=True):
    """`neighbors` is a (Time, ) Long Tensor [no batch dim!]"""
    items, idx = torch.unique(neighbors, return_inverse=True)
    cum_probs = torch.zeros(items.size(0), neighbors.size(0))
    cum_probs[idx, torch.arange(neighbors.size(0))] = 1
    cum_probs = torch.cumsum(cum_probs, dim=1)
    cum_probs = cum_probs / cum_probs.sum(dim=0, keepdims=True)
    e_wrt_t = (-cum_probs * torch.where(cum_probs > 0, torch.log(cum_probs), cum_probs)).sum(dim=0)
    if neg_diff:
        diff = torch.diff(e_wrt_t, 1, 1, append=torch.tensor([0]).to(e_wrt_t))
        e_wrt_t = (torch.sign(diff) * e_wrt_t)
    return e_wrt_t.sum(dim=0) if reduce == 'sum' else e_wrt_t


def hist_transform(neighbors, bins=256):
    """transform series of neighbors into hist vectors"""
    if neighbors.dim() > 1:
        x_dims = neighbors.shape[:-1]
        h = torch.stack([torch.histc(xi, bins=bins) for xi in neighbors.view(-1, neighbors.size(-1))]).reshape(*x_dims, bins)
        return h
    return torch.histc(neighbors, bins=bins)
