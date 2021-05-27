import torch
import torch.nn as nn

__all__ = [
    'angular_distance',
    'cosine_similarity',
    'mean_L1_prop'
]


def mean_L1_prop(output, target):
    L = nn.L1Loss(reduction="none")(output, target).sum(dim=(0, -1), keepdim=True)
    return 100 * (L / target.abs().sum(dim=(0, -1), keepdim=True)).mean()


def cosine_similarity(X, Y, eps=1e-10):
    """
    safely computes the cosine similarity between matrices X and Y.

    Shapes:
    -------
    X : (*, N, D)
    Y : (*, M, D)
    D_xy : (*, N, M)

    Notes:
    ------
    The need for this function arises from the fact that torch.nn.CosineSimilarity only computes the
    diagonal of D_xy, as in cosine_sim(output, target)
    """
    if not isinstance(eps, torch.Tensor):
        eps = torch.tensor(eps).to(X)

    dot_prod = torch.matmul(X, Y.transpose(-2, -1))
    norms = torch.norm(X, p=2, dim=-1).unsqueeze(-1) * torch.norm(Y, p=2, dim=-1).unsqueeze(-2)
    cos_theta = dot_prod / torch.maximum(norms, eps)
    return cos_theta


def angular_distance(X, Y, eps=1e-20):
    """
    angular distance is a valid distance metric based on the cosine similarity
    see https://en.wikipedia.org/wiki/Cosine_similarity#Angular_distance_and_similarity

    Shapes:
    -------
    X : (*, N, D)
    Y : (*, M, D)
    D_xy : (*, N, M)
    """
    if not isinstance(eps, torch.Tensor):
        eps = torch.tensor(eps).to(X)

    def safe_acos(x):
        # torch.acos returns nan near -1 and 1... see https://github.com/pytorch/pytorch/issues/8069
        return torch.acos(torch.clamp(x, min=-1 + eps / 2, max=1 - eps / 2))

    have_negatives = torch.any(X < 0) or torch.any(Y < 0)
    cos_theta = cosine_similarity(X, Y, eps)

    pi = torch.acos(torch.zeros(1)).item() * 2
    D_xy = (1 + int(not have_negatives)) * safe_acos(cos_theta) / pi

    return D_xy