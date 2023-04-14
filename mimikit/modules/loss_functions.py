import torch
import torch.nn as nn

__all__ = [
    'MeanL1Prop',
    'Mean2dDiff',
    "CosineSimilarity",
    "AngularDistance",
]


class MeanL1Prop(nn.Module):
    def __init__(self, raise_on_nan=True, eps=1e-8):
        super(MeanL1Prop, self).__init__()
        self.raise_on_nan = raise_on_nan
        self.eps = eps
        self.l1loss = nn.L1Loss(reduction="none")

    def forward(self, output, target):
        if self.raise_on_nan and torch.any(torch.isnan(output)):
            raise RuntimeError("nan values in output")
        L = self.l1loss(output, target).sum(dim=(0, -1,), keepdim=True)
        target_sums = target.abs().sum(dim=(0, -1,), keepdim=True)
        # make the upcoming division safe
        prop = torch.maximum(L.detach(), torch.tensor(self.eps).to(L.device))
        target_sums = target_sums + (target_sums < 1.).float() * prop
        if self.raise_on_nan and torch.any(torch.isnan(target_sums)):
            raise RuntimeError("nan values in target_sums")
        L = (L / target_sums).mean()
        return L


class Mean2dDiff(nn.Module):
    def __init__(self, raise_on_nan=True, eps=1e-8):
        super(Mean2dDiff, self).__init__()
        self.mean_l1_prop = MeanL1Prop(raise_on_nan, eps)

    def forward(self, output, target):
        """compute the mean_L1_prop loss of the differences along the 2 last axes of `output` and `target`"""
        Lw = self.mean_l1_prop((output[:, :, 1:] - output[:, :, :-1]), target[:, :, 1:] - target[:, :, :-1], )
        Lh = self.mean_l1_prop((output[:, 1:] - output[:, :-1]), target[:, 1:] - target[:, :-1], )
        return Lw + Lh


class CosineSimilarity(nn.Module):
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
    def __init__(self, eps=1e-8):
        super(CosineSimilarity, self).__init__()
        if not isinstance(eps, torch.Tensor):
            eps = torch.tensor(eps)
        self.register_buffer("eps", eps)

    def forward(self, X, Y):
        dot_prod = torch.matmul(X, Y.transpose(-2, -1))
        norms = torch.norm(X, p=2, dim=-1).unsqueeze(-1) * torch.norm(Y, p=2, dim=-1).unsqueeze(-2)
        cos_theta = dot_prod.div_(torch.maximum(norms, self.eps.to(X.device)))
        return cos_theta


class AngularDistance(nn.Module):
    def __init__(self, eps=1e-8):
        super(AngularDistance, self).__init__()
        if not isinstance(eps, torch.Tensor):
            eps = torch.tensor(eps)
        self.register_buffer("eps", eps)
        self.cosine_sim = CosineSimilarity(eps=eps)

    def safe_acos(self, x):
        # torch.acos returns nan near -1 and 1... see https://github.com/pytorch/pytorch/issues/8069
        eps = self.eps.to(x.device)
        return torch.acos(torch.clamp(x, min=-1 + eps / 2, max=1 - eps / 2))

    def forward(self, X, Y):
        """
        angular distance is a valid distance metric based on the cosine similarity
        see https://en.wikipedia.org/wiki/Cosine_similarity#Angular_distance_and_similarity

        Shapes:
        -------
        X : (*, N, D)
        Y : (*, M, D)
        D_xy : (*, N, M)
        """
        have_negatives = torch.any(X < 0) or torch.any(Y < 0)
        cos_theta = self.cosine_sim(X, Y)
        pi = torch.acos(torch.zeros(1)).item() * 2
        D_xy = (1 + int(not have_negatives)) * self.safe_acos(cos_theta) / pi
        return D_xy
