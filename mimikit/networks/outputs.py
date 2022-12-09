from typing import Literal, Tuple, Optional

import torch
from torch import distributions as D, nn as nn


def Logistic(loc, scale) -> D.TransformedDistribution:
    """credits to https://github.com/pytorch/pytorch/issues/7857"""
    return D.TransformedDistribution(
        D.Uniform(0, 1),
        [D.SigmoidTransform().inv, D.AffineTransform(loc, scale)]
    )


class MixtureOfLogistics(nn.Module):
    # TODO: LossModule & InferModule
    def __init__(
            self,
            k_components: int,
            reduction: Literal["sum", "mean", "none"],
            clamp_samples: Tuple[float, float] = (-1., 1.)
    ):
        super(MixtureOfLogistics, self).__init__()
        self.k_components = k_components
        self.reduction = reduction
        self.clamp = clamp_samples

    def forward(self,
                params: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
                targets: Optional[torch.Tensor] = None
                ):
        weight, loc, scale = params
        o_shape, K = loc.shape, self.k_components
        assert weight.size(-1) == loc.size(-1) == scale.size(-1) == K
        weight, loc, scale = weight.view(-1, K), loc.view(-1, K), scale.view(-1, K)
        mixture = D.MixtureSameFamily(
            D.Categorical(logits=weight), Logistic(loc, scale)
        )
        if targets is not None:
            probs = mixture.log_prob(targets.view(-1))
            if self.reduction != "none":
                return getattr(torch, self.reduction)(probs)
            return probs
        return mixture.sample((1,)).clamp(*self.clamp).reshape(*o_shape[:-1])