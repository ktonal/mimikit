from typing import Literal, Tuple, Optional

import torch
from torch import distributions as D, nn as nn


class OutputWrapper(nn.Module):
    def __init__(self, estimator: nn.Module, sampler: nn.Module):
        super(OutputWrapper, self).__init__()
        self.estimator = estimator
        self.sampler = sampler

    def forward(self, *inputs, **sampler_kwargs):
        params = self.estimator(*inputs)
        if not self.training:
            return self.sampler(params, **sampler_kwargs)
        return params


class CategoricalSampler(nn.Module):
        
    def forward(self, logits, *, temperature=None):
        if self.training:
            return logits
        if temperature is None:
            return logits.argmax(dim=-1)
        if not isinstance(temperature, torch.Tensor):
            if isinstance(temperature, (int, float)):
                temperature = [temperature]
            temperature = torch.tensor(temperature)
        if temperature.ndim != logits.ndim:
            temperature = temperature.view(*temperature.shape, *([1] * (logits.ndim - temperature.ndim)))
        logits = logits / temperature.to(logits.device)
        logits = logits - logits.logsumexp(-1, keepdim=True)
        if logits.dim() > 2:
            o_shape = logits.shape
            logits = logits.view(-1, o_shape[-1])
            return torch.multinomial(logits.exp_(), 1).reshape(*o_shape[:-1])
        return torch.multinomial(logits.exp_(), 1)
    

def Logistic(loc, scale) -> D.TransformedDistribution:
    """credits to https://github.com/pytorch/pytorch/issues/7857"""
    return D.TransformedDistribution(
        D.Uniform(0, 1),
        [D.SigmoidTransform().inv, D.AffineTransform(loc, scale)]
    )


class MixtureOfLogisticsBase(nn.Module):
    def __init__(
            self,
            n_components: int,
    ):
        super(MixtureOfLogisticsBase, self).__init__()
        self.n_components = n_components

    def mixture(self,
                params: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
                ):
        weight, loc, scale = params
        o_shape, K = loc.shape, self.n_components
        assert weight.size(-1) == loc.size(-1) == scale.size(-1) == K
        weight, loc, scale = weight.view(-1, K), loc.view(-1, K), scale.view(-1, K)
        return D.MixtureSameFamily(
            D.Categorical(logits=weight), Logistic(loc, scale)
        )
    

class MixtureOfLogisticsLoss(MixtureOfLogisticsBase, nn.Module):
    def __init__(
            self,
            n_components: int,
            reduction: Literal["sum", "mean", "none"],
    ):
        super(MixtureOfLogisticsLoss, self).__init__(n_components=n_components)
        self.reduction = reduction
        
    def forward(self,
                params: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
                targets: torch.Tensor):
        mixture = self.mixture(params)
        probs = mixture.log_prob(targets.view(-1))
        if self.reduction != "none":
            return - getattr(torch, self.reduction)(probs)
        return probs


class MixtureOfLogisticsSampler(MixtureOfLogisticsBase):
    def __init__(
            self,
            n_components: int,
            clamp_samples: Tuple[float, float] = (-1., 1.)
    ):
        super(MixtureOfLogisticsSampler, self).__init__(n_components=n_components)
        self.clamp = clamp_samples
        
    def forward(
            self,
            params: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    ):
        if self.training:
            return params
        mixture = self.mixture(params)
        o_shape = params[0].shape
        return mixture.sample((1,)).clamp(*self.clamp).reshape(*o_shape[:-1])

