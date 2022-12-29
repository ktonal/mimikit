from typing import Literal, Tuple

import torch
from torch import distributions as D, nn as nn


__all__ = [
    "OutputWrapper",
    "CategoricalSampler",
    "MoLBase",
    "MoLSampler",
    "MoLLoss",
    "MoLVectorBase",
    "MoLVectorLoss",
    "MoLVectorSampler"
]


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

    @property
    def sampling_params(self):
        return getattr(self.sampler, "sampling_params", {})


class CategoricalSampler(nn.Module):
    sampling_params = {"temperature"}

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


class MoLBase(nn.Module):
    def __init__(
            self,
            n_components: int,
    ):
        super(MoLBase, self).__init__()
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
    

class MoLLoss(MoLBase, nn.Module):
    def __init__(
            self,
            n_components: int,
            reduction: Literal["sum", "mean", "none"],
    ):
        super(MoLLoss, self).__init__(n_components=n_components)
        self.reduction = reduction
        
    def forward(self,
                params: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
                targets: torch.Tensor
                ):
        mixture = self.mixture(params)
        probs = mixture.log_prob(targets.view(-1))
        if self.reduction != "none":
            return - getattr(torch, self.reduction)(probs)
        return probs


class MoLSampler(MoLBase):
    def __init__(
            self,
            n_components: int,
            clamp_samples: Tuple[float, float] = (-1., 1.)
    ):
        super(MoLSampler, self).__init__(n_components=n_components)
        self.clamp = clamp_samples
        
    def forward(
            self,
            params: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    ):
        if self.training:
            return params
        mixture = self.mixture(params)
        return mixture.sample((1,)).clamp(*self.clamp)


def LogisticVector(loc, scale, dim) -> D.TransformedDistribution:
    """credits to https://github.com/pytorch/pytorch/issues/7857"""
    return D.TransformedDistribution(
        D.Independent(D.Uniform(0, 1).expand((dim,)), 1),
        [D.SigmoidTransform().inv, D.AffineTransform(loc, scale)]
    )


class MoLVectorBase(nn.Module):
    def __init__(
            self,
            n_components: int,
            vector_dim: int,
    ):
        super(MoLVectorBase, self).__init__()
        self.n_components = n_components
        self.vector_dim = vector_dim

    def mixture(self,
                fused_params: torch.Tensor,
                ):
        weight, params = fused_params[..., 0], fused_params[..., 1:]
        loc, scale = params.chunk(2, -1)
        dim, K = self.vector_dim, self.n_components
        # assert weight.size(-1) == loc.size(-2) == scale.size(-2) == K
        # weight, loc, scale = weight.view(-1, K), loc.view(-1, K, dim), scale.view(-1, K, dim)
        return D.MixtureSameFamily(
            D.Categorical(logits=weight), LogisticVector(loc, scale, self.vector_dim)
        )


class MoLVectorLoss(MoLVectorBase, nn.Module):
    def __init__(
            self,
            n_components: int,
            vector_dim: int,
            reduction: Literal["sum", "mean", "none"],
    ):
        super(MoLVectorLoss, self).__init__(n_components=n_components, vector_dim=vector_dim)
        self.reduction = reduction

    def forward(self,
                fused_params: torch.Tensor,
                targets: torch.Tensor
                ):
        mixture = self.mixture(fused_params)
        probs = mixture.log_prob(targets.view(-1, self.vector_dim))
        if self.reduction != "none":
            return - getattr(torch, self.reduction)(probs)
        return probs


class MoLVectorSampler(MoLVectorBase):
    def __init__(
            self,
            n_components: int,
            vector_dim: int,
            clamp_samples: Tuple[float, float] = (-1., 1.)
    ):
        super(MoLVectorSampler, self).__init__(n_components=n_components, vector_dim=vector_dim)
        self.clamp = clamp_samples

    def forward(
            self,
            fused_params: torch.Tensor,
    ):
        if self.training:
            return fused_params
        mixture = self.mixture(fused_params)
        return mixture.sample((1,)).clamp(*self.clamp).squeeze(0)