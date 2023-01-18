from typing import Literal, Tuple
from numpy import pi
import torch
from torch import distributions as D, nn as nn

__all__ = [
    "OutputWrapper",
    "CategoricalSampler",
    "MixOfLogisticsSampler",
    "MixOfLogisticsLoss",
    "MixOfGaussianLoss",
    "MixOfGaussianSampler",
    "MixOfLaplaceLoss",
    "MixOfLaplaceSampler"
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


def as_tensor(temperature, tensor):
    if not isinstance(temperature, torch.Tensor):
        if isinstance(temperature, (int, float)):
            temperature = [temperature]
        temperature = torch.tensor(temperature)
    if temperature.ndim != tensor.ndim:
        temperature = temperature.view(*temperature.shape, *([1] * (tensor.ndim - temperature.ndim)))
    return temperature.to(tensor.device)


class CategoricalSampler(nn.Module):
    sampling_params = {"temperature"}

    def forward(self, logits, *, temperature=None):
        if self.training:
            return logits
        if temperature is None:
            return logits.argmax(dim=-1)
        temperature = as_tensor(temperature, logits)
        logits = logits / temperature
        logits = logits - logits.logsumexp(-1, keepdim=True)
        if logits.dim() > 2:
            o_shape = logits.shape
            logits = logits.view(-1, o_shape[-1])
            return torch.multinomial(logits.exp_(), 1).reshape(*o_shape[:-1])
        return torch.multinomial(logits.exp_(), 1)


class _MixOfRealScalarBase(nn.Module):
    mixture_class = None
    # Would also work with D.Gumbel, D.Cauchy

    def __init__(
            self,
            n_components: int = 1,
            reduction: Literal["sum", "mean", "none"] = "mean",
            max_scale: float = .5,
            beta: float = 1 / 15,
            weight_variance: float = .5,
            weight_l1: float = 0.,
            clamp_samples: Tuple[float, float] = (-1., 1.)
    ):
        super(_MixOfRealScalarBase, self).__init__()
        self.reduction = reduction
        self.n_components = n_components
        self.max_scale = max_scale
        self.beta = beta
        self.weight_variance = weight_variance
        self.weight_l1 = weight_l1
        self.clamp_samples = clamp_samples

    def mixture(self,
                params: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
                temperature=None
                ):
        weight, loc, scale = params
        # loc = torch.tanh(loc)  # this performs poorly!
        scale = torch.sigmoid(scale * self.beta) * self.max_scale
        # scale = torch.maximum(scale.abs(), torch.tensor(1e-4, device=scale.device))
        if temperature is not None:
            scale = scale * as_tensor(temperature, scale)
        return D.MixtureSameFamily(
            # todo:
            #  - logits could also be scaled by temperature when sampling
            #  - scale could have different activations (SoftPlus, exp, ...)
            #  - weights and/or scale can be made constants (works fine, tend to freezes...)
            D.Categorical(logits=weight),
            self.mixture_class(loc, scale)
        )


class _MixOfRealScalarLoss(_MixOfRealScalarBase):
    def forward(self,
                params: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
                targets: torch.Tensor
                ):
        mixture = self.mixture(params)
        probs = mixture.log_prob(targets)
        if self.weight_variance > 0.:
            var_term = mixture.variance.mean() * self.weight_variance
        else:
            var_term = 0.
        if self.weight_l1 > 0.:
            l1_term = nn.L1Loss(reduction='mean')(mixture.mean, targets) * self.weight_l1
        else:
            l1_term = 0.
        if self.reduction != "none":
            return - getattr(torch, self.reduction)(probs) + var_term + l1_term
        return probs


class _MixOfRealScalarSampler(_MixOfRealScalarBase):
    sampling_params = {"temperature"}

    def forward(
            self,
            params: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
            *, temperature=None,
    ):
        if self.training:
            return params
        mixture = self.mixture(params, temperature)
        # TODO :
        #  - sample() might be a bottleneck, maybe good to do it ourselves?...
        #  - sample() takes only one component
        #       -> implement soft sampling with weighted sum?
        return mixture.sample((1,)).clamp(*self.clamp_samples).squeeze(0)


# noinspection PyAbstractClass
class LogisticDistribution(D.TransformedDistribution):
    def __init__(self, loc, scale):
        super(LogisticDistribution, self).__init__(
            D.Uniform(torch.tensor(0., device=loc.device), torch.tensor(1., device=loc.device)),
            [D.SigmoidTransform().inv, D.AffineTransform(loc, scale)]
        )
        self._loc = loc
        self._scale = scale

    @property
    def variance(self):
        return self._scale.pow(2)*(pi**2)/3

    @property
    def mean(self):
        return self._loc


class MixOfLogisticsBase(_MixOfRealScalarBase):
    mixture_class = LogisticDistribution


class MixOfLogisticsLoss(MixOfLogisticsBase, _MixOfRealScalarLoss):
    pass


class MixOfLogisticsSampler(MixOfLogisticsBase, _MixOfRealScalarSampler):
    pass


class MixOfGaussianBase(_MixOfRealScalarBase):
    mixture_class = D.Normal


class MixOfGaussianLoss(MixOfGaussianBase, _MixOfRealScalarLoss):
    pass


class MixOfGaussianSampler(MixOfGaussianBase, _MixOfRealScalarSampler):
    pass


class MixOfLaplaceBase(_MixOfRealScalarBase):
    mixture_class = D.Laplace


class MixOfLaplaceLoss(MixOfLaplaceBase, _MixOfRealScalarLoss):
    pass


class MixOfLaplaceSampler(MixOfLaplaceBase, _MixOfRealScalarSampler):
    pass


class _MixOfRealVectorBase(nn.Module):
    mixture_class = None

    def __init__(
            self,
            vector_dim: int = 1,
            n_components: int = 1,
            reduction: Literal["sum", "mean", "none"] = "mean",
            max_scale: float = .5,
            beta: float = 1 / 15,
            weight_variance: float = .5,
            weight_l1: float = 0.,
            clamp_samples: Tuple[float, float] = (-1., 1.)
    ):
        super(_MixOfRealVectorBase, self).__init__()
        self.vector_dim = vector_dim
        self.reduction = reduction
        self.n_components = n_components
        self.max_scale = max_scale
        self.beta = beta
        self.weight_variance = weight_variance
        self.weight_l1 = weight_l1
        self.clamp_samples = clamp_samples

    def mixture(self,
                fused_params: torch.Tensor,
                ):
        weight, params = fused_params[..., 0], fused_params[..., 1:]
        loc, scale = params.chunk(2, -1)

        return D.MixtureSameFamily(
            D.Categorical(logits=weight), self.mixture_class(loc, scale, self.vector_dim)
        )


# noinspection PyAbstractClass
class LogisticVectorDistribution(D.TransformedDistribution):
    def __init__(self, loc, scale, dim):
        super(LogisticVectorDistribution, self).__init__(
            D.Independent(D.Uniform(
                torch.tensor(0., device=loc.device), torch.tensor(1., device=loc.device)
            ).expand((dim,)), 1),
            [D.SigmoidTransform().inv, D.AffineTransform(loc, scale)]
        )
        self._loc = loc
        self._scale = scale

    @property
    def variance(self):
        return self._scale.pow(2)*(pi**2)/3

    @property
    def mean(self):
        return self._loc


# noinspection PyAbstractClass
class GaussianVectorDistribution(D.Independent):

    def __init__(self, loc, scale, dim):
        super(GaussianVectorDistribution, self).__init__(D.Normal(loc, scale).expand((dim,)), 1)


# noinspection PyAbstractClass
class LaplaceVectorDistribution(D.Independent):
    def __init__(self, loc, scale, dim):
        super(LaplaceVectorDistribution, self).__init__(D.Laplace(loc, scale).expand((dim,)), 1)
