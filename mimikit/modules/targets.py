import torch
from torch import nn as nn

__all__ = [
    "OutputWrapper",
    "CategoricalSampler",
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
