import torch
from torch import nn as nn
from typing import Optional
from dataclasses import dataclass

__all__ = [
    'ParametrizedGaussian'
]


@dataclass(init=True, repr=False, eq=False, frozen=False, unsafe_hash=True)
class ParametrizedGaussian(nn.Module):
    """
    Parametrized Gaussian (often found in variational auto-encoders)
    """
    input_dim: int
    z_dim: int
    bias: bool = False
    return_params: bool = True

    def __post_init__(self):
        nn.Module.__init__(self)
        self.fc = nn.Linear(self.input_dim, self.z_dim * 2, bias=self.bias)

    def forward(self, h):
        mu, logvar = torch.chunk(self.fc(h), 2, dim=-1)
        std = logvar.mul(0.5).exp_()
        eps = torch.randn(*mu.size()).to(h.device)
        z = mu + std * eps
        if self.return_params:
            return z, mu, std
        return z
