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
    bias: Optional[bool] = False
    pre_activation: Optional[nn.Module] = nn.Identity()
    return_params: Optional[bool] = True

    def __postinit__(self):
        nn.Module.__init__(self)
        self.fc1 = nn.Linear(self.input_dim, self.z_dim, bias=self.bias)
        self.fc2 = nn.Linear(self.input_dim, self.z_dim, bias=self.bias)
        self.pre_act = self.pre_activation if self.pre_activation is not None else lambda x: x

    def forward(self, h):
        mu, logvar = self.fc1(self.pre_act(h)), self.fc2(self.pre_act(h))
        std = logvar.mul(0.5).exp_()
        eps = torch.randn(*mu.size()).to(h.device)
        z = mu + std * eps
        if self.return_params:
            return z, mu, std
        return z
