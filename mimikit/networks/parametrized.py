import torch
from torch import nn as nn
from torch.distributions import SigmoidTransform

__all__ = [
    'ParametrizedGaussian',
    "ParametrizedLinear"
]


class ParametrizedGaussian(nn.Module):
    def __init__(self,
                 input_dim: int,
                 z_dim: int,
                 bias: bool = False,
                 min_std: float = 1e-4,
                 return_params: bool = True,
                 ):
        super(ParametrizedGaussian, self).__init__()
        self.fc = nn.Linear(input_dim, z_dim * 2, bias=bias)
        self.min_std = min_std
        self.return_params = return_params

    def forward(self, h):
        mu, logvar = torch.chunk(self.fc(h), 2, dim=-1)
        std = logvar.mul(0.5).exp().clamp(min=self.min_std)
        eps = torch.randn(*mu.size()).to(h.device)
        z = mu + std * eps
        if self.return_params:
            return z, mu, std
        return z


class ParametrizedLinear(nn.Module):

    def __init__(self, in_dim, out_dim, bias=True, as_1x1_conv=False):
        super(ParametrizedLinear, self).__init__()
        if as_1x1_conv:
            self.params = nn.Conv1d(in_dim, out_dim * 3, 1, bias=bias)
        else:
            self.params = nn.Linear(in_dim, out_dim * 3, bias=bias)
        self.chunk_dim = -1 if not as_1x1_conv else -2

    def forward(self, x):
        x_hat, a, b = torch.chunk(self.params(x), 3, dim=self.chunk_dim)
        return x_hat.mul(a).add(b)


class ParametrizedLogistic(nn.Module):

    def __init__(self, in_dim, out_dim,
                 bias=True,
                 min_std: float = 1e-3,
                 as_1x1_conv=False):
        super(ParametrizedLogistic, self).__init__()
        if as_1x1_conv:
            self.params = nn.Conv1d(in_dim, out_dim * 2, 1, bias=bias)
        else:
            self.params = nn.Linear(in_dim, out_dim * 2, bias=bias)
        self.chunk_dim = -1 if not as_1x1_conv else -2
        self.min_std = min_std

    def forward(self, x):
        mu, std = torch.chunk(self.params(x), 2, dim=self.chunk_dim)
        y = torch.rand_like(mu)
        finfo = torch.finfo(y.dtype)
        y = y.clamp(min=finfo.tiny, max=1. - finfo.eps)
        y = y.log() - (-y).log1p()
        return mu + y * std
