import torch
from torch import nn as nn


class ParametrizedGaussian(nn.Module):
    """
    Parametrized Gaussian (often found in variational auto-encoders)
    """

    def __init__(self, input_dim: int, z_dim: int, pre_activation=nn.Identity(), return_params=True, bias=False):
        super(ParametrizedGaussian, self).__init__()
        self.fc1 = nn.Linear(input_dim, z_dim, bias=bias)
        self.fc2 = nn.Linear(input_dim, z_dim, bias=bias)
        self.z_dim = z_dim
        self.pre_act = pre_activation if pre_activation is not None else lambda x: x
        self.return_params = return_params

    def forward(self, h):
        mu, logvar = self.fc1(self.pre_act(h)), self.fc2(self.pre_act(h))
        std = logvar.mul(0.5).exp_()
        eps = torch.randn(*mu.size()).to(h.device)
        z = mu + std * eps
        if self.return_params:
            return z, mu, std
        return z
