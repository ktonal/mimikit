from enum import auto
import torch
from torch import nn

from ..config import Config, private_runtime_field
from ..utils import AutoStrEnum

__all__ = [
    "ActivationEnum",
    "ActivationConfig",
    "Abs",
    "Sin",
    "Cos",
    "GatingUnit",
    "StaticScaledActivation",
    "ScaledActivation",
]


class ActivationEnum(AutoStrEnum):
    Tanh = auto()
    Sigmoid = auto()
    Mish = auto()
    ReLU = auto()
    Identity = auto()
    Abs = auto()
    Sin = auto()
    Cos = auto()


class ActivationConfig(Config):
    act: ActivationEnum = "Identity"
    scaled: bool = False
    static: bool = False
    with_rate: bool = False

    dim: int = private_runtime_field(None)

    def get(self):
        try:
            a = getattr(nn, self.act)()
        except AttributeError:
            a = globals()[self.act]()
        if self.scaled:
            if self.static:
                return StaticScaledActivation(a, self.dim, self.with_rate)
            return ScaledActivation(a, self.dim, self.with_rate)
        return a


class Abs(nn.Module):

    def forward(self, x):
        return x.abs()


class Sin(nn.Module):
    def forward(self, x):
        return torch.sin(x)


class Cos(nn.Module):
    def forward(self, x):
        return torch.cos(x)


class GatingUnit(nn.Module):

    def __init__(self):
        super(GatingUnit, self).__init__()
        self.act_f = nn.Tanh()
        self.act_g = nn.Sigmoid()

    def forward(self, x_f, x_g):
        return self.act_f(x_f) * self.act_g(x_g)


class ScaledActivation(nn.Module):
    def __init__(self, activation, dim, with_rate=True):
        super(ScaledActivation, self).__init__()
        self.activation = activation
        self.s = nn.Linear(dim, dim)
        self.r = nn.Linear(dim, dim) if with_rate else lambda x: 1.
        self.dim = dim

    def forward(self, x):
        s, r = self.s(x), self.r(x)
        return self.activation(r * x / s) * s


class StaticScaledActivation(nn.Module):
    def __init__(self, activation, dim, with_rate=True):
        super(StaticScaledActivation, self).__init__()
        self.activation = activation
        self.s = nn.Parameter(torch.rand(dim, ) * 20, )
        self.r = nn.Parameter(torch.rand(dim, ) * .1, ) if with_rate else 1.
        self.dim = dim

    def forward(self, x):
        s, r = self.s.to(x.device), self.r.to(x.device)
        return self.activation(r * x / s) * s


PI = torch.acos(torch.zeros(1)).item()

# Legacy activations for fft Phases:
# here `phs` is real output: (B, T, n_fft//2+1)


class PhaseA(nn.Module):
    def __init__(self, dim):
        super(PhaseA, self).__init__()
        self.psis = nn.Parameter(torch.ones(dim, ))
        self.tanh = nn.Tanh()

    def forward(self, phs):
        return torch.cos(self.tanh(phs) * self.psis) * PI


class PhaseB(nn.Module):
    def __init__(self, dim):
        super(PhaseB, self).__init__()
        self.psis = nn.Parameter(torch.ones(dim))

    def forward(self, phs):
        return torch.cos(phs * self.psis) * PI


class PhaseC(nn.Module):
    def __init__(self):
        super(PhaseC, self).__init__()
        self.tanh = nn.Tanh()

    def forward(self, phs):
        return self.tanh(phs) * PI
