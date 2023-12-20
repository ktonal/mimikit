from enum import auto
from typing import Dict

import torch
from torch import nn
import dataclasses as dtc

from ..config import Config, private_runtime_field
from ..utils import AutoStrEnum

__all__ = [
    "ActivationEnum",
    "ActivationConfig",
    "Abs",
    "Sin",
    "Cos",
    "GatingUnit",
    "SnakeSin",
    "SnakeCos",
    "StaticScaledActivation",
    "ScaledActivation",
    "PhaseA",
    "PhaseB",
    "PhaseC"
]


class ActivationEnum(AutoStrEnum):
    Tanh = auto()
    Sigmoid = auto()
    Mish = auto()
    ReLU = auto()
    Softplus = auto()
    Identity = auto()
    Abs = auto()
    PhaseA = auto()
    PhaseB = auto()
    PhaseC = auto()
    Sin = auto()
    Cos = auto()
    GLU = auto()
    Softmax = auto()
    ExpMSq = auto()
    SnakeSin = auto()
    SnakeCos = auto()


@dtc.dataclass
class ActivationConfig(Config, type_field=False):
    act: ActivationEnum = "Identity"
    scaled: bool = False
    static: bool = False
    with_rate: bool = False
    params: Dict = dtc.field(default_factory=lambda: {})
    dim: int = private_runtime_field(None)

    def get(self):
        try:
            a = getattr(nn, self.act)
        except AttributeError:
            a = globals()[self.act]
        if self.act in ("PhaseA", "PhaseB"):
            return a(self.dim)
        if self.act == "Softmax":
            a = a(dim=-1, **self.params)
        else:
            a = a(**self.params)
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
    def __init__(self, with_rate=False):
        super(Cos, self).__init__()
        self.with_rate = with_rate
        self.rates = None

    def forward(self, x):
        if self.with_rate:
            if self.rates is None:
                self.rates = nn.Parameter(torch.rand(x.size(-1)).to(x.device))
            x = x * self.rates
        return torch.cos(x)


class ExpMSq(nn.Module):

    def forward(self, x):
        return torch.exp(- x.pow(2))


class GatingUnit(nn.Module):

    def __init__(self):
        super(GatingUnit, self).__init__()
        self.act_f = nn.Tanh()
        self.act_g = nn.Sigmoid()

    def forward(self, x_f, x_g):
        return self.act_f(x_f) * self.act_g(x_g)


# @torch.jit.script
def snake_sin(x, alpha):
    x = x + (alpha + 1e-9).reciprocal() * torch.sin(alpha * x).pow(2)
    return x


# @torch.jit.script
def snake_cos(x, alpha):
    x = x + (alpha + 1e-9).reciprocal() * torch.cos(alpha * x).pow(2)
    return x


class SnakeSin(nn.Module):
    def __init__(self, conv_order=True):
        super().__init__()
        self.conv_order = conv_order
        self.alpha = None

    def forward(self, x):
        if self.alpha is None:
            if self.conv_order:
                self.alpha = nn.Parameter(torch.ones(1, x.shape[1], 1, device=x.device))
            else:
                self.alpha = nn.Parameter(torch.ones(x.shape[-1], device=x.device))

        return snake_sin(x, self.alpha)


class SnakeCos(nn.Module):
    def __init__(self, conv_order=True):
        super().__init__()
        self.conv_order = conv_order
        self.alpha = None

    def forward(self, x):
        if self.alpha is None:
            if self.conv_order:
                self.alpha = nn.Parameter(torch.ones(1, x.shape[1], 1, device=x.device))
            else:
                self.alpha = nn.Parameter(torch.ones(x.shape[-1], device=x.device))
        return snake_cos(x, self.alpha)


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
        self.s = nn.Parameter(torch.ones(dim), )
        self.r = nn.Parameter(torch.ones(dim, )) if with_rate else torch.tensor([1.])
        self.dim = dim

    def forward(self, x):
        s, r = self.s.to(x.device).expand(*(1,) * (len(x.size()) - 1), self.dim),\
               self.r.to(x.device).expand(*(1,) * (len(x.size()) - 1), self.dim)
        return self.activation(r * x / s) * s

# softplus, - logsigmoid,

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
