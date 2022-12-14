from enum import auto
import torch
from torch import nn

from ..config import Config
from ..utils import AutoStrEnum

__all__ = [
    "ActivationEnum",
    "ActivationConfig",
    "Abs",
    "GatingUnit",
    "StaticScaledActivation",
    "ScaledActivation",
    "ScaledAbs",
    "ScaledTanh",
    "ScaledSigmoid"
]


class ActivationEnum(AutoStrEnum):
    Tanh = auto()
    Sigmoid = auto()
    Mish = auto()
    ReLU = auto()
    Identity = auto()
    Abs = auto()

    def object(self):
        try:
            return getattr(nn, self)()
        except AttributeError:
            return globals()[self]


class ActivationConfig(Config):
    act: ActivationEnum = "Identity"
    scaled: bool = False
    with_range: bool = False
    static: bool = False


class Abs(nn.Module):

    def forward(self, x):
        return x.abs()


class GatingUnit(nn.Module):

    def __init__(self):
        super(GatingUnit, self).__init__()
        self.act_f = nn.Tanh()
        self.act_g = nn.Sigmoid()

    def forward(self, x_f, x_g):
        return self.act_f(x_f) * self.act_g(x_g)


class StaticScaledActivation(nn.Module):
    def __init__(self, activation, dim):
        super(StaticScaledActivation, self).__init__()
        self.activation = activation
        self.scales = nn.Parameter(torch.rand(dim, ) * 100, )
        self.rg = nn.Parameter(torch.rand(dim, ), )
        self.dim = dim

    def forward(self, x):
        s, rg = self.scales.to(x.device), self.rg.to(x.device)
        return self.activation(rg * x / s) * s


class ScaledActivation(nn.Module):
    def __init__(self, activation, dim, with_range=True):
        super(ScaledActivation, self).__init__()
        self.activation = activation
        self.scales = nn.Linear(dim, dim)
        self.dim = dim

    def forward(self, x):
        return self.activation(x) / self.activation(self.scales(x))


class ScaledSigmoid(ScaledActivation):
    def __init__(self, dim, with_range=True):
        super(ScaledSigmoid, self).__init__(nn.Sigmoid(), dim, with_range)


class ScaledTanh(ScaledActivation):
    def __init__(self, dim, with_range=True):
        super(ScaledTanh, self).__init__(nn.Tanh(), dim, with_range)


class ScaledAbs(ScaledActivation):
    def __init__(self, dim, with_range=True):
        super(ScaledAbs, self).__init__(Abs(), dim, with_range)