import abc
from enum import auto
from typing import Optional, Iterable, Tuple
import dataclasses as dtc
import torch
from torch import nn as nn

from .activations import ActivationConfig
from .resamplers import Conv1dResampler
from .targets import OutputWrapper
from ..networks.mlp import MLP
from ..config import Config, private_runtime_field
from ..modules.misc import Unsqueeze, Flatten, Chunk, Unfold, ShapeWrap
from ..utils import AutoStrEnum

__all__ = [
    "LinearIO",
    "ChunkedLinearIO",
    "FramedLinearIO",
    "EmbeddingIO",
    "EmbeddingBagIO",
    "EmbeddingConv1d",
    "FramedConv1dIO",
    "MLPIO",
    "IOModule",
    "ZipMode",
    "ZipReduceVariables"
]

T = torch.Tensor


@dtc.dataclass
class IOModule(Config, abc.ABC):
    activation: Optional[ActivationConfig] = None
    dropout: float = 0.
    dropout1d: float = 0.

    in_dim: Optional[int] = private_runtime_field(None)
    out_dim: Optional[int] = private_runtime_field(None)
    hop_length: Optional[int] = private_runtime_field(None)
    frame_size: Optional[int] = private_runtime_field(None)
    class_size: Optional[int] = private_runtime_field(None)
    sampler: Optional[nn.Module] = private_runtime_field(None)
    with_linearizer: bool = private_runtime_field(False)
    with_unfold: bool = private_runtime_field(False)
    with_n_chunks: Optional[int] = private_runtime_field(None)

    def set(self, **kwargs):
        for k, v in kwargs.items():
            if not hasattr(self, k):
                raise AttributeError(f"attribute '{k}' not found in IOModule")
            else:
                if getattr(self, k) is not None:
                    raise RuntimeError(f"can not set attribute '{k}'. "
                                       f"It has already been set to '{getattr(self, k)}'")
            setattr(self, k, v)
        return self

    def not_none(self, *args):
        msg = ""
        for k in args:
            if getattr(self, k) is None:
                msg += f"- '{k}' can not be None with module_type '{type(self).__qualname__}'\n"
        if msg:
            raise ValueError(msg)

    @abc.abstractmethod
    def module(self) -> nn.Module:
        ...

    def wrap(self, module: nn.Module) -> nn.Module:

        before = []
        if self.with_linearizer:
            before += [Linearizer(self.class_size)]

        if self.with_unfold:
            self.not_none("frame_size", "hop_length")
            before += [Unfold(-1, self.frame_size, self.hop_length)]

        after = []
        if self.activation is not None and self.activation.act != "Identity":
            if self.activation.scaled:
                self.activation.dim = self.out_dim
            after += [self.activation.get()]
        if self.dropout > 0:
            after += [nn.Dropout(self.dropout)]
        if self.dropout1d > 0:
            after += [nn.Dropout1d(self.dropout1d)]
        if self.with_n_chunks is not None:
            after += [Chunk(self.with_n_chunks, dim=-1, sum_outputs=True)]

        if self.sampler is not None:
            return OutputWrapper(
                nn.Sequential(*before, module, *after), self.sampler
            )
        return nn.Sequential(*before, module, *after)


class Linearizer(nn.Module):
    def __init__(self, class_size: int):
        super(Linearizer, self).__init__()
        self.class_size = class_size

    def forward(self, x):
        return ((x.float() / self.class_size) - .5) * 2


@dtc.dataclass
class LinearIO(IOModule):
    bias: bool = True

    def module(self) -> nn.Module:
        self.not_none("in_dim", "out_dim")
        mod = nn.Linear(self.in_dim, self.out_dim, bias=self.bias)
        return self.wrap(mod)


@dtc.dataclass
class FramedLinearIO(IOModule):

    def module(self) -> nn.Module:
        self.not_none("frame_size", "hop_length", "out_dim", "class_size")
        mod = nn.Linear(self.frame_size, self.out_dim)
        self.with_linearizer = True
        self.with_unfold = True
        return self.wrap(mod)


@dtc.dataclass
class ChunkedLinearIO(IOModule):
    bias: bool = True
    n_chunks: int = 1

    def module(self) -> nn.Module:
        self.not_none("in_dim", "out_dim")
        mod = nn.Linear(self.in_dim, self.out_dim * self.n_chunks, bias=self.bias)
        self.with_n_chunks = self.n_chunks
        return self.wrap(mod)


@dtc.dataclass
class EmbeddingIO(IOModule):

    def module(self) -> nn.Module:
        self.not_none("class_size", "out_dim")
        mod = nn.Embedding(self.class_size, self.out_dim)
        return self.wrap(mod)


@dtc.dataclass
class EmbeddingBagIO(IOModule):

    def module(self) -> nn.Module:
        self.not_none("class_size", "frame_size", "hop_length", "out_dim")
        mod = ShapeWrap(
            nn.EmbeddingBag(self.class_size, self.out_dim),
            (-1, self.frame_size), (-1, self.out_dim)
        )
        self.with_unfold = True
        return self.wrap(mod)


@dtc.dataclass
class EmbeddingConv1d(IOModule):

    def module(self) -> nn.Module:
        self.not_none("class_size", "frame_size", "hop_length", "out_dim")
        mod = nn.Sequential(
            nn.Embedding(self.class_size, self.out_dim),
            # -> (batch, n_frames, frame_size, hidden_dim)
            Conv1dResampler(in_dim=self.out_dim, t_factor=1 / self.frame_size, d_factor=1)
            # -> (batch, n_frames, hidden_dim)
        )
        self.with_unfold = True
        return self.wrap(mod)


@dtc.dataclass
class FramedConv1dIO(IOModule):

    def module(self) -> nn.Module:
        self.not_none("frame_size", "out_dim")
        mod = nn.Sequential(
            Flatten(-2),  # -> (batch, n_frames * frame_size)
            Unsqueeze(-1),  # -> (batch, n_frames * frame_size, 1)
            Conv1dResampler(in_dim=1, t_factor=1 / self.frame_size, d_factor=self.out_dim)
            # -> (batch, n_frames, hidden_dim)
        )
        self.with_linearizer = True
        self.with_unfold = True
        return self.wrap(mod)


@dtc.dataclass
class MLPIO(IOModule):
    hidden_dim: int = 128
    n_hidden_layers: int = 1
    activation: ActivationConfig = ActivationConfig("Mish")
    bias: bool = True
    dropout: float = 0.
    dropout1d: float = 0.
    min_temperature: Optional[float] = 1e-4

    def module(self) -> nn.Module:
        self.not_none("in_dim", "out_dim")
        mod = MLP(
            in_dim=self.in_dim, out_dim=self.out_dim, hidden_dim=self.hidden_dim,
            n_hidden_layers=self.n_hidden_layers, activation=self.activation.get(),
            bias=self.bias, dropout=self.dropout, dropout1d=self.dropout1d,
            min_temperature=self.min_temperature
        )
        self.activation = None
        return self.wrap(mod)


class ZipMode(AutoStrEnum):
    sum = auto()
    mean = auto()
    static_mix = auto()


class ZipReduceVariables(nn.Module):

    def __init__(
            self,
            mode: ZipMode,
            modules: Iterable[nn.Module]
    ):
        super(ZipReduceVariables, self).__init__()
        self.heads = nn.ModuleList(modules)
        self.M = len(self.heads)
        if mode == "sum":
            self.weights = torch.ones(self.M, requires_grad=False)
        elif mode == "mean":
            self.weights = torch.ones(self.M, requires_grad=False) / self.M
        elif mode == "static_mix":
            self.weights = nn.Parameter(- torch.rand(self.M))

    def forward(self, inputs: Tuple[torch.Tensor, ...]):
        w = self.weights.to(inputs[0].device)
        if w.requires_grad:
            w = nn.Softmax(dim=0)(w)
        y = self.heads[0](inputs[0]) * w[0]
        for i in range(1, self.M):
            y += self.heads[i](inputs[i]) * w[i]
        return y
