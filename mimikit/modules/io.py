from enum import auto
from typing import Optional, Iterable, Tuple, Literal
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
    "ModuleType",
    "LinearParams",
    "ChunkedLinearParams",
    "MLPParams",
    "IOFactory",
    "ZipMode",
    "ZipReduceVariables"
]

T = torch.Tensor


class ModuleType(AutoStrEnum):
    linear = auto()
    chunked_linear = auto()
    framed_linear = auto()
    embedding = auto()
    embedding_bag = auto()
    framed_conv1d = auto()
    embedding_conv1d = auto()
    conv_transpose1d = auto()
    mlp = auto()


class ModuleParams(Config):
    pass


class Linearizer(nn.Module):
    def __init__(self, class_size: int):
        super(Linearizer, self).__init__()
        self.class_size = class_size

    def forward(self, x):
        return ((x.float() / self.class_size) - .5) * 2


class LinearParams(ModuleParams):
    bias: bool = True


class ChunkedLinearParams(ModuleParams):
    bias: bool = True
    n_heads: int = 1
    sum_heads: bool = True


class MLPParams(ModuleParams):
    hidden_dim: int = 128
    n_hidden_layers: int = 1
    activation: ActivationConfig = ActivationConfig("Mish")
    bias: bool = True
    dropout: float = 0.
    dropout1d: float = 0.
    learn_temperature: bool = True


class LearnableFFT(ModuleParams):
    n_fft: int = 2048
    hop_length: int = 512
    coordinate: Literal["mag", "pol"] = "mag"


class LearnableMelSpec(ModuleParams):
    n_mel: int = 88
    n_fft: int = 2048
    hop_length: int = 512


class IOFactory(Config):
    module_type: ModuleType
    params: Optional[ModuleParams] = None
    activation: Optional[ActivationConfig] = None
    dropout: float = 0.
    dropout1d: float = 0.

    in_dim: Optional[int] = private_runtime_field(None)
    out_dim: Optional[int] = private_runtime_field(None)
    hop_length: Optional[int] = private_runtime_field(None)
    frame_size: Optional[int] = private_runtime_field(None)
    class_size: Optional[int] = private_runtime_field(None)
    # for output distributions:
    sampler: Optional[nn.Module] = private_runtime_field(None)
    n_params: Optional[int] = private_runtime_field(None)
    n_components: Optional[int] = private_runtime_field(None)

    def set(self, **kwargs):
        for k, v in kwargs.items():
            if not hasattr(self, k):
                raise AttributeError(f"attribute '{k}' not found in IOFactory")
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
                msg += f"- '{k}' can not be None with module_type '{self.module_type}'\n"
        if msg:
            raise ValueError(msg)

    def get(self) -> nn.Module:
        in_dim, out_dim, class_size, frame_size, hop_length = \
            self.in_dim, self.out_dim, self.class_size, self.frame_size, self.hop_length

        needs_casting = class_size is not None and "embedding" not in self.module_type
        is_freq_transform = self.module_type in ("fft", "melspec", "mfcc")
        unfold_input = isinstance(hop_length, int) or is_freq_transform

        before = []
        if needs_casting:
            before += [Linearizer(class_size)]
        if unfold_input:
            self.not_none("frame_size", "hop_length")
            before += [Unfold(-1, frame_size, hop_length)]

        if self.module_type == "linear":
            self.not_none("in_dim", "out_dim", "params")
            mod = nn.Linear(in_dim, out_dim, bias=self.params.bias)

        elif self.module_type == 'chunked_linear':
            self.not_none("in_dim", "out_dim", "params")
            p = self.params
            n_heads, bias, sum_heads = p.n_heads, p.bias, p.sum_heads
            mod = nn.Sequential(
                nn.Linear(in_dim, out_dim * n_heads, bias=bias),
                Chunk(n_heads, dim=-1, sum_outputs=sum_heads)
            )

        elif self.module_type == 'framed_linear':
            self.not_none("frame_size", "hop_length", "out_dim")
            mod = nn.Linear(frame_size, out_dim)

        elif self.module_type == "embedding":
            self.not_none("class_size", "out_dim")
            mod = nn.Embedding(class_size, out_dim)

        elif self.module_type == 'embedding_bag':
            self.not_none("class_size", "frame_size", "hop_length", "out_dim")
            mod = ShapeWrap(
                nn.EmbeddingBag(class_size, out_dim),
                (-1, frame_size), (-1, out_dim)
            )

        elif self.module_type == 'framed_conv1d':
            self.not_none("frame_size", "out_dim")
            mod = nn.Sequential(
                Flatten(-2),  # -> (batch, n_frames * frame_size)
                Unsqueeze(-1),  # -> (batch, n_frames * frame_size, 1)
                Conv1dResampler(in_dim=1, t_factor=1 / frame_size, d_factor=out_dim)
                # -> (batch, n_frames, hidden_dim)
            )
        elif self.module_type == "embedding_conv1d":  # original SampleRNN bottom tier
            self.not_none("class_size", "frame_size", "hop_length", "out_dim")
            mod = nn.Sequential(
                nn.Embedding(class_size, out_dim),
                # -> (batch, n_frames, frame_size, hidden_dim)
                Conv1dResampler(in_dim=out_dim, t_factor=1 / frame_size, d_factor=1)
                # -> (batch, n_frames, hidden_dim)
            )
        elif self.module_type == "mlp":
            self.not_none("in_dim", "out_dim", "params")
            params = self.params
            assert isinstance(params, MLPParams)
            mod = MLP(
                in_dim=in_dim, out_dim=out_dim, hidden_dim=params.hidden_dim,
                n_hidden_layers=params.n_hidden_layers, activation=params.activation.get(),
                bias=params.bias, dropout=params.dropout, dropout1d=params.dropout1d,
                learn_temperature=params.learn_temperature
            )
        else:
            raise NotImplementedError(f"module_type '{self.module_type}' not implemented")

        after = []
        if self.activation is not None and self.activation.act != "Identity":
            if self.activation.scaled:
                self.activation.dim = out_dim
            after += [self.activation.get()]
        if self.dropout > 0:
            after += [nn.Dropout(self.dropout)]
        if self.dropout1d > 0:
            after += [nn.Dropout1d(self.dropout1d)]

        if self.sampler is not None:
            return OutputWrapper(
                nn.Sequential(*before, mod, *after), self.sampler
            )
        return nn.Sequential(*before, mod, *after)


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