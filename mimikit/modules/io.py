from enum import auto
from typing import Optional, Iterable, Tuple, Literal
import dataclasses as dtc
import torch
from torch import nn as nn

from .activations import ActivationConfig
from .resamplers import Conv1dResampler
from ..config import Config
from ..modules.misc import Unsqueeze, Flatten, Chunk
from ..utils import AutoStrEnum


__all__ = [
    "ModuleType",
    "EmbeddingParams",
    "FramedLinearParams",
    "ChunkedLinearParams",
    "ModuleFactory",
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


class LinearParams(ModuleParams):
    in_dim: int = dtc.field(init=False, default=0)
    bias: bool = True


class ChunkedLinearParams(ModuleParams):
    in_dim: int = dtc.field(init=False, default=0)
    bias: bool = True
    n_heads: int = 1
    sum_heads: bool = dtc.field(init=False, default=True)


class EmbeddingParams(ModuleParams):
    class_size: int = dtc.field(init=False, default=0)
    frame_size: Optional[int] = dtc.field(init=False, default=None)


class FramedLinearParams(ModuleParams):
    class_size: int = dtc.field(init=False, default=0)
    frame_size: Optional[int] = dtc.field(init=False, default=None)


class MLPParams(ModuleParams):
    in_dim: int = dtc.field(init=False, default=0)
    hidden_dim: int = 1
    n_hidden_layers: int = 1


class LearnableFFT(ModuleParams):
    n_fft: int = 2048
    hop_length: int = 512
    coordinate: Literal["mag", "pol"] = "mag"


class LearnableMelSpec(ModuleParams):
    n_mel: int = 88
    n_fft: int = 2048
    hop_length: int = 512


class ModuleFactory(nn.Module):
    # TODO: change to callable with __call__(in_dim, hidden_dim, out_dim)
    # or __call__(feature, network, ...)
    class Config(Config):
        module_type: ModuleType
        module_params: Optional[ModuleParams]
        hidden_dim: int = 0
        unfold_step: Optional[int] = None
        activation: ActivationConfig = ActivationConfig("Identity")
        dropout: float = 0.
        dropout1d: float = 0.

        def set_hidden_dim(self, value: int):
            self.hidden_dim = value
            return self

        def set_frame(self, frame_size: int, hop_length: Optional[int]):
            self.unfold_step = hop_length
            if hasattr(self.module_params, "frame_size"):
                self.module_params.frame_size = frame_size
            return self

        def module(self):
            if hasattr(self.module_params, "in_dim"):
                assert self.module_params.in_dim > 0
            if hasattr(self.module_params, "class_size"):
                assert self.module_params.class_size > 0
            assert self.hidden_dim > 0
            return ModuleFactory(module_type=self.module_type, module_params=self.module_params,
                                 hidden_dim=self.hidden_dim, unfold_step=self.unfold_step,
                                 activation=self.activation,
                                 dropout=self.dropout, dropout1d=self.dropout1d)

    def __init__(self, *,
                 module_type: ModuleType,
                 module_params: ModuleParams,
                 hidden_dim: int,
                 unfold_step: Optional[int],  # input is framed if not None
                 activation: ActivationConfig = ActivationConfig("Identity"),
                 dropout: float = 0.,
                 dropout1d: float = 0.,
                 ):
        super(ModuleFactory, self).__init__()
        class_size = getattr(module_params, "class_size", None)
        frame_size = getattr(module_params, "frame_size", None)
        in_dim = getattr(module_params, "in_dim", None)
        self.class_size = class_size
        self.module_type = module_type
        self.hidden_dim = hidden_dim
        self.frame_size = frame_size
        self.unfold_step = unfold_step

        self.needs_casting = class_size is not None and "embedding" not in module_type
        self.is_freq_transform = module_type in ("fft", "melspec", "mfcc")
        self.unfold_input = isinstance(unfold_step, int) or self.is_freq_transform

        if module_type == "linear":
            assert isinstance(module_params, LinearParams)
            self.input_proj = nn.Linear(in_dim, hidden_dim, bias=module_params.bias)

        elif module_type == 'chunked_linear':
            assert isinstance(module_params, ChunkedLinearParams)
            self.input_proj = nn.Sequential(
                nn.Linear(in_dim, hidden_dim * module_params.n_heads, bias=module_params.bias),
                Chunk(module_params.n_heads, dim=-1, sum_outputs=module_params.sum_heads)
            )

        elif module_type == 'framed_linear':
            self.input_proj = nn.Linear(frame_size, hidden_dim)

        elif module_type == "embedding":
            assert class_size is not None, "class_size can not be None if projection_type == 'embedding'"
            self.input_proj = nn.Embedding(class_size, hidden_dim)

        elif module_type == 'embedding_bag':
            assert class_size is not None, "class_size can not be None if projection_type == 'embedding_bag'"
            self.input_proj = nn.EmbeddingBag(class_size, hidden_dim)

        elif module_type == 'framed_conv1d':
            self.input_proj = nn.Sequential(
                Flatten(-2),  # -> (batch, n_frames * frame_size)
                Unsqueeze(-1),  # -> (batch, n_frames * frame_size, 1)
                Conv1dResampler(in_dim=1, t_factor=1 / frame_size, d_factor=hidden_dim)
                # -> (batch, n_frames, hidden_dim)
            )

        elif module_type == "embedding_conv1d":  # original SampleRNN bottom tier
            assert class_size is not None, "class_size can not be None if projection_type == 'embedding_conv1d'"
            self.input_proj = nn.Sequential(
                nn.Embedding(class_size, hidden_dim),
                # -> (batch, n_frames, frame_size, hidden_dim)
                Conv1dResampler(in_dim=hidden_dim, t_factor=1 / frame_size, d_factor=1)
                # -> (batch, n_frames, hidden_dim)
            )
        else:
            raise NotImplementedError(f"module_type '{module_type}' not implemented")

        act = activation.object() if activation.act != "Identity" else None
        dropout = nn.Dropout(dropout) if dropout > 0 else None
        dropout1d = nn.Dropout1d(dropout1d) if dropout1d > 0. else None
        if act is not None or dropout is not None or dropout1d is not None:
            self.has_act = True
            self.act = nn.Sequential(
                *(m for m in [act, dropout, dropout1d] if m is not None)
            )
        else:
            self.has_act = False

    def forward(self, x: torch.Tensor):
        if self.needs_casting:
            x = self.linearize(x, self.class_size)
        if self.unfold_input:
            x = x.unfold(dimension=-1, size=self.frame_size, step=self.unfold_step)
        if self.is_freq_transform or self.module_type == "embedding_bag":
            B = x.size(0)
            x = self.input_proj(x.view(-1, self.frame_size))
            x = x.squeeze().view(B, -1, self.hidden_dim)
        else:
            x = self.input_proj(x)
        if self.has_act:
            return self.act(x)
        return x

    @staticmethod
    def linearize(x: T, q_levels: int):
        return ((x.float() / q_levels) - .5) * 4


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
            self.weights = torch.zeros(self.M, requires_grad=False)
        elif mode == "mean":
            self.weights = (torch.ones(self.M, requires_grad=False) / self.M).log_()
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
