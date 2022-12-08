from enum import Enum, auto
from typing import Optional, Literal, Iterable, Tuple
import dataclasses as dtc
import torch
from torch import nn as nn

from .resamplers import Conv1dResampler
from ..modules.misc import Unsqueeze, Flatten
from ..utils import AutoStrEnum

T = torch.Tensor


class ProjectionType(AutoStrEnum):
    linear = auto()
    embedding = auto()
    fir = auto()
    fir_embedding = auto()
    # "fft",
    # "melspec",
    # "mfcc"


class FramedInput(nn.Module):
    @dtc.dataclass
    class Config:
        class_size: Optional[int]
        projection_type: ProjectionType
        # other params must be set by the network

    def __init__(self, *,
                 class_size: Optional[int],
                 projection_type: ProjectionType,
                 hidden_dim: int,
                 frame_size: int,
                 unfold_step: Optional[int],  # input is assumed to be already framed if None
                 ):
        super(FramedInput, self).__init__()
        self.class_size = class_size
        self.projection_type = projection_type
        self.hidden_dim = hidden_dim
        self.frame_size = frame_size
        self.unfold_step = unfold_step

        self.needs_casting = class_size is not None and "embedding" not in projection_type
        self.is_freq_transform = projection_type in ("fft", "melspec", "mfcc")
        self.unfold_input = isinstance(unfold_step, int) or self.is_freq_transform

        if projection_type == 'linear':
            self.input_proj = nn.Linear(frame_size, hidden_dim)

        elif projection_type == 'embedding':
            assert class_size is not None, "class_size can not be None if projection_type == 'embedding'"
            self.input_proj = nn.EmbeddingBag(self.class_size, hidden_dim)

        elif projection_type == 'fir':
            self.input_proj = nn.Sequential(
                Flatten(-2),  # -> (batch, n_frames * frame_size)
                Unsqueeze(-1),  # -> (batch, n_frames * frame_size, 1)
                Conv1dResampler(in_dim=1, t_factor=1 / frame_size, d_factor=hidden_dim)
                # -> (batch, n_frames, hidden_dim)
            )
        elif projection_type == "fir_embedding":  # original SampleRNN bottom tier
            assert class_size is not None, "class_size can not be None if projection_type == 'fir_embedding'"
            self.input_proj = nn.Sequential(
                nn.Embedding(class_size, hidden_dim),
                # -> (batch, n_frames, frame_size, hidden_dim)
                Conv1dResampler(in_dim=hidden_dim, t_factor=1 / frame_size, d_factor=1)
                # -> (batch, n_frames, hidden_dim)
            )
        else:
            raise TypeError(f"projection_type '{projection_type}' not supported")

    def forward(self, x: torch.Tensor):
        if self.needs_casting:
            x = self.linearize(x, self.class_size)
        if self.unfold_input:
            x = x.unfold(dimension=-1, size=self.frame_size, step=self.unfold_step)
        if self.is_freq_transform or self.projection_type == "embedding":
            B = x.size(0)
            x = self.input_proj(x.view(-1, self.frame_size))
            return x.squeeze().view(B, -1, self.hidden_dim)
        return self.input_proj(x)

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
