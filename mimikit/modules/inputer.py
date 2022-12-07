from typing import Optional, Literal

import torch
from torch import nn as nn

from .resamplers import Conv1dResampler
from ..modules.misc import Unsqueeze, Flatten

T = torch.Tensor
ProjectionType = Literal[
    "linear",
    "embedding",
    "fir",
    "fir_embedding",
    "fft",
    "melspec",
    "mfcc"
]


class FramedInput(nn.Module):
    """
    input module for converting (possibly not yet) framed time series
    to (pseudo) frequency domain representations,
        e.g.
         (batch, n_frames, frame_size) -> (batch, n_frames, hidden_dim)

    for standard top tier SampleRNN input module:
        mod = FramedInput(
            class_size=q_levels,
            projection_type="linear",
            unfold_step=frame_size,  # if dataset does not frame else None
            frame_size=frame_size,
            hidden_dim=hidden_dim,
            )

    for conditioning a top tier with a discrete feature:
        mod = FramedInput(
            class_size=q_levels,
            projection_type="embedding",  # will be 'bagged'
            unfold_step=frame_size,
            frame_size=frame_size,
            hidden_dim=hidden_dim,
            )

    for a real valued feature:
        mod = FramedInput(
            class_size=None,
            projection_type="linear",
            unfold_step=None,
            frame_size=frame_size,
            hidden_dim=hidden_dim,
            )
    """

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

        self.real_input = class_size is None
        self.needs_casting = class_size is not None and projection_type == 'linear'
        self.is_freq_transform = projection_type in ("fft", "melspec", "mfcc")
        self.unfold_input = isinstance(unfold_step, int) or self.is_freq_transform

        if self.real_input or projection_type == 'linear':
            self.input_proj = nn.Linear(frame_size, hidden_dim)

        elif projection_type == 'embedding':
            assert not self.real_input, "class_size can not be None if projection_type == 'embedding'"
            self.input_proj = nn.EmbeddingBag(self.class_size, hidden_dim)

        elif projection_type == 'fir':
            self.input_proj = nn.Sequential(
                Flatten(-2),  # -> (batch, n_frames * frame_size)
                Unsqueeze(-1),  # -> (batch, n_frames * frame_size, 1)
                Conv1dResampler(in_dim=1, t_factor=1 / frame_size, d_factor=hidden_dim)
                # -> (batch, n_frames, hidden_dim)
            )
        elif projection_type == "fir_embedding":  # original SampleRNN bottom tier
            assert not self.real_input, "class_size can not be None if projection_type == 'fir_embedding'"
            self.input_proj = nn.Sequential(
                nn.Embedding(class_size, hidden_dim),
                # -> (batch, n_frames, frame_size, hidden_dim)
                Conv1dResampler(in_dim=hidden_dim, t_factor=1 / frame_size, d_factor=hidden_dim)
                # -> (batch, n_frames, hidden_dim)
            )
        elif self.is_freq_transform:
            pass
        #     self.hop_length = frame_size if hop_length is None else hop_length
        #     self.input_proj = STFT(n_fft=frame_size,
        #                            freq_bins=hidden_dim,  # fft's output_dim
        #                            hop_length=self.hop_length,
        #                            window=1.,
        #                            freq_scale='linear',
        #                            center=False,
        #                            pad_mode=None,
        #                            output_format="Magnitude",
        #                            sr=sr,
        #                            trainable=True,
        #                            verbose=False)
        else:
            raise TypeError(f"projection_type '{projection_type}' not supported")

    def forward(self, x: torch.Tensor):
        if self.needs_casting:
            x = self.linearize(x, self.class_size)
        if self.unfold_input:
            x = x.unfold(dimension=-1, size=self.frame_size, step=self.unfold_step)
        if self.is_freq_transform:
            B = x.size(0)
            x = self.input_proj(x.view(-1, self.frame_size))
            return x.squeeze().view(B, -1, self.hidden_dim)
        return self.input_proj(x)

    @staticmethod
    def linearize(x: T, q_levels: int):
        return ((x.float() / q_levels) - .5) * 4
