from nnAudio.features import STFT, MelSpectrogram, MFCC
from typing import Optional, Tuple, Dict, Union, Literal

import torch
import torch.nn as nn

from .arm import ARMWithHidden
from .resamplers import LinearResampler, Conv1dResampler
from ..modules.misc import Unsqueeze, Flatten

T = torch.Tensor


class SampleRNNTier(nn.Module):
    """
    Note on the implementation:
    - no skips between between RNNs
    - no embeddings in bottom tier
    - always LSTM (no GRU...)

    """

    def __init__(
            self, *,
            frame_size: Optional[int] = 8,
            hidden_dim: int = 256,
            up_sampling: Optional[int] = None,
            n_rnn: Optional[int] = None,
            q_levels: Optional[int] = 256,
    ):
        super(SampleRNNTier, self).__init__()
        self.frame_size = frame_size
        self.hidden_dim = hidden_dim
        self.up_sampling = up_sampling
        self.n_rnn = n_rnn
        self.q_levels = q_levels

        self.has_input_proj = frame_size is not None
        self.is_bottom = up_sampling is None and n_rnn is None
        self.hidden = None

        assert not (self.is_bottom and not self.has_input_proj), "Bottom Tier's frame_size can not be None"

        if not self.is_bottom:
            if self.has_input_proj:
                self.input_proj = nn.Linear(frame_size, hidden_dim)
            self.rnn = nn.LSTM(hidden_dim, hidden_dim, n_rnn, batch_first=True)
            self.up_sampler = LinearResampler(hidden_dim, t_factor=up_sampling, d_factor=1)
        else:
            self.input_proj = nn.Sequential(
                Flatten(-2),  # -> (batch, n_frames * frame_size)
                Unsqueeze(-1),  # -> (batch, n_frames * frame_size, 1)
                # in_dim == size of embeddings in original impl
                Conv1dResampler(in_dim=1, t_factor=1 / frame_size, d_factor=hidden_dim)
                # -> (batch, n_frames, hidden_dim)
                # TODO: Last Tier is JUST an input module -> try with lstm ?
            )

    def forward(
            self,
            inputs: Tuple[T, Optional[T]]
    ) -> Tuple[T]:
        # x: (batch, n_frames, frame_size)
        # x_upper: (batch, n_frames, hidden_dim)
        x, x_upper = inputs
        Q = self.q_levels
        if Q is not None:
            # otherwise, x is expected to already have a float type
            x = self.linearize(x, Q)
        if self.has_input_proj:
            x = self.input_proj(x)
        if x_upper is not None:
            x += x_upper
        if not self.is_bottom:
            self.hidden = self._reset_hidden(x, self.hidden)
            x, self.hidden = self.rnn(x, self.hidden)
            x = self.up_sampler(x)
            # x: (batch, n_frames * up_sampling, hidden_dim)
            return x,
        else:
            # x: (batch, n_frames, hidden_dim)
            return x,

    @staticmethod
    def linearize(x: T, q_levels: int):
        return ((x.float() / q_levels) - .5) * 4

    def _reset_hidden(self, x: T, hidden: Optional[Tuple[T, T]]) -> Tuple[T, T]:
        if hidden is None or x.size(0) != hidden[0].size(1):
            B = x.size(0)
            h0 = nn.Parameter(torch.randn(self.n_rnn, B, self.hidden_dim).to(x.device))
            c0 = nn.Parameter(torch.randn(self.n_rnn, B, self.hidden_dim).to(x.device))
            return h0, c0
        else:
            return hidden[0].detach(), hidden[1].detach()


"""
Whether top or bottom, srnn input modules must perform:
    
    (B, T)  -->  (B, n_frames, frame_size_i)  -->  (B, n_frames, hidden_dim)
    
Top:
    
    - original does: 
        x = linearize(x)
        x = nn.Linear(frame_size, hidden_dim)(x)
    
    - we could:
    
    1. ----> FFT   ( HAVE TO ALIGN INPUTS! )
        x = FFT(x_unframed)  # (B, T) -> (B, n_frames, fft_dim) 
        x = nn.Linear(fft_dim, hidden_dim)
    
    2. ---> aggregate embeddings  ( for slow features, like segment labels )
        x = nn.EmbeddingBag(class_size, hidden_dim)(x.view(-1, frame_size))
        x = x.view(B, -1, hidden_dim)
    
    3. ---> use conv stride, dilation (~down-sample)
        x: (B, T, 1)
        x: (B, T, D) = Embeddings(x)
        x: (B, n_frames, hidden) = Conv1d(D, hidden, K,
                        stride=frame_size, dilation=frame_size//K)(x)
    
Bottom:
    
    - original does:
        x = nn.Embedding(1, emb_dim)(x)
        x = nn.Conv1d(emb_dim, hidden_dim, frame_size)(x).squeeze()
        
    - we could:
    
    1. ---> add LSTM at the end...
    
    2. ---> bypass embeddings
        x = linearize(x)
        x = nn.Conv1d(1, hidden_dim, frame_size)(x).squeeze()
        
    3. ---> put a small wavenet at the bottom
        x = linearize(x)
        x = WaveNet(x)
    
    
THEN: HOW TO DEAL WITH 2D Inputs?

e.g. multiple envelopes
"""


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
            hop_length=frame_size,  # if dataset does not frame else None
            frame_size=frame_size,
            hidden_dim=hidden_dim,
            )

    for conditioning a top tier with a discrete feature:
        mod = FramedInput(
            class_size=q_levels,
            projection_type="embedding",  # will be 'bagged'
            hop_length=frame_size,
            frame_size=frame_size,
            hidden_dim=hidden_dim,
            )

    for a real valued feature:
        mod = FramedInput(
            class_size=None,
            projection_type="linear",
            hop_length=None,
            frame_size=frame_size,
            hidden_dim=hidden_dim,
            )
    """

    class Config:
        class_size: Optional[int]
        projection_type: Literal["linear", "embedding", "fft"]
        # other params must be set by the network

    def __init__(self, *,
                 class_size: Optional[int],
                 projection_type: Literal["linear", "embedding", "fft"],
                 hop_length: Optional[int],  # input is assumed to be already framed if None
                 frame_size: int,
                 hidden_dim: int,
                 sr: int = 16000,
                 ):
        super(FramedInput, self).__init__()
        self.class_size = class_size
        self.projection_type = projection_type
        self.hop_length = hop_length
        self.frame_size = frame_size
        self.hidden_dim = hidden_dim

        self.real_input = class_size is None
        self.needs_casting = class_size is not None and projection_type == 'linear'
        self.is_freq_transform = projection_type in ("fft", "melspec", "mfcc")
        self.unfold_input = isinstance(hop_length, int) or self.is_freq_transform

        if self.real_input or projection_type == 'linear':
            self.input_proj = nn.Linear(frame_size, hidden_dim)
        elif projection_type == 'embedding':
            assert not self.real_input, "class_size can not be None if projection_type == 'embedding'"
            self.input_proj = nn.EmbeddingBag(self.class_size, hidden_dim)
        elif projection_type == "fft":
            self.hop_length = frame_size if hop_length is None else hop_length
            self.input_proj = STFT(n_fft=frame_size,
                                   freq_bins=hidden_dim,  # fft's output_dim
                                   hop_length=self.hop_length,
                                   window=1.,
                                   freq_scale='linear',
                                   center=False,
                                   pad_mode=None,
                                   output_format="Magnitude",
                                   sr=sr,
                                   trainable=True,
                                   verbose=False)

    def forward(self, x: torch.Tensor):
        if self.needs_casting:
            x = self.linearize(x, self.class_size)
        if self.unfold_input:
            x = x.unfold(dimension=-1, size=self.frame_size, step=self.hop_length)
        if self.is_freq_transform:
            B = x.size(0)
            x = self.input_proj(x.view(-1, self.frame_size))
            return x.squeeze().view(B, -1, self.hidden_dim)
        return self.input_proj(x)

    @staticmethod
    def linearize(x: T, q_levels: int):
        return ((x.float() / q_levels) - .5) * 4


class SampleRNN(ARMWithHidden, nn.Module):
    """
    Top level Model could
    (- reshape single input during training)
    - project and cat/sum/mix multiple inputs
    - collect tiers outputs and cat/sum/mix them before passing them to its output module
    - have multiple inputs/outputs
    .......
    """

    def __init__(
            self, *,
            frame_sizes: Tuple[int, ...] = (16, 8, 8),  # from top to bottom!
            hidden_dim: int = 256,
            n_rnn: int = 2,
            q_levels: int = 256,
    ):
        super(SampleRNN, self).__init__()
        self.frame_sizes = frame_sizes
        self.hidden_dim = hidden_dim
        self.n_rnn = n_rnn
        self.q_levels = q_levels

        tiers = []
        for i, fs in enumerate(self.frame_sizes[:-1]):
            tiers += [SampleRNNTier(
                frame_size=fs,
                hidden_dim=self.hidden_dim,
                up_sampling=fs // (
                    self.frame_sizes[i + 1]
                    if i < len(self.frame_sizes) - 2
                    else 1),
                n_rnn=self.n_rnn,
                q_levels=self.q_levels,
            )]
        tiers += [SampleRNNTier(
            frame_size=self.frame_sizes[-1],
            hidden_dim=self.hidden_dim,
            q_levels=self.q_levels,
            up_sampling=None,
            n_rnn=None
        )]
        self.tiers = nn.ModuleList(tiers)

        # caches for inference
        self.outputs = []
        self.prompt_length = 0

    def forward(self, tiers_input: Tuple[T, ...]):
        prev_output = None
        for tier_input, tier in zip(tiers_input, self.tiers):
            prev_output = tier.forward((tier_input, prev_output))[0]
        return prev_output

    def before_generate(self, prompts: Tuple[torch.Tensor, ...], batch_index: int) -> None:
        self.outputs = [None] * (len(self.frame_sizes) - 1)
        _batch = prompts[0][:, prompts[0].size(1) % self.rf:]

        self.reset_hidden()
        self.prompt_length = len(_batch[0])
        # warm-up
        for t in range(self.rf, self.prompt_length):
            self.generate_step((_batch[:, t - self.rf:t],), t=t)

    def generate_step(
            self,
            inputs: Tuple[torch.Tensor, ...], *,
            t: int = 0,
            **parameters: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, ...]:
        tiers = self.tiers
        outputs = self.outputs
        fs = self.frame_sizes
        # TODO: should be in **parameters + TODO: MLP
        temperature = inputs[1] if len(inputs) > 1 else None
        inputs = inputs[0]
        for i in range(len(tiers) - 1):
            if t % fs[i] == 0:
                inpt = inputs[:, -fs[i]:].unsqueeze(1)
                if i == 0:
                    prev_out = None
                else:
                    prev_out = outputs[i - 1][:, (t // fs[i]) % (fs[i - 1] // fs[i])].unsqueeze(1)
                out, = tiers[i]((inpt, prev_out))
                outputs[i] = out
        if t < self.prior_t:
            return tuple()
        inpt = inputs[:, -fs[-1]:].reshape(-1, 1, fs[-1])
        prev_out = outputs[-1][:, (t % fs[-2]) - fs[-2]].unsqueeze(1)
        out, = tiers[-1]((inpt, prev_out))
        return (out.squeeze(-1) if len(out.size()) > 2 else out),

    def after_generate(self, final_outputs: Tuple[torch.Tensor, ...], batch_index: int) -> None:
        self.outputs = []
        self.reset_hidden()

    def reset_hidden(self) -> None:
        for t in self.tiers:
            t.hidden = None

    @property
    def time_axis(self) -> int:
        return 1

    @property
    def shift(self) -> int:
        return self.frame_sizes[0]

    @property
    def rf(self) -> int:
        return self.frame_sizes[0]

    @property
    def hop_length(self) -> Optional[int]:
        return None

    def output_length(self, n_input_steps: int) -> int:
        return n_input_steps
