from typing import Optional, Tuple, Dict, Literal, Union

import torch
import torch.nn as nn

from .arm import ARMWithHidden
from ..modules.resamplers import LinearResampler

T = torch.Tensor


class SampleRNNTier(nn.Module):

    def __init__(
            self, *,
            hidden_dim: int = 256,
            rnn_class: Literal["lstm", "rnn", "gru", "none"] = "lstm",
            n_rnn: int = 1,
            rnn_dropout: float = 0.,
            up_sampling: Optional[int] = None,
    ):
        super(SampleRNNTier, self).__init__()
        self.hidden_dim = hidden_dim
        self.rnn_class = rnn_class
        self.up_sampling = up_sampling
        self.n_rnn = n_rnn

        self.hidden = None
        self.has_rnn = n_rnn != "none"
        self.has_up_sampling = up_sampling is not None
        if self.has_rnn:
            module = getattr(nn, rnn_class.upper())
            self.rnn = module(hidden_dim, hidden_dim, num_layers=n_rnn,
                              batch_first=True, dropout=rnn_dropout)
        if self.has_up_sampling:
            self.up_sampler = LinearResampler(hidden_dim, t_factor=up_sampling, d_factor=1)

    def forward(
            self,
            inputs: Tuple[T, Optional[T]]
    ) -> Tuple[T]:
        """x: (batch, n_frames, hidden_dim) ; x_upper: (batch, n_frames, hidden_dim)"""
        x, x_upper = inputs
        if x_upper is not None:
            x += x_upper
        if self.has_rnn:
            self.hidden = self._reset_hidden(x, self.hidden)
            x, self.hidden = self.rnn(x, self.hidden)
        if self.has_up_sampling:
            x = self.up_sampler(x)
            # x: (batch, n_frames * up_sampling, hidden_dim)
        return x,

    def _reset_hidden(self, x: T, hidden: Optional[Union[Tuple[T, T]], T]) -> Union[Tuple[T, T], T]:
        if self.rnn_class == "lstm":
            if hidden is None or x.size(0) != hidden[0].size(1):
                B = x.size(0)
                h0 = nn.Parameter(torch.randn(self.n_rnn, B, self.hidden_dim).to(x.device))
                c0 = nn.Parameter(torch.randn(self.n_rnn, B, self.hidden_dim).to(x.device))
                return h0, c0
            else:
                return hidden[0].detach(), hidden[1].detach()
        else:
            if hidden is None or x.size(0) != hidden.size(1):
                B = x.size(0)
                h0 = nn.Parameter(torch.randn(self.n_rnn, B, self.hidden_dim).to(x.device))
                return h0
            else:
                return hidden.detach()


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
                hidden_dim=self.hidden_dim,
                up_sampling=fs // (
                    self.frame_sizes[i + 1]
                    if i < len(self.frame_sizes) - 2
                    else 1),
                n_rnn=self.n_rnn,
            )]
        tiers += [SampleRNNTier(
            hidden_dim=self.hidden_dim,
            up_sampling=None,
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
