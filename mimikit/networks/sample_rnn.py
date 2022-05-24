import dataclasses
from typing import Optional

import torch
import torch.nn as nn
from dataclasses import dataclass

from pytorch_lightning.utilities import AttributeDict

from .single_class_mlp import SingleClassMLP
from ..modules.homs import *
from .resamplers import *

__all__ = [
    "SampleRNNTier",
    "SampleRNNTopTier",
    "SampleRNNBottomTier",
    "TierNetwork",
    "SampleRNN",
    "MixedRNN",
]


class TierNetwork(HOM):
    device = property(lambda self: next(self.parameters()).device)

    def __init__(self, *tiers):
        in_ = ', '.join([f'x{i}' for i in range(len(tiers))])
        hidden = ', '.join([f'h{i}' for i in range(len(tiers))])
        _, kwargs = zip(*[get_input_signature(tier.forward) for tier in tiers])
        kwargs = set([k for kw in kwargs for k in kw.split(", ") if k != 'h=None'])
        kwargs = ', '.join(kwargs)
        super().__init__(
            f"{in_}, {kwargs} -> z",
            (lambda self: self.hidden, f"self -> {hidden}"),
            *[
                (tier, ', '.join(tier.s.in_).replace("x", f"x{i}").replace("h", f"h{i}") + f" -> z, h{i}")
                for i, tier in enumerate(tiers)
            ],
            (lambda *args: setattr(args[0], 'hidden', args[1:]), f"self, {hidden} -> $")
        )
        self.tiers = tiers
        self.frame_sizes = tuple(tier.hp.frame_size for tier in tiers)
        self.hidden = (None,) * len(tiers)
        self.shift = self.rf = self.frame_sizes[0]
        self.outputs = []
        self.rf = self.shift = self.frame_sizes[0]
        self.output_length = lambda n: n

    def reset_hidden(self):
        self.hidden = (None,) * len(self.frame_sizes)

    def before_generate(self, loop, batch, batch_idx):
        self.outputs = [None] * (len(self.frame_sizes) - 1)
        assert batch[0].size(1) % self.shift == 0, f'batch_length must be divisible by {self.shift} ' \
                                                   f'(got {batch[0].size(1)})'
        self.reset_hidden()
        self.hidden = list(self.hidden)
        self.prior_t = len(batch[0][0])
        # warm-up
        for t in range(self.shift, self.prior_t):
            self.generate_step(t, (batch[0][:, t - self.shift:t],), {})
        return {}

    def generate_step(self, t, inputs, ctx):
        tiers = self.tiers
        hiddens = self.hidden
        outputs = self.outputs
        fs = self.frame_sizes
        # only one tensor in the `inputs` tuple
        temperature = inputs[1] if len(inputs) > 1 else None
        inputs = inputs[0]
        for i in range(len(tiers) - 1):
            if t % fs[i] == 0:
                inpt = inputs[:, -fs[i]:].unsqueeze(1)
                if i == 0:
                    prev_out = None
                else:
                    prev_out = outputs[i - 1][:, (t // fs[i]) % (fs[i - 1] // fs[i])].unsqueeze(1)
                out, h = tiers[i](inpt, prev_out, hiddens[i])
                hiddens[i] = h
                outputs[i] = out
        if t < self.prior_t:
            return
        prev_out = outputs[-1]
        inpt = inputs[:, -fs[-1]:].reshape(-1, 1, fs[-1])
        out, _ = tiers[-1](inpt, prev_out[:, (t % fs[-2]) - fs[-2]].unsqueeze(1), temperature=temperature)
        return out.squeeze(-1) if len(out.size()) > 2 else out

    def after_generate(self, *args, **kwargs):
        self.outputs = []
        self.reset_hidden()


class SampleRNNTier(HOM):
    resampler_cls = None
    hp = AttributeDict()

    device = property(lambda self: next(self.parameters()).device)

    def linearize(self, qx):
        """ maps input samples (0 <= qx < 256) to floats (-2. <= x < 2.) """
        return ((qx.float() / self.hp.q_levels) - .5) * 4

    @staticmethod
    def add_upper_tier(x, upper):
        return x if upper is None else (x + upper)

    def reset_hidden(self, x, h):
        if h is None or x.size(0) != h[0].size(1):
            B = x.size(0)
            h0 = nn.Parameter(torch.randn(self.hp.n_rnn, B, self.hp.dim).to(x.device))
            c0 = nn.Parameter(torch.randn(self.hp.n_rnn, B, self.hp.dim).to(x.device))
            return h0, c0
        else:
            return tuple(h_.detach() for h_ in h)


class SampleRNNTopTier(SampleRNNTier):

    def __init__(self,
                 frame_size: int,
                 dim: int,
                 up_sampling: int = 1,
                 n_rnn: int = 2,
                 q_levels: int = 256,
                 linearize=True,
                 ):
        init_ctx = locals()
        init_ctx.pop("self")
        init_ctx.pop("__class__")
        self.hp = AttributeDict(init_ctx)
        self.frame_size = FS = frame_size
        super().__init__(
            "x, z=None, h=None -> x, h",
            *Maybe(linearize,
                   (self.linearize, "x -> x")),
            (nn.Linear(FS, dim), "x -> x"),
            (self.add_upper_tier, "x, z -> x"),
            (self.reset_hidden, "x, h -> h"),
            (nn.LSTM(dim, dim, n_rnn, batch_first=True), "x, h -> x, h"),
            (LinearResampler(dim, t_factor=up_sampling, d_factor=1), "x -> x"),
        )


class SampleRNNBottomTier(SampleRNNTier):

    def __init__(self,
                 frame_size: int,
                 top_dim: int,
                 zin_dim: int,
                 zout_dim: int,
                 io_dim: Optional[int] = None,
                 input_type: str = "lin",
                 learn_temperature: bool = True,
                 n_hidden_layers: int = 0,
                 ):
        init_ctx = locals()
        init_ctx.pop("self")
        init_ctx.pop("__class__")
        self.hp = AttributeDict(init_ctx)
        self.hp.q_levels = io_dim
        self.frame_size = frame_size
        super().__init__(
            f"x, z=None, h=None, temperature=None -> x, h",
            *Maybe(io_dim is not None,
                   *Maybe(input_type == "emb",
                          (nn.Embedding(io_dim, zin_dim), 'x -> x')),
                   *Maybe(input_type == "lin",
                          (lambda x: self.linearize(x.unsqueeze(-1)), "x -> x"))
                   ),
            (Conv1dResampler(zin_dim, 1 / frame_size, top_dim / zin_dim), 'x -> x'),
            (self.add_upper_tier, "x, z -> x"),
            *Maybe(io_dim is not None,
                   (SingleClassMLP(top_dim, zout_dim, io_dim,
                                   learn_temperature=learn_temperature,
                                   n_hidden_layers=n_hidden_layers), 'x, temperature -> x'))
        )


@dataclass(init=True, repr=False, eq=False, frozen=False, unsafe_hash=True)
class SampleRNN(TierNetwork):
    frame_sizes: tuple = (16, 8, 8)  # from top to bottom!
    dim: int = 512
    n_rnn: int = 2
    q_levels: int = 256
    embedding_dim: int = 256
    mlp_dim: int = 512
    input_type: str = "lin"  # bottom tier input type ["emb" | "lin"]
    learn_temperature: bool = True
    n_hidden_layers: int = 0

    def __post_init__(self):
        self.hp = AttributeDict(dataclasses.asdict(self))
        tiers = []
        for i, fs in enumerate(self.frame_sizes[:-1]):
            tiers += [SampleRNNTopTier(fs, self.dim,
                                       up_sampling=fs // (
                                           self.frame_sizes[i + 1]
                                           if i < len(self.frame_sizes) - 2
                                           else 1),
                                       n_rnn=self.n_rnn,
                                       q_levels=self.q_levels,
                                       )]
        tiers += [SampleRNNBottomTier(self.frame_sizes[-1], self.dim,
                                      zin_dim=self.embedding_dim,
                                      zout_dim=self.mlp_dim,
                                      io_dim=self.q_levels,
                                      input_type=self.input_type,
                                      learn_temperature=self.learn_temperature,
                                      n_hidden_layers=self.n_hidden_layers,
        )]
        TierNetwork.__init__(self, *tiers)


@dataclass(init=True, repr=False, eq=False, frozen=False, unsafe_hash=True)
class MixedRNN(TierNetwork):
    """top tiers get mag-fft inputs, bottom tier get embedded samples"""

    frame_sizes: tuple = (128, 64, 8)  # from top to bottom!
    dim: int = 512
    n_rnn: int = 2
    q_levels: int = 256
    embedding_dim: int = 256
    mlp_dim: int = 512

    def __post_init__(self):
        self.hp = AttributeDict(dataclasses.asdict(self))
        tiers = []
        for i, fs in enumerate(self.frame_sizes[:-1]):
            tiers += [SampleRNNTopTier(fs, self.dim,
                                       up_sampling=fs // (self.frame_sizes[i + 1]
                                                          if fs != self.frame_sizes[i + 1] else 1),
                                       n_rnn=self.n_rnn,
                                       linearize=False,
                                       )]
        tiers += [SampleRNNBottomTier(self.frame_sizes[-1], self.dim,
                                      zin_dim=self.embedding_dim,
                                      zout_dim=self.mlp_dim,
                                      io_dim=self.q_levels,
                                      )]
        TierNetwork.__init__(self, *tiers)
