import torch
import torch.nn as nn
from argparse import Namespace
from dataclasses import dataclass

from ..modules.homs import *

__all__ = [
    "SampleRNNTier",
    "SampleRNNNetwork"
]


class SampleRNNTier(HOM):
    """
    in time-domain :
        - bottom tier has embedding and mlp out
        - all other linearize their inputs
    in the tf-domain :
        - bottom has no embedding (4 embedded samples == same shape as 4 FFT)
        - all others get 1 frame (frame_size == n_fft)
    """

    def __init__(self,
                 frame_size: int,
                 dim: int,
                 up_sampling: int = 1,
                 n_rnn: int = 2,
                 q_levels: int = 256,
                 embedding_dim: int = None,
                 mlp_dim: int = None,
                 in_feat=None,
                 out_feat=None
                 ):
        init_ctx = locals()
        init_ctx.pop("self")
        init_ctx.pop("__class__")
        self.hp = Namespace(**init_ctx)

        is_bottom = up_sampling == 1

        def linearize(qx):
            """ maps input samples (0 <= qx < 256) to floats (-2. <= x < 2.) """
            return ((qx.float() / q_levels) - .5) * 4

        def reset_hidden(x, h):
            if h is None or x.size(0) != h[0].size(1):
                B = x.size(0)
                h0 = torch.zeros(n_rnn, B, dim).to(x.device)
                c0 = torch.zeros(n_rnn, B, dim).to(x.device)
                return h0, c0
            else:
                return tuple(h_.detach() for h_ in h)

        FS, E = frame_size, embedding_dim

        super().__init__(
            "x, upper_out=None, h=None -> x, h",
            (lambda x: x.size()[:2], "x -> B, T"),

            *Maybe(not is_bottom,
                   *Maybe(type(in_feat) in (MuLawSignal, ),
                          (linearize, "x -> x"),),
                   (nn.Linear(FS, dim), "x -> x"),
                   ),
            *Maybe(is_bottom,
                   (in_feat.input_module(E), "x -> x"),
                   (lambda x: x.view(-1, FS, E).transpose(1, 2).contiguous(), "x -> x"),
                   (nn.Conv1d(E, dim, (FS,)), "x -> x"),
                   (lambda x, B, T: x.squeeze().reshape(B, T, -1), "x, B, T -> x"),
                   ),

            (lambda x, upper: x if upper is None else (x + upper), "x, upper_out -> x"),

            *Maybe(not is_bottom,
                   (reset_hidden, "x, h -> h"),
                   (nn.LSTM(dim, dim, n_rnn, batch_first=True), "x, h -> x, h"),
                   # upsample T
                   (nn.Linear(dim, dim * up_sampling), "x -> x"),
                   (lambda x, B, T: x.reshape(B, T * up_sampling, dim), "x, B, T -> x")
                   ),

            *Maybe(is_bottom,
                   (out_feat.output_module(dim),
                    "x -> x")
                   )
        )


@dataclass(init=True, repr=False, eq=False, frozen=False, unsafe_hash=True)
class SampleRNNNetwork(nn.Module):
    frame_sizes: tuple = (16, 8, 8)  # from top to bottom!
    dim: int = 512
    n_rnn: int = 2
    q_levels: int = 256
    embedding_dim: int = 256
    mlp_dim: int = 512
    input_features: tuple = ()
    output_feature: object = None

    def __post_init__(self):
        nn.Module.__init__(self)
        n_tiers = len(self.frame_sizes)
        tiers = []
        if n_tiers > 2:
            for i, fs in enumerate(self.frame_sizes[:-2]):
                tiers += [SampleRNNTier(fs, self.dim,
                                        up_sampling=fs // self.frame_sizes[i + 1],
                                        n_rnn=self.n_rnn,
                                        q_levels=self.q_levels,
                                        embedding_dim=None,
                                        mlp_dim=None,
                                        in_feat=self.input_features[i])]
        # before last tier
        tiers += [SampleRNNTier(self.frame_sizes[-2], self.dim,
                                up_sampling=self.frame_sizes[-1],
                                n_rnn=self.n_rnn,
                                q_levels=self.q_levels,
                                embedding_dim=None,
                                mlp_dim=None,
                                in_feat=self.input_features[-2])]
        # bottom tier
        tiers += [SampleRNNTier(self.frame_sizes[-1], self.dim,
                                up_sampling=1,
                                n_rnn=self.n_rnn,
                                q_levels=self.q_levels,
                                embedding_dim=self.embedding_dim,
                                mlp_dim=self.mlp_dim,
                                in_feat=self.input_features[-2],
                                out_feat=self.output_feature)]

        self.tiers = nn.ModuleList(tiers)
        self.hidden = [None] * n_tiers

    def forward(self, tiers_inputs):
        prev_out = None
        for i, (tier, inpt) in enumerate(zip(self.tiers, tiers_inputs)):
            prev_out, self.hidden[i] = tier(inpt, prev_out, self.hidden[i])
        return prev_out

    def reset_h0(self):
        self.hidden = [None] * len(self.frame_sizes)

    def generate_(self, prompt, n_steps, temperature=1.):
        output = self.prepare_prompt(prompt, n_steps, at_least_nd=2)
        # trim to start with a whole number of top frames
        output = output[:, prompt.size(1) % self.frame_sizes[0]:]
        prior_t = prompt.size(1) - (prompt.size(1) % self.frame_sizes[0])

        # init variables
        fs = [*self.frame_sizes]
        outputs = [None] * (len(fs) - 1)
        # hidden are reset if prompt.size(0) != self.hidden.size(0)
        hiddens = self.hidden
        tiers = self.tiers

        for t in self.generate_tqdm(range(fs[0], n_steps + prior_t)):
            for i in range(len(tiers) - 1):
                if t % fs[i] == 0:
                    inpt = output[:, t - fs[i]:t].unsqueeze(1)

                    if i == 0:
                        prev_out = None
                    else:
                        prev_out = outputs[i - 1][:, (t // fs[i]) % (fs[i - 1] // fs[i])].unsqueeze(1)

                    out, h = tiers[i](inpt, prev_out, hiddens[i])
                    hiddens[i] = h
                    outputs[i] = out
            if t < prior_t:  # only used for warming-up
                continue
            prev_out = outputs[-1]
            inpt = output[:, t - fs[-1]:t].reshape(-1, 1, fs[-1])

            out, _ = tiers[-1](inpt, prev_out[:, (t % fs[-1]) - fs[-1]].unsqueeze(1))
            if temperature is None:
                pred = (nn.Softmax(dim=-1)(out.squeeze(1))).argmax(dim=-1)
            else:
                # great place to implement dynamic cooling/heating !
                pred = torch.multinomial(nn.Softmax(dim=-1)(out.squeeze(1) / temperature), 1)
            output.data[:, t] = pred.squeeze()

        return output
