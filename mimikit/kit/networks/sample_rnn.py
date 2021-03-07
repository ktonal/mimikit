import torch
import torch.nn as nn
from typing import Optional
from dataclasses import dataclass


@dataclass(init=True, repr=False, eq=False, frozen=False, unsafe_hash=True)
class SampleRNNTier(nn.Module):
    tier_index: int
    frame_size: int
    dim: int
    up_sampling: int = 1
    n_rnn: int = 2
    q_levels: int = 256
    embedding_dim: Optional[int] = None
    mlp_dim: Optional[int] = None

    is_bottom = property(lambda self: self.up_sampling == 1)

    def linearize(self, q_samples):
        """ maps input samples (0 <= qx < 256) to floats (-.5 <= x < .5) """
        return (q_samples.float() / self.q_levels) - .5

    def embeddings_(self):
        if self.embedding_dim is not None:
            return nn.Embedding(self.q_levels, self.embedding_dim)
        return None

    def input_proj_(self):

        if not self.is_bottom:  # top & middle tiers
            return nn.Linear(self.frame_size, self.dim, bias=False)

        else:  # bottom tier
            class BottomProjector(nn.Module):
                def __init__(self, emb_dim, out_dim, frame_size):
                    super(BottomProjector, self).__init__()
                    self.cnv = nn.Conv1d(emb_dim, out_dim, kernel_size=frame_size, bias=False)

                def forward(self, hx):
                    """ hx : (B x T x FS x E) """
                    B, T, FS, E = hx.size()
                    hx = self.cnv(hx.view(B * T, FS, E).transpose(1, 2).contiguous())
                    # now hx : (B*T, DIM, 1)
                    return hx.squeeze().reshape(B, T, -1)

            return BottomProjector(self.embedding_dim, self.dim, self.frame_size)

    def rnn_(self):
        # no rnn for bottom tier
        if self.is_bottom:
            return None

        return nn.LSTM(self.dim, self.dim, self.n_rnn, batch_first=True)

    def up_sampling_net_(self):
        # no up sampling for bottom tier
        if self.is_bottom:
            return None

        class TimeUpscalerLinear(nn.Module):

            def __init__(self, in_dim, out_dim, up_sampling, **kwargs):
                super(TimeUpscalerLinear, self).__init__()
                self.up_sampling = up_sampling
                self.out_dim = out_dim
                self.fc = nn.Linear(in_dim, out_dim * up_sampling, **kwargs)

            def forward(self, x):
                B, T, _ = x.size()
                return self.fc(x).reshape(B, T * self.up_sampling, self.out_dim)

        return TimeUpscalerLinear(self.dim, self.dim, self.up_sampling)

    def mlp_(self):
        if not self.is_bottom:
            return None
        return nn.Sequential(
            nn.Linear(self.dim, self.mlp_dim), nn.ReLU(),
            nn.Linear(self.mlp_dim, self.mlp_dim), nn.ReLU(),
            nn.Linear(self.mlp_dim, self.q_levels),
        )

    def __post_init__(self):
        nn.Module.__init__(self)
        self.embeddings = self.embeddings_()
        self.inpt_proj = self.input_proj_()
        self.rnn = self.rnn_()
        self.up_net = self.up_sampling_net_()
        self.mlp = self.mlp_()

    def forward(self, input_samples, prev_tier_output=None, hidden=None):

        if self.embeddings is None:
            x = self.linearize(input_samples)
        else:
            x = self.embeddings(input_samples)

        if self.inpt_proj is not None:
            p = self.inpt_proj(x)
            if prev_tier_output is not None:
                x = p + prev_tier_output
            else:
                x = p

        if self.rnn is not None:
            if hidden is None or x.size(0) != hidden[0].size(1):
                # I think this is important that both hidden be initialized with tiny random values...
                # zeros would kill gradients wrt the hidden (which we do not want!)
                # and large random values would make little sense...
                # My instinct tells me that it is to relate to the range of the linearization of the input samples, 
                # but it's just intuition...
                h0 = (torch.randn(self.n_rnn, x.size(0), self.dim) * .05).to(x)
                c0 = (torch.randn(self.n_rnn, x.size(0), self.dim) * .05).to(x)
                hidden = (h0, c0)
            else:
                # TRUNCATED back propagation through time == detach()!
                hidden = tuple(h.detach() for h in hidden)

            x, hidden = self.rnn(x, hidden)

        if self.up_net is not None:
            x = self.up_net(x)

        if self.mlp is not None:
            x = self.mlp(x)

        return x, hidden


@dataclass(init=True, repr=False, eq=False, frozen=False, unsafe_hash=True)
class SampleRNNNetwork(nn.Module):

    frame_sizes: tuple  # from top to bottom!
    dim: int = 512
    n_rnn: int = 2
    q_levels: int = 256
    embedding_dim: int = 256
    mlp_dim: int = 512

    def __post_init__(self):
        nn.Module.__init__(self)
        n_tiers = len(self.frame_sizes)
        tiers = []
        if n_tiers > 2:
            for i, fs in enumerate(self.frame_sizes[:-2]):
                tiers += [SampleRNNTier(i, fs, self.dim,
                                        up_sampling=fs // self.frame_sizes[i + 1],
                                        n_rnn=self.n_rnn,
                                        q_levels=self.q_levels,
                                        embedding_dim=None,
                                        mlp_dim=None)]
        # before last tier
        tiers += [SampleRNNTier(n_tiers - 2, self.frame_sizes[-2], self.dim,
                                up_sampling=self.frame_sizes[-1],
                                n_rnn=self.n_rnn,
                                q_levels=self.q_levels,
                                embedding_dim=None,
                                mlp_dim=None)]
        # bottom tier
        tiers += [SampleRNNTier(n_tiers - 1, self.frame_sizes[-1], self.dim,
                                up_sampling=1,
                                n_rnn=self.n_rnn,
                                q_levels=self.q_levels,
                                embedding_dim=self.embedding_dim,
                                mlp_dim=self.mlp_dim)]

        self.tiers = nn.ModuleList(tiers)
        self.hidden = [None] * n_tiers

    def forward(self, tiers_inputs):
        prev_out = None
        for i, (tier, inpt) in enumerate(zip(self.tiers, tiers_inputs)):
            prev_out, self.hidden[i] = tier(inpt, prev_out, self.hidden[i])
        return prev_out

    def reset_h0(self):
        self.hidden = [None] * len(self.frame_sizes)
