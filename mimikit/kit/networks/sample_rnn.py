import torch
import torch.nn as nn
from numpy import cumprod

from ..modules.homs import propped
from ..modules import homs as H, ops as Ops


class TimeUpscalerLinear(nn.Module):

    def __init__(self, in_dim, out_dim, upscaling, **kwargs):
        super(TimeUpscalerLinear, self).__init__()
        self.upscaling = upscaling
        self.out_dim = out_dim
        self.fc = nn.Linear(in_dim, out_dim * upscaling, **kwargs)

    def forward(self, x):
        B, T, _ = x.size()
        return self.fc(x).reshape(B, T * self.upscaling, self.out_dim)


@propped(nn.Module)
class SampleRNNTier(nn.Module):
    tier_index: int
    frame_size: int
    dim: int
    up_sampling: int = 1
    n_rnn: int = 1
    q_levels: int = 256
    embedding_dim: int = None
    mlp_dim: int = None

    def rnn_(self):
        # no rnn for bottom tier
        if self.up_sampling == 1:
            return None
        return nn.GRU(self.frame_size if self.tier_index == 0 else self.dim,
                      self.dim, self.n_rnn, batch_first=True)

    def up_sampling_net_(self):
        # no up sampling for bottom tier
        if self.up_sampling == 1:
            return None

        class WithRes(nn.Module):
            def __init__(self, mod):
                super(WithRes, self).__init__()
                self.mod = mod

            def forward(self, x):
                return self.mod(x) + x

        # return nn.Sequential(
        #     Ops.Transpose(1, 2),
        #     nn.ConvTranspose1d(self.dim, self.dim,
        #                        kernel_size=self.up_sampling, stride=self.up_sampling),
        #     # nn.Tanh(),
        #     # WithRes(nn.ConvTranspose1d(self.dim, self.dim, kernel_size=3, padding=1)),
        #     nn.Tanh(),
        #     Ops.Transpose(1, 2)
        # )
        return TimeUpscalerLinear(self.dim, self.dim, self.up_sampling)

    def input_proj_(self):
        if self.tier_index == 0:  # top tier
            return None
        if self.embedding_dim is None:  # middle tiers
            return nn.Sequential(
                nn.Linear(self.frame_size,
                          self.dim),
                # nn.Tanh()
            )
        else:  # bottom
            # return nn.Sequential(
            #     nn.Linear(self.frame_size * self.embedding_dim,
            #               self.dim),
            #     # nn.Tanh()
            # )
            class WithRes(nn.Module):
                def __init__(self, mod):
                    super(WithRes, self).__init__()
                    self.mod = mod

                def forward(self, x):
                    return self.mod(x) + x

            return nn.Sequential(
                Ops.Transpose(1, 2),
                nn.Conv1d(self.embedding_dim, self.dim,
                          kernel_size=self.frame_size),
                nn.Tanh(),
                # WithRes(nn.ConvTranspose1d(self.dim, self.dim, kernel_size=3, padding=1)),
                # nn.Tanh(),
            )

    def embeddings_(self):
        if self.embedding_dim is not None:
            new_dim = self.embedding_dim * self.frame_size

            class Reshape(nn.Module):
                def forward(self, x):
                    B, T = x.size()[:2]
                    x = x.view(B*T, *x.size()[2:])
                    return x
                    # return x.view(x.size(0), x.size(1), new_dim)

            return nn.Sequential(
                nn.Embedding(self.q_levels, self.embedding_dim),
                Reshape()
            )
        return None

    def mlp_(self):
        if self.mlp_dim is None:
            return None
        return nn.Sequential(
            nn.Linear(self.dim, self.mlp_dim), nn.Tanh(),
            nn.Linear(self.mlp_dim, self.mlp_dim), nn.Tanh(),
            nn.Linear(self.mlp_dim, self.q_levels, bias=False)
        )

    def __post_init__(self):
        self.inpt_proj = self.input_proj_()
        self.embeddings = self.embeddings_()
        self.rnn = self.rnn_()
        self.up_net = self.up_sampling_net_()
        if self.rnn is not None:
            self.h0 = nn.Parameter(torch.zeros(self.rnn.num_layers, 1, self.rnn.hidden_size))
            # self.c0 = nn.Parameter(torch.zeros(self.rnn.num_layers, 1, self.rnn.hidden_size))
        self.mlp = self.mlp_()

    def linearize(self, q_samples):
        return ((q_samples.float() / (self.q_levels / 2)) - 1) * 2

    def forward(self, input_samples, prev_tier_output=None, hidden=None):
        # print("FORWARD", self.tier_index)
        if self.embeddings is None:
            x = self.linearize(input_samples)
        else:
            x = self.embeddings(input_samples)
            # print(x.size(), prev_tier_output.size())
        if self.inpt_proj is not None:
            B, T = prev_tier_output.size()[:2]
            x = self.inpt_proj(x)
            if self.up_sampling == 1:
                # print(x.size(), prev_tier_output.size())
                x = x.squeeze().reshape(B, T, -1)
            if prev_tier_output is not None:
                x = x + prev_tier_output
        if self.rnn is not None:
            # if hidden is None:
            #     hidden = self.h0.repeat(1, x.size(0), 1), self.c0.repeat(1, x.size(0), 1)
            # else:
            #     hidden = tuple(h[:, :x.size(0)] for h in hidden)
            if hidden is None:
                hidden = self.h0.repeat(1, x.size(0), 1)
            else:
                hidden = hidden[:, :x.size(0)]

            x, hidden = self.rnn(x, hidden)
        else:
            hidden = None
        if self.up_net is not None:
            x = self.up_net(x)
        if self.mlp is not None:
            x = self.mlp(x)
        return x, hidden


@propped(nn.Module)
class SampleRNNNetwork(nn.Module):
    frame_sizes: tuple  # from top to bottom!
    dim: int = 512
    n_rnn: int = 1
    q_levels: int = 256
    embedding_dim: int = None
    mlp_dim: int = None

    def __post_init__(self):
        tiers = []
        for i, fs in enumerate(self.frame_sizes[:-2]):
            tiers += [SampleRNNTier(i, fs, self.dim,
                                    fs // self.frame_sizes[i + 1],
                                    self.n_rnn, self.q_levels, None, None)]
        tiers += [SampleRNNTier(len(self.frame_sizes) - 2, self.frame_sizes[-2], self.dim,
                                self.frame_sizes[-1],
                                self.n_rnn, self.q_levels, None, None)]
        tiers += [SampleRNNTier(len(self.frame_sizes) - 1, self.frame_sizes[-1], self.dim, 1,
                                self.n_rnn, self.q_levels, self.embedding_dim, self.mlp_dim)]
        self.tiers = nn.ModuleList(tiers)
        self.hidden = [None] * len(self.frame_sizes)

    def forward(self, tiers_inputs):
        prev_out = None
        for i, (tier, inpt) in enumerate(zip(self.tiers, tiers_inputs)):
            prev_out, hidden = tier(inpt, prev_out, self.hidden[i].detach() if self.hidden[i] is not None else None)
            # tuple(h.detach() for h in self.hidden[i]) if self.hidden[i] is not None else None)
            self.hidden[i] = hidden

        return prev_out

    def reset_h0(self):
        self.hidden = [None] * len(self.frame_sizes)
