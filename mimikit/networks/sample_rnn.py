import torch
import torch.nn as nn
from typing import Optional
from dataclasses import dataclass

from .generating_net import GeneratingNetwork

__all__ = [
    "SampleRNNTier",
    "SampleRNNNetwork"
]


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

    def embeddings_(self):
        if self.embedding_dim is not None:
            return nn.Embedding(self.q_levels, self.embedding_dim)
        return None

    def input_proj_(self):

        if not self.is_bottom:  # top & middle tiers
            return nn.Sequential(nn.Linear(self.frame_size, self.dim), )

        else:  # bottom tier
            class BottomProjector(nn.Module):
                def __init__(self, emb_dim, out_dim, frame_size):
                    super(BottomProjector, self).__init__()
                    self.cnv = nn.Conv1d(emb_dim, out_dim, kernel_size=frame_size)

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
        # we do not learn hidden for practical reasons : storing them as params in the state dict
        # would force the network to always generate with the train batch_size or to implement some logic
        # to handle this shape change. We choose the simpler solution : hidden are initialized to zeros
        # whenever we need fresh ones.
        self.h0, self.c0 = None, None
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

        class ConvUpscaler(nn.Module):
            def __init__(self, dim, up_sampling):
                super(ConvUpscaler, self).__init__()
                self.cv = nn.ConvTranspose1d(dim, dim, kernel_size=up_sampling, stride=up_sampling)

            def forward(self, x):
                return self.cv(x.transpose(1, 2).contiguous()).transpose(1, 2).contiguous()

        return TimeUpscalerLinear(self.dim, self.dim, self.up_sampling)

    def mlp_(self):
        if self.mlp_dim is None:
            return None
        return nn.Sequential(
            nn.Linear(self.dim, self.mlp_dim), nn.ReLU(),
            nn.Linear(self.mlp_dim, self.mlp_dim), nn.ReLU(),
            nn.Linear(self.mlp_dim, self.q_levels if self.is_bottom else self.dim),
        )

    def __post_init__(self):
        nn.Module.__init__(self)
        self.embeddings = self.embeddings_()
        self.inpt_proj = self.input_proj_()
        self.rnn = self.rnn_()
        self.up_net = self.up_sampling_net_()
        self.mlp = self.mlp_()

    def reset_hidden(self, batch_size, device):
        self.h0 = torch.zeros(self.n_rnn, batch_size, self.dim).to(device)
        self.c0 = torch.zeros(self.n_rnn, batch_size, self.dim).to(device)
        return self.h0, self.c0

    def linearize(self, q_samples):
        """ maps input samples (0 <= qx < 256) to floats (-2. <= x < 2.) """
        return ((q_samples.float() / self.q_levels) - .5) * 4

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
                hidden = self.reset_hidden(x.size(0), x.device)
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
class SampleRNNNetwork(GeneratingNetwork, nn.Module):

    frame_sizes: tuple = (16, 8, 8)  # from top to bottom!
    dim: int = 512
    n_rnn: int = 2
    q_levels: int = 256
    embedding_dim: int = 256
    mlp_dim: int = 512

    def __post_init__(self):
        nn.Module.__init__(self)
        GeneratingNetwork.__init__(self)
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

