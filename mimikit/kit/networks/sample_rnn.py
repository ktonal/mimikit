import torch.nn as nn
from ..modules import homs as H


class TopTier(nn.Module):
    rnn_class = nn.GRU

    def __init__(self, frame_size, dim, n_rnn, upscaling):
        super(TopTier, self).__init__()
        self.rnn = self.rnn_class(frame_size, dim, n_rnn, batch_first=True)
        # !! ConvTranspose1d has only output_channels biases where orginal source for SampleRNN
        # use linear projections with output_channels * upscaling biases !!
        self.up_net = nn.ConvTranspose1d(dim, dim, kernel_size=upscaling, stride=upscaling)

    def forward(self, x):
        # TODO : x (ints) -> x (-2 floats 2)
        x, _ = self.rnn(x)
        x = self.up_net(x.transpose(1, 2)).transpose(1, 2)
        return x


class MiddleTier(nn.Module):
    rnn_class = nn.GRU

    def __init__(self, frame_size, dim, n_rnn, upscaling):
        super(MiddleTier, self).__init__()
        self.inpt_proj = nn.Linear(frame_size, dim)
        self.rnn = self.rnn_class(dim, dim, n_rnn, batch_first=True)
        self.up_net = nn.ConvTranspose1d(dim, dim, kernel_size=upscaling, stride=upscaling)

    def forward(self, input_samples, prev_tier_output):
        x = self.inpt_proj(input_samples) + prev_tier_output
        x, _ = self.rnn(x)
        x = self.up_net(x.transpose(1, 2)).transpose(1, 2)
        return x


class BottomTier(nn.Module):

    def __init__(self, frame_size, dim, q_levels, emb_size):
        super(BottomTier, self).__init__()
        self.embeddings = nn.Embedding(q_levels, emb_size)
        self.emb_proj = nn.Linear(frame_size * emb_size, dim)
        self.fc_out = nn.Sequential(
            nn.Sequential(nn.Linear(dim, dim), nn.ReLU(inplace=True)),
            nn.Sequential(nn.Linear(dim, dim), nn.ReLU(inplace=True)),
            nn.Linear(dim, q_levels)
        )

    def forward(self, input_samples, prev_tier_output):
        out = self.embeddings(input_samples)
        out = self.emb_proj(out.view(*out.size()[:2], -1))
        out = out + prev_tier_output
        return self.fc_out(out)


class SampleRNNNetwork(H.Tiers):
    pass
