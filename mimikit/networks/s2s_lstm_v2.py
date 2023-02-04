from enum import auto

import torch
import torch.nn as nn
from typing import *

from ..networks.arm import ARMWithHidden
from ..config import NetworkConfig
from ..io_spec import IOSpec
from ..utils import AutoStrEnum
from ..modules import LinearResampler


__all__ = [
    "EncoderLSTM",
    "DecoderLSTM",
]


class DownSampling(AutoStrEnum):
    edge_sum = auto()
    edge_mean = auto()
    sum = auto()
    mean = auto()
    linear_resample = auto()


class UpSampling(AutoStrEnum):
    repeat = auto()
    interp = auto()
    linear_resample = auto()


class H0Init(AutoStrEnum):
    zeros = auto()
    ones = auto()
    randn = auto()


def _reset_hidden(lstm, x, h):
    if h is None or x.size(0) != h[0].size(1):
        B = x.size(0)
        h0 = torch.zeros(2, B, lstm.output_dim).to(x.device)
        c0 = torch.zeros(2, B, lstm.output_dim).to(x.device)
        return h0, c0
    else:
        return tuple(h_.detach() for h_ in h)


class EncoderLSTM(nn.Module):
    def __init__(self,
                 downsampling: Literal[
                     'edge_sum', 'edge_mean',
                     'sum', 'mean', 'linear_resample'
                 ],
                 input_dim: int = 512,
                 output_dim: int = 512,
                 num_layers: int = 1,
                 hop: int = 4,
                 apply_residuals: bool = False
                 ):
        super().__init__()
        self.downsampling = downsampling
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.hop = hop
        self.apply_residuals = apply_residuals
        self.lstm = nn.ModuleList(
            [nn.LSTM(input_dim, output_dim,
                     batch_first=True,
                     bidirectional=True
                     ),
             *(nn.LSTM(output_dim, output_dim,
                       batch_first=True, bidirectional=True)
               for _ in range(num_layers - 1))
             ])
        if downsampling == "linear_resample":
            self.fc = LinearResampler(output_dim, 1 / hop, 1)
        self.fc_out = nn.Linear(output_dim, output_dim)
        self.hidden = [None] * num_layers

    def forward(self, x):
        assert x.size(1) == self.hop
        for n, lstm in enumerate(self.lstm):
            y, self.hidden[n] = lstm(x, self.reset_hidden(x, self.hidden[n]))
            # sum forward and backward nets
            y = y.view(*y.size()[:-1], self.output_dim, 2).sum(dim=-1)
            if n > 0 and self.apply_residuals:
                x += y
            else:
                x = y
        ds = self.downsampling
        if ds == "linear_resample":
            return self.fc_out(self.fc(x)), self.hidden[-1]
        x = x.unfold(1, self.hop, self.hop)
        if 'edge' in ds:
            x = x[..., [0, -1]]
        if 'sum' in ds:
            return self.fc_out(x.sum(dim=-1)), self.hidden[-1]
        return self.fc_out(x.mean(dim=-1)), self.hidden[-1]

    def reset_hidden(self, x, h):
        return _reset_hidden(self, x, h)


class DecoderLSTM(nn.Module):
    def __init__(self,
                 upsampling: Literal[
                     'repeat', 'interp', 'linear_resample'
                 ],
                 model_dim: int = 512,
                 num_layers: int = 1,
                 hop: int = 4,
                 apply_residuals: bool = False
                 ):
        super().__init__()
        self.upsampling = upsampling
        self.model_dim = model_dim
        self.num_layers = num_layers
        self.hop = hop
        self.apply_residuals = apply_residuals
        self.lstm = nn.ModuleList(
            [nn.LSTM(model_dim, model_dim,
                     batch_first=True,
                     bidirectional=True
                     ),
             *(nn.LSTM(model_dim, model_dim,
                       batch_first=True, bidirectional=True)
               for _ in range(num_layers - 1))
             ])
        if upsampling == "linear_resample":
            self.fc = LinearResampler(model_dim, hop, 1)
        self.hidden = [None] * num_layers

    def forward(self, x, hidden=None):
        assert x.size(1) == 1
        us = self.upsampling
        if us == "linear_resample":
            x = self.fc(x)
        elif us == 'repeat':
            x = x.repeat_interleave(self.hop, 1)
        elif us == 'interp':
            # ???...
            interp = nn.functional.interpolate(hidden[0].permute(1, 2, 0), (self.hop,), ).permute(0, 2, 1)
            x = x.expand(-1, self.hop, -1) + interp
        # only first lstm gets hidden from encoder
        # (!= to prev impl: all decoder got last encoder hidden)
        self.hidden[0] = hidden
        for n, lstm in enumerate(self.lstm):
            y, self.hidden[n] = lstm(x, self.reset_hidden(x, self.hidden[n]))
            # sum forward and backward nets
            y = y.view(*y.size()[:-1], self.model_dim, 2).sum(dim=-1)
            if self.apply_residuals:
                x += y
            else:
                x = y
        return x

    def reset_hidden(self, x, h):
        return _reset_hidden(self, x, h)


class Seq2SeqLSTMNetwork(ARMWithHidden, nn.Module):
    class Config(NetworkConfig):
        io_spec: IOSpec = None
        input_dim: int = 513
        model_dim: int = 1024
        num_layers: int = 1
        n_lstm: int = 1
        bottleneck: str = "add"
        n_fc: int = 1
        hop: int = 8
        bias: Optional[bool] = False
        weight_norm: bool = False
        input_module: Optional[nn.Module] = None
        output_module: Optional[nn.Module] = None
        with_tbptt: bool = False
        with_sampler: bool = True

    @classmethod
    def from_config(cls, cfg: "Seq2SeqLSTMNetwork.Config"):
        enc = EncoderLSTM(cfg.input_dim,
                          cfg.model_dim, cfg.num_layers, cfg.n_lstm, cfg.bottleneck, cfg.n_fc,
                          bias=cfg.bias, weight_norm=cfg.weight_norm, hop=cfg.hop,
                          with_tbptt=cfg.with_tbptt)
        dec = DecoderLSTM(cfg.model_dim, cfg.num_layers, cfg.bottleneck,
                          bias=cfg.bias, weight_norm=(cfg.weight_norm,) * 2)
        return cls(cfg, input_module=None, output_module=None, encoder=enc, decoder=dec)

    def __init__(self,
                 config: "Seq2SeqLSTMNetwork.Config",
                 input_module: nn.Module,
                 output_module: nn.Module,
                 encoder: EncoderLSTM,
                 decoder: DecoderLSTM,
                 ):
        super(Seq2SeqLSTMNetwork, self).__init__()
        self._config = config
        self.inpt_mod = input_module
        self.enc = encoder
        self.dec = decoder
        if config.with_sampler:
            self.sampler = ParametrizedGaussian(config.model_dim, config.model_dim, bias=config.bias)
        self.outpt_mod = output_module
        self.output_length = lambda n: n

    def forward(self, x, temperature=None):
        x = self.inpt_mod(x)
        coded, (h_enc, c_enc) = self.enc(x)
        return self.decode(coded, h_enc, c_enc, temperature)

    def decode(self, x, h_enc, c_enc, temperature=None):
        coded = tile((x.unsqueeze(1) if len(x.shape) < 3 else x), 1, self.hp.hop)
        if self.hp.with_sampler:
            residuals, _, _ = self.sampler(coded)
            coded = coded + residuals
        output, (_, _) = self.dec(coded, h_enc, c_enc)
        return self.outpt_mod(output, *((temperature,) if temperature is not None else ()))

    def reset_hidden(self):
        self.enc.hidden = None

    def before_generate(self, loop, batch, batch_idx):
        self.reset_hidden()
        self.forward(*batch)
        return {}

    def generate_step(self, t, inputs, ctx):
        return self.forward(*inputs)

    def after_generate(self, outputs, ctx, batch_idx):
        self.reset_hidden()
