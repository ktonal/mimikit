import torch
import torch.nn as nn
from typing import Optional
import dataclasses as dtc

from mimikit.modules.misc import Abs
from pytorch_lightning.utilities import AttributeDict

from ..networks.parametrized_gaussian import ParametrizedGaussian

__all__ = [
    'EncoderLSTM',
    'DecoderLSTM',
    'Seq2SeqLSTMNetwork',
    'MultiSeq2SeqLSTM'
]


@dtc.dataclass(init=True, repr=False, eq=False, frozen=False, unsafe_hash=True)
class EncoderLSTM(nn.Module):
    input_d: int = 512
    model_dim: int = 512
    num_layers: int = 1
    n_lstm: Optional[int] = 1
    bottleneck: Optional[str] = "add"
    n_fc: Optional[int] = 1
    bias: Optional[bool] = False
    weight_norm: Optional[bool] = False
    hop: int = 8
    with_tbptt: bool = False

    def __post_init__(self):
        nn.Module.__init__(self)
        self.lstms = nn.ModuleList([
            nn.LSTM(self.input_d if i == 0 else self.model_dim,
                    self.model_dim if self.bottleneck == "add" else self.model_dim // 2,
                    bias=self.bias,
                    num_layers=self.num_layers,
                    batch_first=True, bidirectional=True)
            for i in range(self.n_lstm)
        ])
        self.fc = nn.Sequential(
            *[nn.Sequential(nn.Linear(self.model_dim, self.model_dim), nn.Tanh()) for _ in range(self.n_fc - 1)],
            nn.Linear(self.model_dim, self.model_dim, bias=False),  # NO ACTIVATION !
        )
        if self.weight_norm:
            for lstm in self.lstms:
                for name, p in dict(lstm.named_parameters()).items():
                    if "weight" in name:
                        torch.nn.utils.weight_norm(lstm, name)
        self.hidden = None

    def forward(self, x):
        ht, ct = None, None
        hidden = self.reset_hidden(x, self.hidden)
        for i, lstm in enumerate(self.lstms):
            out, (ht, ct) = lstm(x, hidden)
            # sum forward and backward nets
            out = out.view(*out.size()[:-1], self.model_dim, 2).sum(dim=-1)
            # take residuals AFTER the first lstm
            x = out if i == 0 else x + out
            if self.with_tbptt:
                self.hidden = ht, ct
        states = self.first_and_last_states(x)
        out = self.fc(states)
        return out, (ht, ct)

    def first_and_last_states(self, sequence):
        rg = torch.arange(sequence.size(1) // self.hop)
        first_states = sequence[:, rg * self.hop, :]
        last_states = sequence[:, (rg + 1) * self.hop - 1, :]
        if self.bottleneck == "add":
            return first_states + last_states
        else:
            return torch.cat((first_states, last_states), dim=-1)

    def reset_hidden(self, x, h):
        if h is None or x.size(0) != h[0].size(1):
            B = x.size(0)
            h0 = torch.zeros(self.num_layers * 2, B, self.model_dim).to(x.device)
            c0 = torch.zeros(self.num_layers * 2, B, self.model_dim).to(x.device)
            return h0, c0
        else:
            return tuple(h_.detach() for h_ in h)


@dtc.dataclass(init=True, repr=False, eq=False, frozen=False, unsafe_hash=True)
class DecoderLSTM(nn.Module):
    model_dim: int = 512
    num_layers: int = 1
    bottleneck: Optional[str] = "add"
    bias: Optional[bool] = False
    weight_norm: Optional[tuple] = (False, False)

    def __post_init__(self):
        nn.Module.__init__(self)
        self.lstm1 = nn.LSTM(self.model_dim, self.model_dim if self.bottleneck == "add" else self.model_dim // 2,
                             bias=self.bias,
                             num_layers=self.num_layers, batch_first=True, bidirectional=True)
        self.lstm2 = nn.LSTM(self.model_dim, self.model_dim if self.bottleneck == "add" else self.model_dim // 2,
                             bias=self.bias,
                             num_layers=self.num_layers, batch_first=True, bidirectional=True)
        for lstm, wn in zip([self.lstm1, self.lstm2], self.weight_norm):
            if wn:
                for name, p in dict(lstm.named_parameters()).items():
                    if "weight" in name:
                        torch.nn.utils.weight_norm(lstm, name)

    def forward(self, x, hidden, cells):
        if hidden is None or cells is None:
            output, (_, _) = self.lstm1(x)
        else:
            # ALL decoders get hidden states from encoder
            output, (_, _) = self.lstm1(x, (hidden, cells))
        # sum forward and backward nets
        output = output.view(*output.size()[:-1], self.model_dim, 2).sum(dim=-1)
        if hidden is None or cells is None:
            output2, (hidden, cells) = self.lstm2(output)
        else:
            output2, (hidden, cells) = self.lstm2(output, (hidden, cells))
        # sum forward and backward nets
        output2 = output2.view(*output2.size()[:-1], self.model_dim, 2).sum(dim=-1)
        # sum the outputs
        return output + output2, (hidden, cells)


def tile(a, dim, n_tile):
    init_dim = a.size(dim)
    repeat_idx = [1] * a.dim()
    repeat_idx[dim] = n_tile
    a = a.repeat(*(repeat_idx,))
    order_index = torch.LongTensor(torch.cat([init_dim * torch.arange(n_tile) + i for i in range(init_dim)]))
    return torch.index_select(a, dim, order_index.to(a.device))


class Seq2SeqLSTMNetwork(nn.Module):
    device = property(lambda self: next(self.parameters()).device)

    def __init__(self,
                 input_dim: int = 513,
                 model_dim: int = 1024,
                 num_layers: int = 1,
                 n_lstm: int = 1,
                 bottleneck: str = "add",
                 n_fc: int = 1,
                 hop: int = 8,
                 bias: Optional[bool] = False,
                 weight_norm: bool = False,
                 input_module: Optional[nn.Module] = None,
                 output_module: Optional[nn.Module] = None,
                 with_tbptt: bool = False,
                 with_sampler: bool = True
                 ):
        init_ctx = locals()
        super(Seq2SeqLSTMNetwork, self).__init__()
        init_ctx.pop("output_module")
        init_ctx.pop("input_module")
        init_ctx.pop("self")
        init_ctx.pop("__class__")
        self.hp = AttributeDict(init_ctx)
        self.inpt_mod = input_module if input_module is not None else nn.Identity()
        self.enc = EncoderLSTM(input_dim,
                               model_dim, num_layers, n_lstm, bottleneck, n_fc,
                               bias=bias, weight_norm=weight_norm, hop=hop,
                               with_tbptt=with_tbptt)
        self.dec = DecoderLSTM(model_dim, num_layers, bottleneck,
                               bias=bias, weight_norm=(weight_norm,) * 2)
        if with_sampler:
            self.sampler = ParametrizedGaussian(model_dim, model_dim, bias=bias)
        self.outpt_mod = nn.Sequential(
            nn.Linear(model_dim, input_dim, bias=False), Abs()
        ) if output_module is None else output_module
        self.hop = self.rf = self.shift = self.hp.hop
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


class MultiSeq2SeqLSTM(nn.Module):
    device = property(lambda self: next(self.parameters()).device)

    def __init__(self):
        super(MultiSeq2SeqLSTM, self).__init__()
        self.hp = {}
        lstms = []
        for i in range(3):
            lstms += [Seq2SeqLSTMNetwork(
                input_dim=256 if i > 0 else 513,
                model_dim=256,
                hop=4
            )]
        self.s2s = nn.ModuleList(lstms)

    def forward(self, x, i=0):
        x, (h_enc, c_enc) = self.s2s[i].enc(x)
        if i == len(self.s2s) - 1:
            return self.s2s[i].decode(x, h_enc, c_enc)
        else:
            return self.s2s[i].decode(self.forward(x, i + 1), h_enc, c_enc)

    def reset_hidden(self):
        for s2s in self.s2s:
            s2s.enc.hidden = None

    def generate_step(self, t, inputs, ctx):
        return self.forward(*inputs)
