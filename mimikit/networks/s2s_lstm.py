import torch
import torch.nn as nn
from typing import Optional

from .arm import NetworkConfig
from ..networks.arm import ARMWithHidden
from ..io_spec import IOSpec
from ..networks.parametrized_gaussian import ParametrizedGaussian

__all__ = [
    'EncoderLSTM',
    'DecoderLSTM',
    'Seq2SeqLSTMNetwork',
    'MultiSeq2SeqLSTM'
]


class EncoderLSTM(nn.Module):
    def __init__(
            self,
            input_d: int = 512,
            model_dim: int = 512,
            num_layers: int = 1,
            n_lstm: Optional[int] = 1,
            bottleneck: Optional[str] = "add",
            n_fc: Optional[int] = 1,
            bias: Optional[bool] = False,
            weight_norm: Optional[bool] = False,
            hop: int = 8,
            with_tbptt: bool = False,
    ):
        super(EncoderLSTM, self).__init__()
        self.model_dim = model_dim
        self.num_layers = num_layers
        self.n_lstm = n_lstm
        self.bottleneck = bottleneck
        self.hop = hop
        self.with_tbptt = with_tbptt

        self.lstms = nn.ModuleList([
            nn.LSTM(input_d if i == 0 else model_dim,
                    model_dim if bottleneck == "add" else model_dim // 2,
                    bias=bias,
                    num_layers=num_layers,
                    batch_first=True, bidirectional=True)
            for i in range(n_lstm)
        ])
        self.fc = nn.Sequential(
            *[nn.Sequential(nn.Linear(model_dim, model_dim), nn.Tanh())
              for _ in range(n_fc - 1)],
            nn.Linear(model_dim, model_dim, bias=False),  # NO ACTIVATION !
        )
        if weight_norm:
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


class DecoderLSTM(nn.Module):
    def __init__(self,
                 model_dim: int = 512,
                 num_layers: int = 1,
                 bottleneck: Optional[str] = "add",
                 bias: Optional[bool] = False,
                 weight_norm: Optional[tuple] = (False, False),
                 ):
        super(DecoderLSTM, self).__init__()
        self.model_dim = model_dim
        self.num_layers = num_layers
        self.bottleneck = bottleneck

        self.lstm1 = nn.LSTM(model_dim, model_dim if bottleneck == "add" else model_dim // 2,
                             bias=self.bias,
                             num_layers=num_layers,
                             batch_first=True, bidirectional=True)
        self.lstm2 = nn.LSTM(model_dim, model_dim if bottleneck == "add" else model_dim // 2,
                             bias=bias,
                             num_layers=num_layers,
                             batch_first=True, bidirectional=True)
        for lstm, wn in zip([self.lstm1, self.lstm2], weight_norm):
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
