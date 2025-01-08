from enum import auto
import dataclasses as dtc
import h5mapper as h5m
import torch
import torch.nn as nn
from typing import Tuple, Set, Dict
from typing_extensions import Literal

from ..features.item_spec import ItemSpec
from ..networks.arm import ARMWithHidden
from .arm import NetworkConfig
from ..io_spec import IOSpec, Continuous
from ..utils import AutoStrEnum
from ..modules import LinearResampler, ZipReduceVariables

__all__ = [
    "EncoderLSTM",
    "DecoderLSTM",
    "Seq2SeqLSTMNetwork"
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
        h0 = torch.randn(2, B, lstm.output_dim).to(x.device) * .01
        c0 = torch.randn(2, B, lstm.output_dim).to(x.device) * .01
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
                 apply_residuals: bool = False,
                 weight_norm: bool = False
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
        self.fc_out = nn.Linear(output_dim, output_dim, bias=False)
        self.hidden = [None] * num_layers
        if weight_norm:
            for module in self.modules():
                if isinstance(module, nn.ModuleList) or list(module.children()) != []:
                    continue
                for name in dict(module.named_parameters()):
                    nn.utils.weight_norm(module, name)

    def forward(self, x):
        if x.size(1) != self.hop:
            assert x.size(1) % self.hop == 0
            n_chunks = x.size(1)//self.hop
            x = x.view(x.size(0) * n_chunks, self.hop, x.size(-1))
        else:
            n_chunks = 1
        inputs = x
        outputs = torch.zeros(inputs.size(0), inputs.size(1), self.output_dim, device=inputs.device)
        if self.training or self.hidden is None:
            self.hidden = [[None] * len(self.lstm)] * n_chunks
        for i in range(n_chunks):
            mask = torch.arange(i, inputs.size(0), step=n_chunks, device=x.device)
            x = inputs[mask]
            for n, lstm in enumerate(self.lstm):
                lstm.flatten_parameters()
                y, self.hidden[i][n] = lstm(x, self.hidden[i][n])
                # y, self.hidden[n] = lstm(x, self.reset_hidden(x, self.hidden[n]))
                # sum forward and backward nets
                y = y.view(*y.size()[:-1], self.output_dim, 2).sum(dim=-1)
                if n > 0 and self.apply_residuals:
                    x = x + y
                else:
                    x = y
            outputs[mask] = x
        x = outputs.view(inputs.size(0)//n_chunks, n_chunks*self.hop, self.output_dim)
        ds = self.downsampling
        if ds == "linear_resample":
            return self.fc_out(self.fc(x)), [h[-1] for h in self.hidden]
        x = x.unfold(1, self.hop, self.hop)
        if 'edge' in ds:
            x = x[..., [0, -1]]
        if 'sum' in ds:
            return self.fc_out(x.sum(dim=-1)), [h[-1] for h in self.hidden]
        return self.fc_out(x.mean(dim=-1)), [h[-1] for h in self.hidden]

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
                 apply_residuals: bool = False,
                 weight_norm: bool = False
                 ):
        super().__init__()
        self.upsampling = upsampling
        self.output_dim = self.model_dim = model_dim
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
        if weight_norm:
            for module in self.modules():
                if isinstance(module, nn.ModuleList) or list(module.children()) != []:
                    continue
                for name in dict(module.named_parameters()):
                    nn.utils.weight_norm(module, name)

    def forward(self, x, hidden=None):
        if x.size(1) > 1:
            n_chunks = x.size(1)
            x = x.view(x.size(0) * n_chunks, 1, x.size(-1))
        else:
            n_chunks = 1
        us = self.upsampling
        if us == "linear_resample":
            x = self.fc(x)
        elif us == 'repeat':
            x = x.repeat_interleave(self.hop, 1)
        elif us == 'interp':
            # ???...
            # interp = nn.functional.interpolate(hidden[0].permute(1, 2, 0), (self.hop,), ).permute(0, 2, 1)
            x = x.expand(-1, self.hop, -1)
        inputs = x
        outputs = torch.zeros(inputs.size(0), self.hop, self.output_dim, device=inputs.device)
        assert isinstance(hidden, list)
        assert len(hidden) == n_chunks
        self.hidden = hidden
        for i in range(n_chunks):
            mask = torch.arange(i, inputs.size(0), step=n_chunks, device=x.device)
            x = inputs[mask]
            self.hidden[i] = hidden[i]
            for n, lstm in enumerate(self.lstm):
                lstm.flatten_parameters()
                y, _ = lstm(x, self.hidden[i])
                # y, self.hidden[n] = lstm(x, self.reset_hidden(x, self.hidden[n]))
                # sum forward and backward nets
                y = y.view(*y.size()[:-1], self.model_dim, 2).sum(dim=-1)
                if self.apply_residuals:
                    x = x + y
                else:
                    x = y
            outputs[mask] = x
        x = outputs.view(outputs.size(0)//n_chunks, n_chunks*self.hop, self.output_dim)
        return x

    def reset_hidden(self, x, h):
        return _reset_hidden(self, x, h)


class Seq2SeqLSTMNetwork(ARMWithHidden, nn.Module):
    @dtc.dataclass
    class Config(NetworkConfig):
        io_spec: IOSpec = None
        model_dim: int = 1024
        enc_downsampling: DownSampling = "edge_sum"
        enc_n_lstm: int = 1
        enc_apply_residuals: bool = False
        enc_weight_norm: bool = False
        dec_upsampling: UpSampling = "linear_resample"
        dec_n_lstm: int = 1
        dec_apply_residuals: bool = False
        dec_weight_norm: bool = False
        hop: int = 8

    @classmethod
    def from_config(cls, cfg: "Seq2SeqLSTMNetwork.Config"):
        if isinstance(cfg.io_spec.inputs[0].elem_type, Continuous):
            # no input module if input is FFT
            input_dim = cfg.io_spec.inputs[0].elem_type.size
            input_module = sum
        else:
            input_dim = cfg.model_dim
            input_modules = [spec.module.copy()
                                 .set(out_dim=cfg.model_dim, frame_size=1, hop_length=1)
                                 .module()
                             for spec in cfg.io_spec.inputs]
            input_module = ZipReduceVariables(mode="sum", modules=input_modules)

        enc = EncoderLSTM(
            downsampling=cfg.enc_downsampling,
            input_dim=input_dim,
            output_dim=cfg.model_dim, num_layers=cfg.enc_n_lstm,
            weight_norm=cfg.enc_weight_norm, hop=cfg.hop, apply_residuals=cfg.enc_apply_residuals
        )
        dec = DecoderLSTM(
            upsampling=cfg.dec_upsampling, model_dim=cfg.model_dim, num_layers=cfg.dec_n_lstm,
            hop=cfg.hop, apply_residuals=cfg.dec_apply_residuals, weight_norm=cfg.dec_apply_residuals
        )
        output_modules = [spec.module.copy()
                              .set(in_dim=cfg.model_dim)
                              .module()
                          for spec in cfg.io_spec.targets]
        output_module = ZipReduceVariables(mode="sum", modules=output_modules)
        return cls(cfg, input_module=input_module, output_module=output_module, encoder=enc, decoder=dec)

    def __init__(self,
                 config: "Seq2SeqLSTMNetwork.Config",
                 input_module: nn.Module,
                 output_module: nn.Module,
                 encoder: EncoderLSTM,
                 decoder: DecoderLSTM,
                 ):
        super(Seq2SeqLSTMNetwork, self).__init__()
        self._config = config
        self.input_module = input_module
        self.enc = encoder
        self.dec = decoder
        self.output_module = output_module
        self.output_length = lambda n: n

    def forward(self, x: Tuple, temperature=None):
        x = self.input_module(x)
        coded, hidden = self.enc(x)
        return self.decode(coded, hidden, temperature)

    def decode(self, x, hidden, temperature=None):
        output = self.dec(x, hidden)
        return self.output_module((output,), *((temperature,) if temperature is not None else ()))

    def reset_hidden(self):
        self.enc.hidden = None
        self.dec.hidden = None

    def before_generate(self, prompts: Tuple[torch.Tensor, ...], batch_index: int) -> None:
        self.reset_hidden()
        # initialize hidden
        self(tuple(p[:, p.size(1)%self._config.hop:-self._config.hop] for p in prompts))
        # keep only the lasts
        self.enc.hidden = [self.enc.hidden[-1]]
        return

    def generate_step(self, inputs: Tuple[torch.Tensor, ...], *, t: int = 0, **parameters: Dict[str, torch.Tensor]) -> \
    Tuple[torch.Tensor, ...]:
        return self.forward(inputs)

    def after_generate(self, final_outputs: Tuple[torch.Tensor, ...], batch_index: int) -> None:
        self.reset_hidden()

    @property
    def generate_params(self) -> Set[str]:
        return {p for m in getattr(self.output_module, "heads", []) for p in getattr(m, "sampling_params", {})}

    @property
    def config(self) -> NetworkConfig:
        return self._config

    @property
    def rf(self):
        return self._config.hop

    def train_batch(self, item_spec: ItemSpec) -> Tuple[Tuple[h5m.Input, ...], Tuple[h5m.Input, ...]]:
        hop = self._config.hop
        return tuple(
            spec.to_batch_item(
                ItemSpec(shift=0, unit=item_spec.unit) + item_spec
            )
            for spec in self.config.io_spec.inputs
        ), tuple(
            spec.to_batch_item(
                ItemSpec(shift=hop, unit=item_spec.unit) + item_spec
            )
            for spec in self.config.io_spec.targets
        )

    def test_batch(self, item_spec: ItemSpec) -> Tuple[Tuple[h5m.Input, ...], Tuple[h5m.Input, ...]]:
        return tuple(
            spec.to_batch_item(
                item_spec
            )
            for spec in self.config.io_spec.inputs
        ), ()

