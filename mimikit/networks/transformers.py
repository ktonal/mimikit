from typing import Set, Tuple, Dict
import dataclasses as dtc
import h5mapper as h5m
import torch.nn as nn
import torch
from torch.nn import Transformer, TransformerEncoder, TransformerEncoderLayer,\
    TransformerDecoder, TransformerDecoderLayer
import math

__all__ = [
    'PositionalEncoding',
    'SimpleTransformer'
]

from .arm import ARM
from ..io_spec import IOSpec, ZipReduceVariables
from ..features import ItemSpec, Step
from ..networks.arm import NetworkConfig


class PositionalEncoding(nn.Module):
    r"""Inject some information about the relative or absolute position of the tokens
        in the sequence. The positional encodings have the same dimension as
        the embeddings, so that the two can be summed. Here, we use sine and cosine
        functions of different frequencies.
    .. math::
        \text{PosEncoder}(pos, 2i) = sin(pos/10000^(2i/d_model))
        \text{PosEncoder}(pos, 2i+1) = cos(pos/10000^(2i/d_model))
        \text{where pos is the word position and i is the embed idx)
    Args:
        d_model: the embed dim (required).
        dropout: the dropout value (default=0.1).
        max_len: the max. length of the incoming sequence (default=5000).
    Examples:
        >>> pos_encoder = PositionalEncoding(d_model)
    """

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.pe = nn.Parameter(pe)
        # self.register_buffer('pe', pe)

    def forward(self, x):
        r"""Inputs of forward function
        Args:
            x: the sequence fed to the positional encoder model (required).
        Shape:
            x: [sequence length, batch size, embed dim]
            output: [sequence length, batch size, embed dim]
        Examples:
            >>> output = pos_encoder(x)
        """

        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class SimpleTransformer(ARM, nn.Module):
    @dtc.dataclass
    class Config(NetworkConfig):
        io_spec: IOSpec = None
        model_dim: int = 256
        n_heads: int = 8
        feedforward_dim: int = 1024
        num_layers: int = 8
        with_layer_norm: bool = False
        dropout: float = 0.0
        input_dropout: float = .1
        rf: int = 64

    @classmethod
    def from_config(cls, config: Config):
        layer = TransformerDecoderLayer(d_model=config.model_dim,
                                        nhead=config.n_heads,
                                        dim_feedforward=config.feedforward_dim,
                                        dropout=config.dropout,
                                        activation="relu")
        model = TransformerDecoder(layer, num_layers=config.num_layers,
                                   norm=None if not config.with_layer_norm
                                   else nn.LayerNorm(config.model_dim))
        input_modules = [spec.module.copy()
                             .set(out_dim=config.model_dim)
                             .module()
                         for spec in config.io_spec.inputs]
        input_module = ZipReduceVariables(mode="sum", modules=input_modules)
        output_modules = [spec.module.copy()
                              .set(in_dim=config.model_dim)
                              .module()
                          for spec in config.io_spec.targets]
        return cls(config, model, input_module=input_module, output_modules=output_modules)

    @property
    def config(self) -> NetworkConfig:
        return self._config

    @property
    def rf(self):
        return self._config.rf

    def train_batch(self, item_spec: ItemSpec) -> Tuple[Tuple[h5m.Input, ...], Tuple[h5m.Input, ...]]:
        return tuple(
            spec.to_batch_item(
                item_spec
            )
            for spec in self.config.io_spec.inputs
        ), tuple(
            spec.to_batch_item(
                ItemSpec(shift=1, length=0, unit=Step()) + item_spec
            )
            for spec in self.config.io_spec.targets
        )

    def test_batch(self, item_spec: ItemSpec) -> Tuple[Tuple[h5m.Input, ...], Tuple[h5m.Input, ...]]:
        return self.train_batch(item_spec)

    def before_generate(self, prompts: Tuple[torch.Tensor, ...], batch_index: int) -> None:
        pass

    def generate_step(self, inputs: Tuple[torch.Tensor, ...], *, t: int = 0, **parameters: Dict[str, torch.Tensor]) -> \
            Tuple[torch.Tensor, ...]:
        return self(inputs, **parameters)

    def after_generate(self, final_outputs: Tuple[torch.Tensor, ...], batch_index: int) -> None:
        pass

    @property
    def generate_params(self) -> Set[str]:
        return {"temperature"}

    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def _generate_square_context_mask(self, sz, k=8):
        mask = torch.zeros(sz, sz)
        rg = torch.arange(k)
        for i in range(sz):
            mask[i, torch.clamp(rg + i - (2 // k), min=0, max=sz - 1)] = 1.
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def __init__(self, config: Config, transformer, input_module, output_modules):
        super(SimpleTransformer, self).__init__()
        self._config = config
        self.model = transformer
        self.input_module = input_module
        self.output_modules = nn.ModuleList(output_modules)
        self.dp1d = nn.Dropout1d(config.input_dropout)
        self.src_mask = None
        self.tgt_padding_mask = None
        # self.pe = PositionalEncoding(config.model_dim, dropout=0., max_len=config.rf+2)

    def forward(self, src: Tuple, **parameters):
        src = self.input_module(src)
        if self.training:
            src = self.dp1d(src)
        src = src.permute(1, 0, 2).contiguous()
        device = src.device
        if self.src_mask is None or self.src_mask.size(0) != len(src):
            mask = self._generate_square_subsequent_mask(len(src)).to(device)
            self.src_mask = mask
            self.tgt_padding_mask = torch.zeros(src.size(1), len(src), dtype=torch.bool, device=src.device)
            self.tgt_padding_mask[0] = True
        out = self.model(tgt=src, memory=src,
                         tgt_mask=self.src_mask,
                         memory_mask=self.src_mask,
                         )\
            .permute(1, 0, 2).contiguous()
        if not self.training:
            out = out[:, -1:]
        return tuple(mod(out, **parameters) for mod in self.output_modules)
