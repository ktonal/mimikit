from typing import Set, Tuple, Dict, Optional, Iterable, List
import dataclasses as dtc
import h5mapper as h5m
import torch.nn as nn
import torch
from torch.nn import TransformerDecoder, TransformerDecoderLayer, Transformer
import math

__all__ = [
    'PositionalEncoding',
    'SimpleTransformer',
    "JukeBox"
]

from .arm import ARM
from ..modules import LinearResampler, FramedLinearIO, FramedConv1dIO, EmbeddingConv1d
from ..io_spec import IOSpec, ZipReduceVariables
from ..features import ItemSpec, Step, Discrete
from ..networks.arm import NetworkConfig

T = torch.Tensor


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
        # TODO: REVERSE for alignment with last position, regardless of input'slength
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        # self.pe = nn.Parameter(pe)
        self.register_buffer('pe', pe)

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

    def __init__(self, config: Config, transformer, input_module, output_modules):
        super(SimpleTransformer, self).__init__()
        self._config = config
        self.model = transformer
        self.input_module = input_module
        self.output_modules = nn.ModuleList(output_modules)
        self.dp1d = nn.Dropout1d(config.input_dropout)
        self.src_mask = None
        self.tgt_padding_mask = None
        self.pe = PositionalEncoding(config.model_dim, dropout=0., max_len=2048)

    # TODO: generate_step with query: (N, 1, D) and keys/values: (N, RF, D)
    def forward(self, src: Tuple, **parameters):
        src = self.input_module(src)
        if self.training:
            src = self.dp1d(src)
        src = src.permute(1, 0, 2).contiguous()
        src = self.pe(src)
        device = src.device
        if self.src_mask is None or self.src_mask.size(0) != len(src):
            mask = self._generate_square_subsequent_mask(len(src)).to(device)
            self.src_mask = mask
            self.tgt_padding_mask = torch.zeros(src.size(1), len(src), dtype=torch.bool, device=src.device)
            self.tgt_padding_mask[0] = True
        out = self.model(tgt=src, memory=src,
                         tgt_mask=self.src_mask,
                         memory_mask=self.src_mask,
                         ) \
            .permute(1, 0, 2).contiguous()
        if not self.training:
            out = out[:, -1:]
        return tuple(mod(out, **parameters) for mod in self.output_modules)


class TransformerTier(nn.Module):

    def __init__(
            self, *,
            input_module: nn.Module = nn.Identity(),
            model_dim: Optional[int] = 256,
            n_heads: int = 8,
            feedforward_dim: int = 1024,
            num_layers: int = 8,
            with_layer_norm: bool = False,
            dropout: float = 0.0,
            activation: nn.Module = nn.ReLU(),
            norm_first: bool = False,
            positional_encoding: Optional[int] = 4096,
            weight_norm: bool = False,
            up_sampling: Optional[int] = None,
    ):
        super(TransformerTier, self).__init__()
        self.input_module = input_module
        self.weight_norm = weight_norm
        self.up_sampling = up_sampling
        self.has_transformer = model_dim is not None
        self.has_pe = positional_encoding is not None and self.has_transformer
        if self.has_pe:
            self.pe = PositionalEncoding(model_dim, dropout=0., max_len=positional_encoding)
        if self.has_transformer:
            self.src_mask = None
            self.tgt_padding_mask = None
            layer = TransformerDecoderLayer(d_model=model_dim,
                                            nhead=n_heads,
                                            dim_feedforward=feedforward_dim,
                                            dropout=dropout,
                                            norm_first=norm_first,
                                            activation=activation)
            self.model = TransformerDecoder(layer, num_layers=num_layers,
                                            norm=None if not with_layer_norm
                                            else nn.LayerNorm(model_dim))

        self.has_up_sampling = up_sampling is not None
        if self.has_up_sampling:
            self.up_sampler = LinearResampler(model_dim, t_factor=up_sampling, d_factor=1)
            if weight_norm:
                for module in self.up_sampler.children():
                    for name in dict(module.named_parameters()):
                        nn.utils.weight_norm(module, name)
        if weight_norm:
            for module in self.input_module.modules():
                if isinstance(module, nn.ModuleList) or list(module.children()) != []:
                    continue
                for name in dict(module.named_parameters()):
                    nn.utils.weight_norm(module, name)

    def forward(
            self,
            inputs: Tuple[Tuple[T, ...], Optional[T]]
    ) -> T:
        x, x_upper = inputs
        x = self.input_module(x)
        if x_upper is not None:
            x += x_upper
        if self.has_transformer:
            x = x.permute(1, 0, 2).contiguous()
            if self.has_pe:
                x = self.pe(x)
            if self.src_mask is None or self.src_mask.size(0) != len(x):
                device = x.device
                mask = Transformer.generate_square_subsequent_mask(len(x), device)
                self.src_mask = mask
                self.tgt_padding_mask = torch.zeros(x.size(1), len(x), dtype=torch.bool, device=x.device)
                self.tgt_padding_mask[0] = True
            x = self.model(tgt=x, memory=x,
                           tgt_mask=self.src_mask,
                           memory_mask=self.src_mask,
                           ) \
                .permute(1, 0, 2).contiguous()
            x = nn.Tanh()(x)
        if self.has_up_sampling:
            x = self.up_sampler(x)
            # x: (batch, n_frames * up_sampling, hidden_dim)
        return x


class JukeBox(ARM, nn.Module):

    @dtc.dataclass
    class Config(NetworkConfig):
        io_spec: IOSpec = None
        frame_sizes: Tuple[int, ...] = (32, 16, 4)
        model_dim: int = 256
        n_heads: int = 8
        feedforward_dim: int = 1024
        num_layers: int = 1
        layer_activation: str = "Mish"
        norm_first: bool = False
        with_layer_norm: bool = False
        dropout: float = 0.0
        positional_encoding: Optional[int] = 4096
        weight_norm: bool = False
        input_dropout: float = 0.
        rf: int = 64

    @classmethod
    def from_config(cls, config: Config):
        tiers = []
        h_dim = config.model_dim
        for i, fs in enumerate(config.frame_sizes[:-1]):
            modules = tuple(in_spec.module.copy()
                            .set(frame_size=fs, hop_length=fs, out_dim=h_dim).module()
                            for in_spec in config.io_spec.inputs)
            input_module = ZipReduceVariables(mode="sum", modules=modules)
            tiers += [
                TransformerTier(
                    input_module=input_module,
                    model_dim=config.model_dim,
                    n_heads=config.n_heads,
                    feedforward_dim=config.feedforward_dim,
                    num_layers=config.num_layers,
                    with_layer_norm=config.with_layer_norm,
                    dropout=config.dropout,
                    activation=getattr(nn, config.layer_activation)(),
                    norm_first=config.norm_first,
                    positional_encoding=config.positional_encoding,
                    weight_norm=config.weight_norm,
                    up_sampling=fs // (
                        config.frame_sizes[i + 1]
                        if i < len(config.frame_sizes) - 2
                        else 1)
                )]
        modules = []
        for in_spec in config.io_spec.inputs:
            if isinstance(in_spec.elem_type, Discrete):
                params = dict(class_size=in_spec.elem_type.size)
                if isinstance(in_spec.module, FramedLinearIO):
                    module_type = FramedConv1dIO
                else:
                    module_type = EmbeddingConv1d
            else:
                params = dict()
                module_type = FramedConv1dIO
            modules += [module_type()
                            .set(**params,
                                 frame_size=config.frame_sizes[-1],
                                 hop_length=1, out_dim=h_dim).module()]
        input_module = ZipReduceVariables(mode="sum", modules=modules)
        tiers += [
            TransformerTier(
                input_module=input_module,
                model_dim=None,
                weight_norm=config.weight_norm,
                up_sampling=None
            )]
        output_module = [target_spec.module.copy().set(in_dim=h_dim).module()
                         for target_spec in config.io_spec.targets]
        return cls(
            config=config, tiers=tiers, output_module=output_module)

    def __init__(
            self, *,
            config: "JukeBox.Config",
            tiers: Iterable[nn.Module],
            output_module: List[nn.Module],
    ):
        super(JukeBox, self).__init__()
        self._config = config
        self.frame_sizes = config.frame_sizes
        self.tiers: List[TransformerTier] = nn.ModuleList(tiers)
        self.output_modules = nn.ModuleList(output_module)

        if config.weight_norm:
            for module in self.output_modules.modules():
                if isinstance(module, nn.ModuleList) or list(module.children()) != []:
                    continue
                for name in dict(module.named_parameters()):
                    nn.utils.weight_norm(module, name)

    def forward(self, inputs: Tuple, **parameters):
        prev_output = None
        fs0 = self.frame_sizes[0]
        for tier, fs in zip(self.tiers[:-1], self.frame_sizes[:-1]):
            tier_input = tuple(inpt[:, fs0 - fs:-fs] for inpt in inputs)
            prev_output = tier.forward((tier_input, prev_output))
        fs = self.frame_sizes[-1]
        # :-1 is surprising but right!
        tier_input = tuple(inpt[:, fs0 - fs:-1] for inpt in inputs)
        prev_output = self.tiers[-1].forward((tier_input, prev_output))
        if not self.training:
            prev_output = prev_output[:, -1:]
        output = tuple(mod(prev_output, **parameters) for mod in self.output_modules)
        return output

    @property
    def config(self) -> NetworkConfig:
        return self._config

    @property
    def rf(self):
        return self._config.rf

    def train_batch(self, item_spec: ItemSpec):
        # fit lengths to target -> input gets extra
        return tuple(
            spec.to_batch_item(
                ItemSpec(shift=0, length=self.frame_sizes[0],
                         unit=spec.unit) + item_spec
            )
            for spec in self.config.io_spec.inputs
        ), tuple(
            spec.to_batch_item(
                ItemSpec(shift=self.frame_sizes[0], unit=spec.unit) + item_spec
            )
            for spec in self.config.io_spec.targets
        )

    def test_batch(self, item_spec: ItemSpec):
        # fit lengths to input -> target looses extra
        return tuple(
            spec.to_batch_item(
                item_spec.to(spec.unit)
            )
            for spec in self.config.io_spec.inputs
        ), tuple(
            spec.to_batch_item(
                ItemSpec(shift=self.frame_sizes[0],
                         length=-self.frame_sizes[0],
                         unit=spec.unit) + item_spec
            )
            for spec in self.config.io_spec.targets
        )

    def before_generate(self, prompts: Tuple[torch.Tensor, ...], batch_index: int) -> None:
        pass

    def generate_step(
            self,
            inputs: Tuple[torch.Tensor, ...], *,
            t: int = 0,
            **parameters: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, ...]:
        # TODO: SampleRNN like generate_step()
        return self(inputs, **parameters)

    def after_generate(self, final_outputs: Tuple[torch.Tensor, ...], batch_index: int) -> None:
        pass

    @property
    def generate_params(self) -> Set[str]:
        return {"temperature"}
