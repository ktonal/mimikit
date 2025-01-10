from typing import Set, Tuple, Dict, Optional, Iterable, List
import dataclasses as dtc
import h5mapper as h5m
import torch.nn as nn
import torch
from torch import Tensor
from torch.nn import TransformerDecoder, TransformerDecoderLayer, Transformer
import math

__all__ = [
    'PositionalEncoding',
    "TransformerTier",
    "JukeBox"
]

from .arm import ARM
from ..modules import LinearResampler, FramedLinearIO, FramedConv1dIO, EmbeddingConv1d, OneHotConv1dIO
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


class TransformerDecoderLayerForCaching(nn.TransformerDecoderLayer):

    def __init__(self, *args, **kwargs):
        super(TransformerDecoderLayerForCaching, self).__init__(*args, **kwargs)
        # self.linear1 = self.linear2 = nn.Identity()
        # self.norm1 = self.norm2 = self.norm3

    def forward(
            self,
            tgt: Tensor,
            memory: Tensor,
            tgt_mask: Optional[Tensor] = None,
            memory_mask: Optional[Tensor] = None,
            tgt_key_padding_mask: Optional[Tensor] = None,
            memory_key_padding_mask: Optional[Tensor] = None,
            tgt_is_causal: bool = False,
            memory_is_causal: bool = False,
            caching_pass: bool = False
    ) -> Tensor:
        if self.training or caching_pass:
            x = tgt
            if self.norm_first:
                x = x + self._sa_block(self.norm1(x), tgt_mask, tgt_key_padding_mask, tgt_is_causal,
                                       False)
                x = x + self._mha_block(self.norm2(x), memory, memory_mask, memory_key_padding_mask, memory_is_causal)
                x = x + self._ff_block(self.norm3(x))
            else:
                x = self.norm1(x + self._sa_block(x, tgt_mask, tgt_key_padding_mask, tgt_is_causal,
                                                  False))
                x = self.norm2(x + self._mha_block(x, memory, memory_mask, memory_key_padding_mask, memory_is_causal))
                x = self.norm3(x + self._ff_block(x))
        else:
            x = tgt
            if self.norm_first:
                x = x[-1:] + self._sa_block(self.norm1(x), tgt_mask, tgt_key_padding_mask, tgt_is_causal,
                                            True)
                # x is now only the last step
                x = x + self._mha_block(self.norm2(x), memory, memory_mask, memory_key_padding_mask, memory_is_causal)
                x = x + self._ff_block(self.norm3(x))
            else:
                x = self.norm1(
                    x[-1:] + self._sa_block(x, tgt_mask, tgt_key_padding_mask, tgt_is_causal,
                                            True))
                x = self.norm2(x + self._mha_block(x, memory, torch.zeros(1, memory.size(0), device=memory.device),
                                                   memory_key_padding_mask, memory_is_causal))
                x = self.norm3(x + self._ff_block(x))
        return x

    # self-attention block
    def _sa_block(self, x: Tensor,
                  attn_mask: Optional[Tensor], key_padding_mask: Optional[Tensor], is_causal: bool = False,
                  cached_pass: bool = False) -> Tensor:
        if cached_pass:
            q = x[-1:]
            attn_mask = torch.zeros(1, x.size(0), device=x.device)
        else:
            q = x
        # print(x)
        x = self.self_attn(q, x, x,
                           attn_mask=attn_mask,
                           key_padding_mask=key_padding_mask,
                           is_causal=is_causal,
                           need_weights=False)[0]
        return self.dropout1(x)


class TransformerDecoderWithCaching(nn.TransformerDecoder):

    def __init__(self, decoder_layer, num_layers, norm=None):
        super(TransformerDecoderWithCaching, self).__init__(decoder_layer, num_layers, norm)
        self.cache = None
        self.mem_cache = None
        self.infer_step = 0

    def train(self, mode: bool = True):
        if mode:
            self.cache = None
        return super(TransformerDecoderWithCaching, self).train(mode)

    def init_cache(self, tgt: Tensor, memory: Tensor,
                   tgt_mask: Optional[Tensor] = None,
                   memory_mask: Optional[Tensor] = None,
                   tgt_key_padding_mask: Optional[Tensor] = None,
                   memory_key_padding_mask: Optional[Tensor] = None
                   ) -> Tensor:
        self.cache = torch.zeros(self.num_layers, *tgt.size(), device=tgt.device)
        self.mem_cache = memory.clone()
        for i, mod in enumerate(self.layers):
            self.cache[i] = tgt
            tgt = mod(tgt, memory, tgt_mask, memory_mask,
                      tgt_key_padding_mask, memory_key_padding_mask,
                      caching_pass=True)

            if self.norm is not None:
                tgt = self.norm(tgt)
            return tgt

    def forward(self, tgt: Tensor, memory: Tensor,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None
                ) -> Tensor:
        if self.training:
            return super(TransformerDecoderWithCaching, self).forward(tgt, memory, tgt_mask, memory_mask,
                                                                      tgt_key_padding_mask, memory_key_padding_mask)
        if self.cache is None:
            return self.init_cache(tgt, memory, tgt_mask, memory_mask,
                                   tgt_key_padding_mask, memory_key_padding_mask)
        output = tgt
        self.mem_cache = self.mem_cache.roll(-1, 0)
        self.mem_cache[-1:] = memory[-1:]
        for i, mod in enumerate(self.layers):
            self.cache[i] = self.cache[i].roll(-1, 0)
            self.cache[i, -1:] = output[-1:]
            output = mod(self.cache[i], self.mem_cache, tgt_mask, memory_mask,
                         tgt_key_padding_mask, memory_key_padding_mask)
        if self.norm is not None:
            output = self.norm(output)
        return output


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
            fast_generate: bool = False,
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
            layer_cls = TransformerDecoderLayerForCaching if fast_generate else TransformerDecoderLayer
            model_cls = TransformerDecoderWithCaching if fast_generate else TransformerDecoder
            layer = layer_cls(d_model=model_dim,
                              nhead=n_heads,
                              dim_feedforward=feedforward_dim,
                              dropout=dropout,
                              norm_first=norm_first,
                              activation=activation)
            self.model = model_cls(layer, num_layers=num_layers,
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
            x += x_upper  # x_upper could be used as memory (?)
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
        fast_generate: bool = False

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
                        else 1),
                    fast_generate=config.fast_generate
                )]
        modules = []
        for in_spec in config.io_spec.inputs:
            if isinstance(in_spec.elem_type, Discrete):
                params = dict(class_size=in_spec.elem_type.size)
                module_type = OneHotConv1dIO
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
            prev_output = tier((tier_input, prev_output))
        fs = self.frame_sizes[-1]
        # :-1 is surprising but right!
        tier_input = tuple(inpt[:, fs0 - fs:-1] for inpt in inputs)
        prev_output = self.tiers[-1]((tier_input, prev_output))
        if not self.training:
            prev_output = prev_output[:, -1:]
        output = tuple(mod(prev_output, **parameters) for mod in self.output_modules)
        return output

    @property
    def config(self) -> "JukeBox.Config":
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
        if not self.config.fast_generate:
            return
        self.outputs = [None] * (len(self.frame_sizes) - 1)
        prompt_length = prompts[0].size(1)
        offset = prompt_length % self.rf
        self.prompt_length = prompt_length - offset
        prev_output = None
        # init the caches with first forward pass
        for tier, fs in zip(self.tiers[:-1], self.frame_sizes[:-1]):
            tier_input = tuple(inpt[:, -self.rf - fs:-fs] for inpt in prompts)
            # print("offset", offset, "fs", fs, "tier_input", tier_input[0].shape[1])
            prev_output = tier((tier_input, prev_output))

    def generate_step(
            self,
            inputs: Tuple[torch.Tensor, ...], *,
            t: int = 0,
            **parameters: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, ...]:
        # TODO: SampleRNN like generate_step()
        # return self._generate_step(inputs, t=t, **parameters)
        if self.config.fast_generate:
            return self._generate_step(inputs, t=t, **parameters)
        return self(inputs, **parameters)

    def _generate_step(
            self,
            inputs: Tuple[torch.Tensor, ...], *,
            t: int = 0,
            **parameters: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, ...]:
        tiers = self.tiers
        outputs = self.outputs
        fs = self.frame_sizes
        for i in range(len(tiers) - 1):
            if t % fs[i] == 0:
                inpt = tuple(inpt[:, -fs[i]:] for inpt in inputs)
                if i == 0:
                    prev_out = None
                else:
                    prev_out = outputs[i - 1][:, (t // fs[i]) % (fs[i - 1] // fs[i])].unsqueeze(1)
                out = tiers[i]((inpt, prev_out))
                outputs[i] = out
        if t < self.prompt_length:
            return tuple()
        inpt = tuple(inpt[:, -fs[-1]:] for inpt in inputs)
        prev_out = outputs[-1][:, (t % fs[-2]) - fs[-2]].unsqueeze(1)
        out = tiers[-1]((inpt, prev_out))[:, -1:]
        outputs = tuple(mod(out, **parameters) for mod in self.output_modules)
        return tuple(out.squeeze(-1) if len(out.size()) > 2 else out for out in outputs)

    def after_generate(self, final_outputs: Tuple[torch.Tensor, ...], batch_index: int) -> None:
        for tier in self.tiers:
            if tier.has_transformer:
                tier.model.cache = None

    @property
    def generate_params(self) -> Set[str]:
        return {"temperature"}
