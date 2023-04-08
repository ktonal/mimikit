from enum import auto
from typing import Optional, Tuple, Dict, Union, Iterable, List, Set
import dataclasses as dtc
import torch
import torch.nn as nn

from .arm import ARMWithHidden, NetworkConfig
from ..io_spec import IOSpec
from ..features.functionals import *
from ..features.item_spec import ItemSpec
from ..modules.io import IOModule, ZipReduceVariables, ZipMode, FramedLinearIO, FramedConv1dIO, EmbeddingConv1d
from ..modules.resamplers import LinearResampler
from ..utils import AutoStrEnum

__all__ = [
    "SampleRNN"
]

T = torch.Tensor


class RNNType(AutoStrEnum):
    lstm = auto()
    rnn = auto()
    gru = auto()
    none = auto()


class H0Init(AutoStrEnum):
    zeros = auto()
    ones = auto()
    randn = auto()


class SampleRNNTier(nn.Module):

    def __init__(
            self, *,
            input_module: nn.Module = nn.Identity(),
            hidden_dim: int = 256,
            rnn_class: RNNType = "lstm",
            n_rnn: int = 1,
            rnn_dropout: float = 0.,
            rnn_bias: bool = True,
            h0_init: H0Init = 'zeros',
            weight_norm: bool = False,
            up_sampling: Optional[int] = None,
    ):
        super(SampleRNNTier, self).__init__()
        self.input_module = input_module
        self.hidden_dim = hidden_dim
        self.rnn_class = rnn_class
        self.n_rnn = n_rnn
        self.rnn_dropout = rnn_dropout
        self.rnn_bias = rnn_bias
        self.h0_init = h0_init
        self.weight_norm = weight_norm
        self.up_sampling = up_sampling

        self.hidden = None
        self.has_rnn = rnn_class != "none"
        self.has_up_sampling = up_sampling is not None
        if self.has_rnn:
            module = getattr(nn, rnn_class.upper())
            self.rnn = module(hidden_dim, hidden_dim, num_layers=n_rnn,
                              batch_first=True, dropout=rnn_dropout, bias=rnn_bias)
            if weight_norm:
                for name in dict(self.rnn.named_parameters()):
                    nn.utils.weight_norm(self.rnn, name)
        if self.has_up_sampling:
            self.up_sampler = LinearResampler(hidden_dim, t_factor=up_sampling, d_factor=1)
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
        """x: (batch, n_frames, hidden_dim) ; x_upper: (batch, n_frames, hidden_dim)"""
        x, x_upper = inputs
        x = self.input_module(x)
        if x_upper is not None:
            x += x_upper
        if self.has_rnn:
            self.hidden = self._reset_hidden(x, self.hidden)
            self.rnn.flatten_parameters()
            x, self.hidden = self.rnn(x, self.hidden)
        if self.has_up_sampling:
            x = self.up_sampler(x)
            # x: (batch, n_frames * up_sampling, hidden_dim)
        return x

    def _reset_hidden(self, x: T, hidden: Optional[Union[Tuple[T, T], T]]) -> Union[Tuple[T, T], T]:
        if self.rnn_class == "lstm":
            if hidden is None or x.size(0) != hidden[0].size(1):
                B = x.size(0)
                h0 = nn.Parameter(self._init_h0(self.n_rnn, B, self.hidden_dim).to(x.device))
                c0 = nn.Parameter(self._init_h0(self.n_rnn, B, self.hidden_dim).to(x.device))
                return h0, c0
            else:
                return hidden[0].detach(), hidden[1].detach()
        else:
            if hidden is None or x.size(0) != hidden.size(1):
                B = x.size(0)
                h0 = self._init_h0(self.n_rnn, B, self.hidden_dim).to(x.device)
                return h0
            else:
                return hidden.detach()

    def _init_h0(self, *dims):
        return getattr(torch, self.h0_init)(*dims)


class SampleRNN(ARMWithHidden, nn.Module):
    @dtc.dataclass
    class Config(NetworkConfig):
        frame_sizes: Tuple[int, ...] = (16, 8, 8)
        hidden_dim: int = 256
        rnn_class: RNNType = "lstm"
        n_rnn: int = 1
        rnn_dropout: float = 0.
        rnn_bias: bool = True
        h0_init: H0Init = "zeros"
        weight_norm: bool = False
        inputs_mode: ZipMode = "sum"
        io_spec: IOSpec = None

    @classmethod
    def from_config(cls, config: "SampleRNN.Config") -> "SampleRNN":
        tiers = []
        h_dim = config.hidden_dim
        for i, fs in enumerate(config.frame_sizes[:-1]):
            modules = tuple(in_spec.module.copy()
                            .set(frame_size=fs, hop_length=fs, out_dim=h_dim).module()
                            for in_spec in config.io_spec.inputs)
            input_module = ZipReduceVariables(mode=config.inputs_mode, modules=modules)
            tiers += [
                SampleRNNTier(
                    input_module=input_module,
                    hidden_dim=config.hidden_dim,
                    rnn_class=config.rnn_class,
                    n_rnn=config.n_rnn,
                    rnn_dropout=config.rnn_dropout,
                    rnn_bias=config.rnn_bias,
                    h0_init=config.h0_init,
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
        input_module = ZipReduceVariables(mode=config.inputs_mode, modules=modules)
        tiers += [
            SampleRNNTier(
                input_module=input_module,
                hidden_dim=config.hidden_dim,
                rnn_class="none",
                up_sampling=None
            )]
        output_module = [target_spec.module.copy().set(in_dim=h_dim).module()
                         for target_spec in config.io_spec.targets]
        return cls(
            config=config, tiers=tiers, output_module=output_module)

    def __init__(
            self, *,
            config: "SampleRNN.Config",
            tiers: Iterable[nn.Module],
            output_module: List[nn.Module],
    ):
        super(SampleRNN, self).__init__()
        self._config = config
        self.frame_sizes = config.frame_sizes
        self.tiers: List[SampleRNNTier] = nn.ModuleList(tiers)
        self.output_modules = nn.ModuleList(output_module)

        if config.weight_norm:
            for module in self.output_modules.modules():
                if isinstance(module, nn.ModuleList) or list(module.children()) != []:
                    continue
                for name in dict(module.named_parameters()):
                    nn.utils.weight_norm(module, name)

        # caches for inference
        self.outputs = []
        self.prompt_length = 0

    def forward(self, inputs: Tuple):
        # TODO: _forward_one_for_all, _forward_one_for_each
        #  or add transforms WITHIN input modules...
        prev_output = None
        fs0 = self.frame_sizes[0]
        for tier, fs in zip(self.tiers[:-1], self.frame_sizes[:-1]):
            tier_input = tuple(inpt[:, fs0 - fs:-fs] for inpt in inputs)
            prev_output = tier.forward((tier_input, prev_output))
        fs = self.frame_sizes[-1]
        # :-1 is surprising but right!
        tier_input = tuple(inpt[:, fs0 - fs:-1] for inpt in inputs)
        prev_output = self.tiers[-1].forward((tier_input, prev_output))
        output = tuple(mod(prev_output) for mod in self.output_modules)
        return output

    def before_generate(self, prompts: Tuple[torch.Tensor, ...], batch_index: int) -> None:
        self.outputs = [None] * (len(self.frame_sizes) - 1)
        self.reset_hidden()
        prompt_length = prompts[0].size(1)
        offset = prompt_length % self.rf
        self.prompt_length = prompt_length - offset
        # warm-up: TODO: make one forward pass (cache outputs and hidden)
        for t in range(self.rf, self.prompt_length):
            self.generate_step(tuple(p[:, t + offset - self.rf:t + offset] for p in prompts), t=t)

    def generate_step(
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
        out = tiers[-1]((inpt, prev_out))
        outputs = tuple(mod(out, **parameters) for mod in self.output_modules)
        return tuple(out.squeeze(-1) if len(out.size()) > 2 else out for out in outputs)

    def after_generate(self, final_outputs: Tuple[torch.Tensor, ...], batch_index: int) -> None:
        self.outputs = []
        self.reset_hidden()

    def reset_hidden(self) -> None:
        for t in self.tiers:
            t.hidden = None

    @property
    def config(self):
        return self._config

    @property
    def rf(self):
        return self.frame_sizes[0]

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

    @property
    def generate_params(self) -> Set[str]:
        return {p for m in self.output_modules for p in getattr(m, "sampling_params", {})}

    # TODO?
    #  - per tier feature (diff q_levels, ...)
    #  - per tier config (hidden_dim, rnn,...)
    #  - time & dim resampling (tier_0.hidden_dim = 512, tier_1.hidden_dim = 256, ...)
    #  - chunk_length / batch_length schedule
