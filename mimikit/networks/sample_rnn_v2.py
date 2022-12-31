from enum import auto
from typing import Optional, Tuple, Dict, Union, Iterable, List, Set
import dataclasses as dtc
import torch
import torch.nn as nn

from .arm import ARMWithHidden, ARMConfig
from .io_spec import IOSpec, InputSpec, TargetSpec, Objective
from ..features.audio import MuLawSignal
from ..features.ifeature import DiscreteFeature, TimeUnit
from ..modules.io import IOFactory, ZipReduceVariables, ZipMode, MLPParams, LinearParams
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


class SampleRNNTier(nn.Module):

    def __init__(
            self, *,
            input_module: nn.Module = nn.Identity(),
            hidden_dim: int = 256,
            rnn_class: RNNType = "lstm",
            n_rnn: int = 1,
            rnn_dropout: float = 0.,
            rnn_bias: bool = True,
            up_sampling: Optional[int] = None,
    ):
        super(SampleRNNTier, self).__init__()
        self.input_module = input_module
        self.hidden_dim = hidden_dim
        self.rnn_class = rnn_class
        self.n_rnn = n_rnn
        self.rnn_dropout = rnn_dropout
        self.rnn_bias = rnn_bias
        self.up_sampling = up_sampling

        self.hidden = None
        self.has_rnn = rnn_class != "none"
        self.has_up_sampling = up_sampling is not None
        if self.has_rnn:
            module = getattr(nn, rnn_class.upper())
            self.rnn = module(hidden_dim, hidden_dim, num_layers=n_rnn,
                              batch_first=True, dropout=rnn_dropout, bias=rnn_bias)
        if self.has_up_sampling:
            self.up_sampler = LinearResampler(hidden_dim, t_factor=up_sampling, d_factor=1)

    def forward(
            self,
            inputs: Tuple[Tuple[T, ...], Optional[T]]
    ) -> T:
        """x: (batch, n_frames, hidden_dim) ; x_upper: (batch, n_frames, hidden_dim)"""
        x, x_upper = inputs
        x = self.input_module(x)
        if x_upper is not None:
            # print("IN", x.size(), x_upper.size())
            x += x_upper
        if self.has_rnn:
            self.hidden = self._reset_hidden(x, self.hidden)
            x, self.hidden = self.rnn(x, self.hidden)
        if self.has_up_sampling:
            x = self.up_sampler(x)
            # x: (batch, n_frames * up_sampling, hidden_dim)
        return x

    def _reset_hidden(self, x: T, hidden: Optional[Union[Tuple[T, T], T]]) -> Union[Tuple[T, T], T]:
        if self.rnn_class == "lstm":
            if hidden is None or x.size(0) != hidden[0].size(1):
                B = x.size(0)
                h0 = nn.Parameter(torch.randn(self.n_rnn, B, self.hidden_dim).to(x.device))
                c0 = nn.Parameter(torch.randn(self.n_rnn, B, self.hidden_dim).to(x.device))
                return h0, c0
            else:
                return hidden[0].detach(), hidden[1].detach()
        else:
            if hidden is None or x.size(0) != hidden.size(1):
                B = x.size(0)
                h0 = torch.randn(self.n_rnn, B, self.hidden_dim).to(x.device)
                return h0
            else:
                return hidden.detach()


class SampleRNN(ARMWithHidden, nn.Module):
    @dtc.dataclass
    class Config(ARMConfig):
        io_spec: IOSpec = None
        frame_sizes: Tuple[int, ...] = (16, 8, 8)
        hidden_dim: int = 256
        rnn_class: RNNType = "lstm"
        n_rnn: int = 1
        rnn_dropout: float = 0.
        rnn_bias: bool = True
        inputs_mode: ZipMode = "sum"

    @classmethod
    def from_config(cls, config: "SampleRNN.Config") -> "SampleRNN":
        tiers = []
        h_dim = config.hidden_dim
        for i, fs in enumerate(config.frame_sizes[:-1]):
            modules = tuple(in_spec.module.copy()
                            .set(frame_size=fs, hop_length=fs, out_dim=h_dim).get()
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
                    up_sampling=fs // (
                        config.frame_sizes[i + 1]
                        if i < len(config.frame_sizes) - 2
                        else 1)
                )]
        modules = []
        for in_spec in config.io_spec.inputs:
            if "framed" in in_spec.module.module_type:
                modules += [IOFactory(module_type="framed_conv1d")
                                .set(class_size=in_spec.feature.class_size,
                                     frame_size=config.frame_sizes[-1],
                                     hop_length=1, out_dim=h_dim).get()]
            else:
                modules += [IOFactory(module_type="embedding_conv1d")
                                .set(class_size=in_spec.feature.class_size,
                                     frame_size=config.frame_sizes[-1],
                                     hop_length=1, out_dim=h_dim).get()]
        input_module = ZipReduceVariables(mode=config.inputs_mode, modules=modules)
        tiers += [
            SampleRNNTier(
                input_module=input_module,
                hidden_dim=config.hidden_dim,
                rnn_class="none",
                up_sampling=None
            )]
        output_module = [target_spec.module.copy().set(in_dim=h_dim).get()
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

        # caches for inference
        self.outputs = []
        self.prompt_length = 0

    def forward(self, inputs: Tuple):
        # TODO: _forward_one_for_all, _forward_one_for_each
        #  or add transforms before input modules...
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
        _batch = tuple(p[:, p.size(1) % self.rf:] for p in prompts)
        self.reset_hidden()
        self.prompt_length = len(_batch[0])
        # warm-up
        for t in range(self.rf, self.prompt_length):
            self.generate_step(tuple(b[:, t - self.rf:t] for b in _batch), t=t)

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

    def train_batch(self, length=1, unit=TimeUnit.step, downsampling=1):
        return tuple(spec.feature.copy()
                     .batch_item(length + self.frame_sizes[0], unit, downsampling)
                     for spec in self.config.io_spec.inputs
                     ), \
               tuple(spec.feature.copy()
                     .add_shift(self.frame_sizes[0], TimeUnit.sample)
                     .batch_item(length, unit, downsampling)
                     for spec in self.config.io_spec.targets
                     )

    qx_io = IOSpec(
        inputs=(InputSpec(
            feature=MuLawSignal(),
            module=IOFactory(
                module_type="framed_linear",
                params=LinearParams()
            )),),
        targets=(TargetSpec(
            feature=MuLawSignal(),
            module=IOFactory(
                module_type="mlp",
                params=MLPParams(hidden_dim=128, n_hidden_layers=0)
            ),
            objective=Objective("categorical_dist")
        ),))

    @property
    def generate_params(self) -> Set[str]:
        return getattr(self.output_modules, "sampling_params", {})

    # TODO?
    #  - per tier feature (diff q_levels, ...)
    #  - per tier config (hidden_dim, rnn,...)
    #  - chunk_length schedule
