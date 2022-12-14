from enum import auto
from typing import Optional, Tuple, Dict, Union, Iterable
import dataclasses as dtc
import torch
import torch.nn as nn

from .mlp import MLP
from .arm import ARMWithHidden, ARMConfig
from ..features.ifeature import Batch
from ..modules.io import ModuleFactory, ZipReduceVariables, ZipMode
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
            inputs: Tuple[Union[T, Tuple[T, ...]], Optional[T]]
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
                h0 = nn.Parameter(torch.randn(self.n_rnn, B, self.hidden_dim).to(x.device))
                return h0
            else:
                return hidden.detach()


class SampleRNN(ARMWithHidden, nn.Module):

    # @dtc.dataclass
    class Config(ARMConfig):
        batch: Batch = Batch([], [])
        frame_sizes: Tuple[int, ...] = (16, 8, 8)
        hidden_dim: int = 256
        rnn_class: RNNType = "lstm"
        n_rnn: int = 1
        rnn_dropout: float = 0.
        rnn_bias: bool = True
        unfold_inputs: bool = False

        inputs: Tuple[ModuleFactory.Config, ...] = (
            ModuleFactory.Config(module_type="framed_linear", module_params=None),
        )
        inputs_mode: ZipMode = "sum"
        learn_temperature: bool = True

    @classmethod
    def from_config(cls, config: "SampleRNN.Config") -> "SampleRNN":
        assert len(config.inputs) == 1, "More than 1 input isn't supported yet"
        q_levels = config.inputs[0].class_size
        tiers = []
        for i, fs in enumerate(config.frame_sizes[:-1]):
            modules = tuple(ModuleFactory(
                class_size=q_levels, module_type=cfg.projection_type,
                hidden_dim=config.hidden_dim, frame_size=fs,
                unfold_step=fs if config.unfold_inputs else None
            )
                            for cfg in config.inputs)
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
        modules = tuple(ModuleFactory(
            class_size=q_levels,
            module_type="fir" if "embedding" not in cfg.projection_type else "fir_embedding",
            hidden_dim=config.hidden_dim, frame_size=config.frame_sizes[-1],
            unfold_step=1 if config.unfold_inputs else None
        )
                        for cfg in config.inputs)
        input_module = ZipReduceVariables(mode=config.inputs_mode, modules=modules)
        tiers += [
            SampleRNNTier(
                input_module=input_module,
                hidden_dim=config.hidden_dim,
                rnn_class="none",
                up_sampling=None
            )]
        output_module = MLP(
            in_dim=config.hidden_dim,
            hidden_dim=config.hidden_dim,
            out_dim=q_levels,
            learn_temperature=config.learn_temperature
        )
        return cls(
            config=config, tiers=tiers, output_module=output_module)

    def __init__(
            self, *,
            config: "SampleRNN.Config",
            tiers: Iterable[nn.Module],
            output_module: nn.Module,
    ):
        super(SampleRNN, self).__init__()
        self._config = config
        self.frame_sizes = config.frame_sizes
        self.tiers = nn.ModuleList(tiers)
        self.output_module = output_module

        # caches for inference
        self.outputs = []
        self.prompt_length = 0

    def forward(self, tiers_input: Tuple[T, ...]):
        prev_output = None
        for tier_input, tier in zip(tiers_input, self.tiers):
            prev_output = tier.forward((tier_input, prev_output))
        output = self.output_module(prev_output)
        return output

    def before_generate(self, prompts: Tuple[torch.Tensor, ...], batch_index: int) -> None:
        self.outputs = [None] * (len(self.frame_sizes) - 1)
        _batch = prompts[0][:, prompts[0].size(1) % self.rf:]

        self.reset_hidden()
        self.prompt_length = len(_batch[0])
        # warm-up
        for t in range(self.rf, self.prompt_length):
            self.generate_step((_batch[:, t - self.rf:t],), t=t)

    def generate_step(
            self,
            inputs: Tuple[torch.Tensor, ...], *,
            t: int = 0,
            **parameters: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, ...]:
        tiers = self.tiers
        outputs = self.outputs
        fs = self.frame_sizes
        temperature = parameters.get("temperature", None)
        # TODO: multiple inputs
        inputs = inputs[0]
        for i in range(len(tiers) - 1):
            if t % fs[i] == 0:
                inpt = inputs[:, -fs[i]:].unsqueeze(1)
                if i == 0:
                    prev_out = None
                else:
                    prev_out = outputs[i - 1][:, (t // fs[i]) % (fs[i - 1] // fs[i])].unsqueeze(1)
                out = tiers[i](((inpt,), prev_out))
                outputs[i] = out
        if t < self.prompt_length:
            return tuple()
        inpt = inputs[:, -fs[-1]:].reshape(-1, 1, fs[-1])
        prev_out = outputs[-1][:, (t % fs[-2]) - fs[-2]].unsqueeze(1)
        out = tiers[-1](((inpt,), prev_out))
        out = self.output_module(out, temperature=temperature)
        return (out.squeeze(-1) if len(out.size()) > 2 else out),

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
    def time_axis(self) -> int:
        return 1

    @property
    def shift(self) -> int:
        return self.frame_sizes[0]

    @property
    def rf(self) -> int:
        return self.frame_sizes[0]

    @property
    def hop_length(self) -> Optional[int]:
        return None

    def output_length(self, n_input_steps: int) -> int:
        return n_input_steps
