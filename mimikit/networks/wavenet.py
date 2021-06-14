import math
import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import Optional
from itertools import accumulate
import operator

from ..modules import homs as H, ops as Ops
from .generating_net import GeneratingNetwork

__all__ = [
    'WaveNetLayer',
    'WNNetwork'
]


@dataclass(init=True, repr=False, eq=False, frozen=False, unsafe_hash=True)
class WaveNetLayer(nn.Module):
    layer_i: int
    gate_dim: int = 128
    residuals_dim: Optional[int] = None
    skip_dim: Optional[int] = None
    kernel_size: int = 2
    groups: int = 1
    cin_dim: Optional[int] = None
    gin_dim: Optional[int] = None
    gated_units: bool = True
    pad_input: int = 1
    accum_outputs: int = -1

    stride: int = 1
    bias: bool = True

    dilation: int = 1

    shift_diff = property(
        lambda self:
        (self.kernel_size - 1) * self.dilation if self.pad_input == 0 else 0
    )
    input_padding = property(
        lambda self:
        self.pad_input * (self.kernel_size - 1) * self.dilation if self.pad_input else 0
    )
    output_padding = property(
        lambda self:
        - self.accum_outputs * self.shift_diff
    )
    receptive_field = property(
        lambda self:
        self.kernel_size * self.dilation
    )

    @property
    def conv_kwargs(self):
        return dict(kernel_size=self.kernel_size, dilation=self.dilation,
                    stride=self.stride, bias=self.bias, groups=self.groups)

    @property
    def kwargs_1x1(self):
        return dict(kernel_size=1, bias=self.bias, groups=self.groups)

    def gcu_(self):
        mod = H.AddPaths(
            # core
            nn.Conv1d(self.gate_dim, self.residuals_dim, **self.conv_kwargs),
            # conditioning parameters :
            nn.Conv1d(self.cin_dim, self.residuals_dim, **self.conv_kwargs) if self.cin_dim else None,
            nn.Conv1d(self.gin_dim, self.residuals_dim, **self.conv_kwargs) if self.gin_dim else None
        )
        return H.GatedUnit(mod) if self.gated_units else nn.Sequential(mod, nn.Tanh())

    def residuals_(self):
        return nn.Conv1d(self.residuals_dim, self.gate_dim, **self.kwargs_1x1)

    def skips_(self):
        return H.Skips(
            nn.Conv1d(self.residuals_dim, self.skip_dim, **self.kwargs_1x1)
        )

    def accumulator(self):
        return H.AddPaths(nn.Identity(), nn.Identity())

    def __post_init__(self):
        nn.Module.__init__(self)

        with_residuals = self.residuals_dim is not None
        if not with_residuals:
            self.residuals_dim = self.gate_dim

        self.gcu = self.gcu_()

        if with_residuals:
            self.residuals = self.residuals_()
        else:  # keep the signature but just pass through
            self.residuals = nn.Identity()

        if self.skip_dim is not None:
            self.skips = self.skips_()
        else:
            self.skips = lambda y, skp: skp

        if self.accum_outputs:
            self.accum = self.accumulator()
        else:
            self.accum = lambda y, x: y

    def forward(self, inputs):
        x, cin, gin, skips = inputs
        if self.pad_input == 0:
            cause = self.shift_diff if x.size(2) > self.kernel_size else self.kernel_size - 1
            slc = slice(cause, None) if self.accum_outputs <= 0 else slice(None, -cause)
            padder = nn.Identity()
        else:
            slc = slice(None)
            padder = Ops.CausalPad((0, 0, self.input_padding))
        y = self.gcu((padder(x), cin, gin))
        if skips is not None and y.size(-1) != skips.size(-1):
            skips = skips[:, :, slc]
        skips = self.skips(y, skips)
        y = self.accum(self.residuals(y), x[:, :, slc])
        return y, cin[:, :, slc] if cin is not None else cin, gin[:, :, slc] if gin is not None else gin, skips

    def output_length(self, input_length):
        if bool(self.pad_input):
            # no matter what, padding input gives the same output shape
            return input_length
        # output is gonna be less than input
        numerator = input_length - self.dilation * (self.kernel_size - 1) - 1
        return math.floor(1 + numerator / self.stride)


@dataclass(init=True, repr=False, eq=False, frozen=False, unsafe_hash=True)
class WNNetwork(GeneratingNetwork):
    n_layers: tuple = (4,)
    q_levels: int = 256
    n_cin_classes: Optional[int] = None
    cin_dim: Optional[int] = None
    n_gin_classes: Optional[int] = None
    gin_dim: Optional[int] = None
    gate_dim: int = 128
    kernel_size: int = 2
    groups: int = 1
    accum_outputs: int = 0
    pad_input: int = 0
    gated_units: bool = True
    skip_dim: Optional[int] = None
    residuals_dim: Optional[int] = None
    head_dim: Optional[int] = None
    reverse_dilation_order: bool = False

    def inpt_(self):
        return H.Paths(
            nn.Sequential(nn.Embedding(self.q_levels, self.gate_dim), Ops.Transpose(1, 2)),
            nn.Sequential(nn.Embedding(self.n_cin_classes, self.cin_dim),
                          Ops.Transpose(1, 2)) if self.cin_dim else None,
            nn.Sequential(nn.Embedding(self.n_gin_classes, self.gin_dim), Ops.Transpose(1, 2)) if self.gin_dim else None
        )

    def layers_(self):
        # allow custom sequences of kernel_sizes
        if isinstance(self.kernel_size, tuple):
            assert sum(self.n_layers) == len(self.kernel_size), "total number of layers and of kernel sizes must match"
            k_iter = iter(self.kernel_size)
            dilation_iter = iter(accumulate([1, *self.kernel_size], operator.mul))
        else:
            # reverse_dilation_order leads to the connectivity of the FFTNet
            k_iter = iter([self.kernel_size] * sum(self.n_layers))
            dilation_iter = iter([self.kernel_size**(i if not self.reverse_dilation_order else block-1-i)
                                  for block in self.n_layers for i in range(block)])

        return nn.Sequential(*[
            WaveNetLayer(i if not self.reverse_dilation_order else block - 1 - i,
                         gate_dim=self.gate_dim,
                         skip_dim=self.skip_dim,
                         residuals_dim=self.residuals_dim,
                         kernel_size=next(k_iter),
                         cin_dim=self.cin_dim,
                         gin_dim=self.gin_dim,
                         groups=self.groups,
                         gated_units=self.gated_units,
                         pad_input=self.pad_input,
                         accum_outputs=self.accum_outputs,
                         dilation=next(dilation_iter)
                         )
            for block in self.n_layers for i in range(block)
        ])

    def outpt_(self):
        # TODO : Support auxiliary classifier for conditioned networks
        return nn.Sequential(
            nn.ReLU(),
            nn.Conv1d(self.gate_dim if self.skip_dim is None else self.skip_dim,
                      self.gate_dim if self.head_dim is None else self.head_dim,
                      kernel_size=1),
            nn.ReLU(),
            nn.Conv1d(self.gate_dim if self.head_dim is None else self.head_dim,
                      self.q_levels, kernel_size=1),
            Ops.Transpose(1, 2)
        )

    def __post_init__(self):
        nn.Module.__init__(self)
        self.inpt = self.inpt_()
        self.layers = self.layers_()
        self.outpt = self.outpt_()

        rf = 0
        for i, layer in enumerate(self.layers):
            if i == (len(self.layers) - 1):
                rf += layer.receptive_field
            elif self.layers[i + 1].layer_i == 0:
                rf += layer.receptive_field - 1
        self.receptive_field = rf

        if self.pad_input == 1:
            self.shift = 1
        elif self.pad_input == -1:
            self.shift = self.receptive_field
        else:
            self.shift = sum(layer.shift_diff for layer in self.layers) + 1

    def forward(self, inputs):
        x, cin, gin = self.inpt(inputs[0],
                                inputs[1] if len(inputs) > 1 else None,
                                inputs[2] if len(inputs) == 3 else None)
        y, _, _, skips = self.layers((x, cin, gin, None))
        return self.outpt(skips if skips is not None else y)

    def output_shape(self, input_shape):
        return input_shape[0], self.all_output_lengths(input_shape[1])[-1], input_shape[-1]

    def all_output_lengths(self, input_length):
        out_length = input_length
        lengths = []
        for layer in self.layers:
            out_length = layer.output_length(out_length)
            lengths += [out_length]
        return tuple(lengths)

    def generation_slices(self):
        # input is always the last receptive field
        input_slice = slice(-self.receptive_field, None)
        if self.pad_input == 1:
            # then there's only one future time-step : the last output
            output_slice = slice(-1, None)
        else:
            # for all other cases, the first output has the shift of the whole network
            output_slice = slice(None, 1)
        return input_slice, output_slice

    @staticmethod
    def predict_(outpt, temp):
        if temp is None:
            return nn.Softmax(dim=-1)(outpt).argmax(dim=-1, keepdims=True)
        else:
            return torch.multinomial(nn.Softmax(dim=-1)(outpt / temp), 1)

    def generate_(self, prompt, n_steps, temperature=0.5, benchmark=False):
        return self.generate_slow(prompt, n_steps, temperature)

    def generate_slow(self, prompt, n_steps, temperature=0.5):

        output = self.prepare_prompt(prompt, n_steps, at_least_nd=2)
        prior_t = prompt[0].size(1)

        rf = self.receptive_field
        _, out_slc = self.generation_slices()

        for t in self.generate_tqdm(range(prior_t, prior_t + n_steps)):
            inputs = tuple(map(lambda x: x[:, t - rf:t] if x is not None else None, output))
            output[0].data[:, t:t + 1] = self.predict_(self.forward(inputs)[:, out_slc],
                                                       temperature)
        return output[0]

    def generate_fast(self, prompt, n_steps, temperature=0.5):
        # TODO : add support for conditioned networks
        output = self.prepare_prompt(prompt, n_steps, at_least_nd=2)
        prior_t = prompt.size(1) if isinstance(prompt, torch.Tensor) else prompt[0].size(1)

        inpt = output[:, prior_t - self.receptive_field:prior_t]
        z, cin, gin = self.inpt(inpt, None, None)
        qs = [(z.clone(), None)]
        # initialize queues with one full forward pass
        skips = None
        for layer in self.layers:
            z, _, _, skips = layer((z, cin, gin, skips))
            qs += [(z.clone(), skips.clone() if skips is not None else skips)]

        outpt = self.outpt(skips if skips is not None else z)[:, -1:].squeeze()
        outpt = self.predict_(outpt, temperature)
        output.data[:, prior_t:prior_t + 1] = outpt

        qs = {i: q for i, q in enumerate(qs)}

        # disable padding and dilation
        dilations = {}
        for mod in self.modules():
            if isinstance(mod, nn.Conv1d) and mod.dilation != (1,):
                dilations[mod] = mod.dilation
                mod.dilation = (1,)
        for layer in self.layers:
            layer.pad_input = 0

        # cache the indices of the inputs for each layer
        lyr_slices = {}
        for l, layer in enumerate(self.layers):
            d, k = layer.dilation, layer.kernel_size
            rf = d * k
            indices = [qs[l][0].size(2) - 1 - n for n in range(0, rf, d)][::-1]
            lyr_slices[l] = torch.tensor(indices).long().to(self.device)

        for t in self.generate_tqdm(range(prior_t + 1, prior_t + n_steps)):

            x, cin, gin = self.inpt(output[:, t - 1:t], None, None)
            q, _ = qs[0]
            q = q.roll(-1, 2)
            q[:, :, -1:] = x
            qs[0] = (q, None)

            for l, layer in enumerate(self.layers):
                z, skips = qs[l]
                zi = torch.index_select(z, 2, lyr_slices[l])
                if skips is not None:
                    # we only need one skip : the first or the last of the kernel's indices
                    i = lyr_slices[l][0 if layer.accum_outputs > 0 else -1].item()
                    skips = skips[:, :, i:i + 1]

                z, _, _, skips = layer((zi, cin, gin, skips))

                if l < len(self.layers) - 1:
                    q, skp = qs[l + 1]

                    q = q.roll(-1, 2)
                    q[:, :, -1:] = z
                    if skp is not None:
                        skp = skp.roll(-1, 2)
                        skp[:, :, -1:] = skips

                    qs[l + 1] = (q, skp)
                else:
                    y, skips = z, skips

            outpt = self.outpt(skips if skips is not None else y).squeeze()
            outpt = self.predict_(outpt, temperature)
            output.data[:, t:t + 1] = outpt

        # reset the layers' parameters
        for mod, d in dilations.items():
            mod.dilation = d
        for layer in self.layers:
            layer.pad_input = self.pad_input

        return output
