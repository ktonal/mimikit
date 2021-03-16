import math
import torch.nn as nn
from dataclasses import dataclass
from ..modules import homs as H, ops as Ops
from typing import Optional


@dataclass(init=True, repr=False, eq=False, frozen=False, unsafe_hash=True)
class WaveNetLayer(nn.Module):
    layer_i: int
    layer_dim: int = 128
    kernel_size: int = 2
    cin_dim: Optional[int] = None
    gin_dim: Optional[int] = None
    dilation = property(lambda self: self.kernel_size ** self.layer_i)
    stride: int = 1
    bias: bool = True
    groups: int = 1

    pad_input: int = 1
    accum_outputs: int = -1
    residuals_dim: Optional[int] = None
    skip_dim: Optional[int] = None

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
        return H.GatedUnit(
            H.AddPaths(
                # core
                nn.Conv1d(self.layer_dim, self.residuals_dim, **self.conv_kwargs),
                # conditioning parameters :
                nn.Conv1d(self.cin_dim, self.residuals_dim, **self.kwargs_1x1) if self.cin_dim else None,
                nn.Conv1d(self.gin_dim, self.residuals_dim, **self.kwargs_1x1) if self.gin_dim else None
            ))

    def residuals_(self):
        return nn.Conv1d(self.residuals_dim, self.layer_dim, **self.kwargs_1x1)

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
            self.residuals_dim = self.layer_dim

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
            shift = self.shift_diff if x.size(2) > self.kernel_size else self.kernel_size - 1
            slc = slice(shift, None) if self.accum_outputs <= 0 else slice(None, -shift)
            padder = nn.Identity()
        else:
            slc = slice(None)
            padder = Ops.CausalPad((0, 0, self.input_padding))
        # print("x", x.size(), "skips", skips.size() if skips is not None else skips)
        y = self.gcu(padder(x), cin, gin)
        if skips is not None and y.size(-1) != skips.size(-1):
            skips = skips[:, :, slc]
            # print("x", x.size(), "skips", skips.size() if skips is not None else skips)
        skips = self.skips(y, skips)
        # if not self.training:
        #     print(y.size(), slc, x[:, :, slc].size(), skips.size())
        y = self.accum(self.residuals(y), x[:, :, slc])
        return y, cin, gin, skips

    def output_length(self, input_length):
        if bool(self.pad_input):
            # no matter what, padding input gives the same output shape
            return input_length
        # output is gonna be less than input
        numerator = input_length - self.dilation * (self.kernel_size - 1) - 1
        return math.floor(1 + numerator / self.stride)


# Finally : define the network class

class WNNetwork(nn.Module):

    def __init__(self,
                 n_layers=(4,),
                 mu=256,
                 n_cin_classes=None,
                 cin_dim=None,
                 n_gin_classes=None,
                 gin_dim=None,
                 layers_dim=128,
                 kernel_size=2,
                 groups=1,
                 accum_outputs=0,
                 pad_input=0,
                 skip_dim=None,
                 residuals_dim=None,
                 head_dim=None,
                 ):
        nn.Module.__init__(self)
        self.inpt = H.Paths(
            nn.Sequential(nn.Embedding(mu, layers_dim), Ops.Transpose(1, 2)),
            nn.Sequential(nn.Embedding(n_cin_classes, cin_dim), Ops.Transpose(1, 2)) if cin_dim else None,
            nn.Sequential(nn.Embedding(n_gin_classes, gin_dim), Ops.Transpose(1, 2)) if gin_dim else None
        )
        self.layers = nn.Sequential(*[
            WaveNetLayer(i,
                         layer_dim=layers_dim,
                         kernel_size=kernel_size,
                         cin_dim=cin_dim,
                         gin_dim=gin_dim,
                         groups=groups,
                         pad_input=pad_input,
                         accum_outputs=accum_outputs,
                         skip_dim=skip_dim,
                         residuals_dim=residuals_dim,
                         ) for block in n_layers for i in range(block)
        ])

        self.outpt = nn.Sequential(
            nn.ReLU(),
            nn.Conv1d(layers_dim if skip_dim is None else skip_dim,
                      layers_dim if head_dim is None else head_dim, kernel_size=1),
            nn.ReLU(),
            nn.Conv1d(layers_dim if head_dim is None else head_dim,
                      mu, kernel_size=1),
            Ops.Transpose(1, 2)
        )

        self.pad_input = pad_input

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

    def forward(self, xi, cin=None, gin=None):
        x, cin, gin = self.inpt(xi, cin, gin)
        y, _, _, skips = self.layers((x, cin, gin, None))
        # print("y", y.size(), "skips", skips.size() if skips is not None else skips)
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
