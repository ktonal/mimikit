import math
import torch.nn as nn
from dataclasses import dataclass
from ..modules import homs as H, ops as Ops


@dataclass(init=True, repr=False, eq=False, frozen=False, unsafe_hash=True)
class WaveNetLayer(nn.Module):
    layer_i: int
    layer_dim: int = 128
    kernel_size: int = 2
    cin_dim: int = None
    gin_dim: int = None
    dilation = property(lambda self: self.kernel_size ** self.layer_i)
    stride: int = 1
    bias: bool = True
    groups: int = 1

    pad_input: int = 1
    accum_outputs: int = -1
    strict: int = False
    with_residual_conv: bool = True
    with_skip_conv: bool = True

    shift_diff = property(
        lambda self:
        int(self.strict) + (self.kernel_size - 1) * self.dilation if self.pad_input == 0 else int(self.strict)
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
                nn.Sequential(Ops.CausalPad((0, 0, self.input_padding)),
                              nn.Conv1d(self.layer_dim, self.layer_dim, **self.conv_kwargs), ),
                # conditioning parameters :
                nn.Conv1d(self.cin_dim, self.layer_dim, **self.kwargs_1x1) if self.cin_dim else None,
                nn.Conv1d(self.gin_dim, self.layer_dim, **self.kwargs_1x1) if self.gin_dim else None
            ))

    def residuals_(self):
        return nn.Conv1d(self.layer_dim, self.layer_dim, **self.kwargs_1x1)

    def skips_(self):
        return H.Skips(
            nn.Sequential(
                nn.Conv1d(self.layer_dim, self.layer_dim, **self.kwargs_1x1),
                Ops.CausalPad((0, 0, self.output_padding))  # pad the output before taking the sum
            ))

    def accumulator(self):
        return H.AddPaths(nn.Identity(),
                          nn.Identity())

    def __post_init__(self):
        nn.Module.__init__(self)
        self.gcu = self.gcu_()

        if self.with_residual_conv:
            self.residuals = self.residuals_()
        else:  # keep the signature but just pass through
            self.residuals = nn.Identity()

        if self.with_skip_conv:
            self.skips = self.skips_()
        else:
            self.skips = lambda y, skp: skp

        if self.accum_outputs:
            self.accum = self.accumulator()
        else:
            self.accum = lambda y, x: y

    def forward(self, inputs):
        x, cin, gin, skips = inputs
        y = self.gcu(x, cin, gin)
        skips = self.skips(y, skips)
        y = self.accum(self.residuals(y), x)
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
                 strict=False,
                 accum_outputs=0,
                 pad_input=0,
                 with_skip_conv=False,
                 with_residual_conv=False,
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
                         strict=strict,
                         with_skip_conv=with_skip_conv,
                         with_residual_conv=with_residual_conv,
                         ) for block in n_layers for i in range(block)
        ])

        self.outpt = nn.Sequential(
            nn.ReLU(),
            nn.Conv1d(layers_dim, layers_dim, kernel_size=1),
            nn.ReLU(),
            nn.Conv1d(layers_dim, mu, kernel_size=1),
            Ops.Transpose(1, 2)
        )

        self.strict = strict
        self.pad_input = pad_input

        # properties of the network used for making batches and generation:

        self.all_rel_shifts = tuple(layer.shift_diff for layer in self.layers)

        rf = 0
        for i, layer in enumerate(self.layers):
            if i == (len(self.layers) - 1):
                rf += layer.receptive_field
            elif self.layers[i + 1].layer_i == 0:
                rf += layer.receptive_field - 1
        self.receptive_field = rf

        if not self.strict and self.pad_input == 1:
            self.shift = 1
        elif self.pad_input == -1:
            self.shift = self.receptive_field
        elif self.strict and self.pad_input == 1:
            self.shift = len(self.layers)
        else:
            self.shift = sum(self.all_rel_shifts) + int(not self.strict)

    def forward(self, xi, cin=None, gin=None):
        x, cin, gin = self.inpt(xi, cin, gin)
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
