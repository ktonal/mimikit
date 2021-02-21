import math
import torch.nn as nn
from dataclasses import dataclass

from ..modules import homs as H, ops as Ops


# First : Inputs and Outputs classes

class WNInputs(nn.Sequential):
    def __init__(self, mu, emb_dim,
                 n_cin_classes=None, cin_dim=None,
                 n_gin_classes=None, gin_dim=None):
        super(WNInputs, self).__init__(
            H.Paths(
                nn.Embedding(mu, emb_dim),
                nn.Embedding(n_cin_classes, cin_dim) if cin_dim else None,
                nn.Embedding(n_gin_classes, gin_dim) if gin_dim else None),
            Ops.Transpose(1, 2)
        )

    def forward(self, xi, cin=None, gin=None):
        return self(xi, cin, gin)


class WNOutputs(nn.Sequential):
    def __init__(self, input_dim, mid_dim, output_dim):
        super(WNOutputs, self).__init__(
            nn.ReLU(inplace=True),
            nn.Conv1d(input_dim, mid_dim, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(mid_dim, output_dim, kernel_size=1),
            Ops.Transpose(1, 2)
        )

    def forward(self, skips):
        out = self(skips)
        act = nn.LogSoftmax(dim=-1) if self.training else nn.Softmax(dim=-1)
        return act(out)


# Then : to facilitate experimentation, we split the layer into 2 classes.
# WNLayerProps computes and holds the properties of the layer (kernel_size, dilation, padding...)
# WaveNetLayer builds the modules from the props
# changing the logic that "renders" the network amounts to subclassing the props
# and subclassing the layer with the hew props. Aka :
# ``class MyWNLayer(WaveNetLayer): props_cls = MyLayerProps``

@dataclass
class WNLayerProps:
    layer_i: int
    kernel_size: int = 2
    dilation: int = 1
    stride: int = 1
    bias: bool = True
    groups: int = 1

    pad_input: int = 0
    accum_outputs: int = 0
    strict: int = False

    input_padding: int = None
    output_padding: int = None
    shift_diff: int = None
    receptive_field: int = None

    sides = {1: 1, -1: -1, 0: 0, "left": 1, "right": -1}

    def __post_init__(self):

        self.pad_input = self.sides[self.pad_input]
        self.accum_outputs = - self.sides[self.accum_outputs]

        self.dilation = self.kernel_size ** self.layer_i
        self.receptive_field = self.kernel_size * self.dilation

        self.shift = self.receptive_field + self.layer_i * int(self.strict)
        self.shift_diff = (int(self.strict) + (self.kernel_size - 1) * self.dilation) \
            if self.pad_input == 0 else int(self.strict)

        if self.pad_input:
            self.input_padding = self.pad_input * (self.kernel_size - 1) * self.dilation
        else:
            self.input_padding = 0

        if self.accum_outputs:
            self.output_padding = - self.accum_outputs * self.shift_diff

    def output_length(self, input_length):
        if abs(self.pad_input):
            # no matter what, padding input gives the same output shape
            return input_length
        # output is gonna be less than input
        numerator = input_length - self.dilation * (self.kernel_size - 1) - 1
        return math.floor(1 + numerator / self.stride)

    @property
    def conv_kwargs(self):
        return dict(kernel_size=self.kernel_size, dilation=self.dilation,
                    stride=self.stride, bias=self.bias, groups=self.groups)

    @property
    def kwargs_1x1(self):
        return dict(kernel_size=1, bias=self.bias, groups=self.groups)


class WaveNetLayer(nn.Module):

    @property
    def props_cls(self):
        return WNLayerProps

    def __init__(self,
                 layer_index,
                 input_dim=512,
                 layer_dim=512,
                 groups=1,
                 strict=False,
                 kernel_size=2,
                 accum_outputs=0,
                 concat_outputs=0,
                 pad_input=0,
                 learn_padding=False,
                 with_skip_conv=False,
                 with_residual_conv=False,
                 local_cond_dim=None,
                 global_cond_dim=None):
        super(WaveNetLayer, self).__init__()

        props = self.props_cls(layer_index, kernel_size, groups=groups,
                               strict=strict, accum_outputs=accum_outputs, pad_input=pad_input)

        self.gcu = H.GatedUnit(
            H.AddPaths(
                # core
                nn.Conv1d(input_dim, layer_dim, kernel_size, **props.conv_kwargs),
                # conditioning parameters :
                nn.Conv1d(local_cond_dim, layer_dim, **props.kwargs_1x1) if local_cond_dim else None,
                nn.Conv1d(global_cond_dim, layer_dim, **props.kwargs_1x1) if global_cond_dim else None
            ))

        if with_residual_conv:
            self.residuals = nn.Conv1d(layer_dim, layer_dim, **props.kwargs_1x1)
        else:  # keep the signature but just pass through
            self.residuals = nn.Identity()

        if with_skip_conv:
            self.skips = H.Skips(
                nn.Sequential(
                    nn.Conv1d(layer_dim, layer_dim, **props.kwargs_1x1),
                    Ops.SidedPad((0, 0, props.output_padding))  # pad the output before taking the sum
                ))
        else:
            self.skips = lambda y, skp: skp

        if accum_outputs:
            self.accum = H.AddPaths(Ops.SidedPad((0, 0, props.output_padding)), nn.Identity())
        else:
            self.accum = lambda y, x: y

    def forward(self, x, cin=None, gin=None, skips=None):
        y = self.gcu(x, cin, gin)
        skips = self.skips(y, skips)
        y = self.accum(self.residuals(y), x)
        return y, skips if skips is not None else y


# Make the block class :

WNBlock = H.block("WNBlock", WaveNetLayer, nn.Sequential)


# Finally : define the network class

class WNNetwork(nn.Module):

    def __init__(self,
                 n_layers=(4,),
                 mu=255,
                 n_cin_classes=None,
                 cin_dim=None,
                 n_gin_classes=None,
                 gin_dim=None,
                 layers_dim=128,
                 kernel_size=2,
                 groups=1,
                 strict=False,
                 accum_outputs=0,
                 concat_outputs=0,
                 pad_input=0,
                 learn_padding=False,
                 with_skip_conv=False,
                 with_residual_conv=False,
                 ):
        super(WNNetwork, self).__init__()
        self.inpt = WNInputs(mu, layers_dim, n_cin_classes, cin_dim, n_gin_classes, gin_dim)
        self.layers = WNBlock(n_layers, layers_dim, layers_dim, groups, strict, kernel_size,
                              accum_outputs, concat_outputs, pad_input, learn_padding,
                              with_skip_conv, with_residual_conv,
                              cin_dim, gin_dim
                              )
        self.outpt = WNOutputs(layers_dim, layers_dim, mu)

    def forward(self, xi, cin=None, gin=None):
        x, cin, gin = self.inpt(xi, cin, gin)
        _, skips = self.layers(x, cin, gin, skips=None)
        return self.outpt(skips)

