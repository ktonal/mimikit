import torch
import torch.nn as nn
from typing import Optional, Tuple
from itertools import accumulate, chain
import operator as opr
from copy import deepcopy

from ..modules.misc import Chunk, CausalPad
from ..modules.activations import GatingUnit


class WNLayer(nn.Module):

    def __init__(
            self,
            input_dim: Optional[int] = None,
            dims_dilated: Tuple[int] = (128,),
            dims_1x1: Tuple[int] = tuple(),
            residuals_dim: Optional[int] = None,
            apply_residuals: bool = False,
            skips_dim: Optional[int] = None,
            kernel_size: int = 2,
            groups: int = 1,
            act_f: nn.Module = nn.Tanh(),
            act_g: Optional[nn.Module] = nn.Sigmoid(),
            pad_side: int = 1,
            stride: int = 1,
            bias: bool = True,
            dilation: int = 1,
            dropout: float = 0.,  # TODO
    ):
        super(WNLayer, self).__init__()
        self.input_dim = input_dim
        self.dims_dilated = dims_dilated
        self.dims_1x1 = dims_1x1
        self.residuals_dim = residuals_dim
        self.apply_residuals = apply_residuals
        self.skips_dim = skips_dim
        self.kernel_size = kernel_size
        self.groups = groups
        self.act_f = act_f
        self.act_g = act_g
        self.pad_side = pad_side
        self.stride = stride
        self.bias = bias
        self.dilation = dilation

        self.cause = (kernel_size - 1) * dilation
        self.needs_padding = pad_side != 0
        self.has_gated_units = act_g is not None
        self.has_skips = skips_dim is not None
        self.has_residuals = residuals_dim is not None
        if residuals_dim is None:
            main_dim = residuals_dim = dims_dilated[0]
        else:
            main_dim = residuals_dim
            apply_residuals = True

        # todo:
        in_dim = main_dim if input_dim is None else input_dim

        kwargs_dil = dict(kernel_size=(kernel_size,), dilation=dilation, stride=stride, bias=bias, groups=groups)
        kwargs_1x1 = dict(kernel_size=(1,), stride=stride, bias=bias, groups=groups)

        if self.needs_padding:
            self.pad = CausalPad((0, 0, pad_side * self.cause))

        if self.has_gated_units:
            self.conv_dil = nn.ModuleList([
                nn.Sequential(nn.Conv1d(in_dim, d * 2, **kwargs_dil), Chunk(2, dim=1, sum_outputs=False))
                for d in dims_dilated
            ])
            self.conv_1x1 = nn.ModuleList([
                nn.Sequential(nn.Conv1d(d, main_dim * 2, **kwargs_1x1), Chunk(2, dim=1, sum_outputs=False))
                for d in dims_1x1
            ])
            self.activation = GatingUnit()
        else:
            self.conv_dil = nn.ModuleList([
                nn.Conv1d(in_dim, main_dim, **kwargs_dil) for d in dims_dilated
            ])
            self.conv_1x1 = nn.ModuleList([
                nn.Conv1d(d, main_dim, **kwargs_1x1) for d in dims_1x1
            ])
            self.activation = nn.Tanh()
        if self.has_skips:
            self.conv_skip = nn.Conv1d(main_dim, skips_dim, **kwargs_1x1)
        if self.has_residuals:
            self.conv_res = nn.Conv1d(main_dim, main_dim, **kwargs_1x1)

    def forward(self, *,
                inputs_dilated: Tuple[torch.Tensor, ...],
                inputs_1x1: Tuple[torch.Tensor, ...],
                skips: Optional[torch.Tensor] = None
                ):
        """ each dilated is an independent path but is summed with all 1x1."""
        if self.needs_padding:
            inputs_dilated = tuple(self.pad(x) for x in inputs_dilated)
        if self.has_gated_units:
            x_f, x_g = self.conv_dil[0](inputs_dilated[0])
            for conv, x in zip(self.conv_dil[1:], inputs_dilated[1:]):
                y_f, y_g = conv(x)
                x_f += y_f
                x_g += y_g
            for conv, x in zip(self.conv_1x1, inputs_1x1):
                y_f, y_g = conv(x)
                x_f += y_f
                x_g += y_g
            y = self.activation(x_f, x_g)
        else:
            y = sum(conv(x) for conv, x in zip(self.conv_dil, inputs_dilated))
            y += sum(conv(x) for conv, x in zip(self.conv_1x1, inputs_1x1))
            y = self.activation(y)
        if self.has_skips:
            if not self.needs_padding:
                skips = self.trim_cause(skips) if skips is not None else skips
            if skips is None:
                skips = self.conv_skip(y)
            else:
                skips = self.conv_skip(y) + skips
        if self.has_residuals:
            res = self.conv_res(y)


    def trim_cause(self, x):
        cause, pad_side, kernel_size = self.cause, self.pad_side, self.kernel_size
        # remove dilation for generate_fast
        cs = kernel_size - 1 if x.size(2) == kernel_size and not self.training else cause
        return x[:, :, slice(cs, None) if pad_side >= 0 else slice(None, -cs)]
