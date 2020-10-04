import torch
import torch.nn as nn
import numpy as np
from music_modelling_kit.modules import GatedConv


class Padder(nn.Module):
    def __init__(self, dim, length, learn_values=False):
        super(Padder, self).__init__()
        if learn_values:
            self.p = nn.Parameter(torch.randn(dim, length))
        else:
            self.p = torch.zeros(dim, length)

    def forward(self, x, side):
        self.p = self.p.to(x)
        return torch.cat((x, torch.stack([self.p] * x.size(0)))[::-side], dim=-1)


class FreqLayer(nn.Module):
    def __init__(self, gate_c, residuals_c, skips_c, strict=False, learn_values=False, **kwargs):
        super(FreqLayer, self).__init__()
        self.gate = GatedConv(residuals_c, gate_c, **kwargs)
        kwargs.pop("kernel_size")
        self.residuals = nn.Conv1d(gate_c, residuals_c, 1, **kwargs)
        self.skips = nn.Conv1d(gate_c, skips_c, 1, **kwargs)
        self.pad_y = Padder(residuals_c, self.output_len_diff(strict), learn_values)
        self.pad_skip = Padder(skips_c, self.output_len_diff(strict), learn_values)

    def forward(self, x):
        y = self.gate(x)
        skip = self.skips(y)
        y = self.residuals(y)
        return y, skip

    def output_len_diff(self, strict):
        return (self.gate.kernel - 1) * self.gate.dilation + int(strict)


class FreqBlock(nn.Module):
    def __init__(self, gate_c, residuals_c, skips_c, n_layers, layer_func, **kwargs):
        super(FreqBlock, self).__init__()
        self.block = nn.ModuleList(
            [FreqLayer(gate_c, residuals_c, skips_c,
                       dilation=2 ** i, **kwargs) for i in range(n_layers)])
        self.layer_func = layer_func

    def forward(self, x, skip):
        for i, layer in enumerate(self.block):
            x, skip = self.layer_func(layer, x, skip)
        return x, skip


def accum(x, y, shift=1):
    if shift == 0:
        if x.size(-1) == y.size(-1):
            return x + y
        else:
            raise ValueError("zero shift with size: %i and %i" % (x.size(-1), y.size(-1)))
    side = np.sign(shift)
    aligned_x = slice(*((shift, None)[::side]))
    n_aligned = x.size(-1) - abs(shift)
    aligned_y = slice(*((None, n_aligned * side)[::side]))
    compl_y = slice(*((n_aligned * side, None)[::side]))
    aligned = x[:, :, aligned_x] + y[:, :, aligned_y]
    rest = y[:, :, compl_y]
    return torch.cat((aligned, rest)[::side], dim=-1)


def concat(x, y, shift=1):
    if shift == 0:
        if x.size(-1) != y.size(-1):
            return concat(x, y, - x.size(-1) - y.size(-1))
        return y
    side = np.sign(shift)
    compl_x = slice(*((None, shift)[::side]))
    return torch.cat((x[:, :, compl_x], y)[::side], dim=-1)


def layer_func(layer, x, skip, strict=True,
               accum_outputs=0, concat_outputs=0,
               pad_input=0):
    if pad_input != 0:
        inpt = layer.pad_y(x, pad_input)
    else:
        inpt = x

    y, h = layer(inpt)
    shift = x.size(-1) - y.size(-1) + int(strict)

    if accum_outputs:
        if skip is None:
            skip = torch.zeros_like(h).to(h)
        y, skip = accum(x, y, shift * accum_outputs), accum(skip, h, shift * accum_outputs)

    if concat_outputs:
        if skip is None:
            skip = torch.zeros_like(h).to(h)
        y, skip = concat(x, y, shift * concat_outputs), concat(skip, h, shift * concat_outputs)

    if skip is None or (not concat_outputs and not accum_outputs):
        skip = h

    return y, skip

