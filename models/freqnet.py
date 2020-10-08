from .model_base import Model
from ..modules import GatedLinearInput, AbsLinearOutput, GatedConv

from functools import partial
import torch
import torch.nn as nn
import numpy as np


class Padder(nn.Module):
    def __init__(self, dim, length, learn_padding=False):
        super(Padder, self).__init__()
        if learn_padding:
            self.p = nn.Parameter(torch.randn(dim, length))
        else:
            self.p = torch.zeros(dim, length)

    def forward(self, x, side):
        self.p = self.p.to(x)
        return torch.cat((x, torch.stack([self.p] * x.size(0)))[::-side], dim=-1)


class FreqLayer(nn.Module):
    def __init__(self, gate_c, residuals_c, skips_c, strict=False, learn_padding=False, **kwargs):
        super(FreqLayer, self).__init__()
        self.gate = GatedConv(residuals_c, gate_c, **kwargs)
        kwargs.pop("kernel_size")
        self.residuals = nn.Conv1d(gate_c, residuals_c, 1, **kwargs)
        self.skips = nn.Conv1d(gate_c, skips_c, 1, **kwargs)
        self.pad_y = Padder(residuals_c, self.output_len_diff(strict), learn_padding)
        self.pad_skip = Padder(skips_c, self.output_len_diff(strict), learn_padding)

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


layer_funcs = dict(

    strict_recursive=partial(layer_func, strict=True),
    strict_concat_left=partial(layer_func, strict=True, concat_outputs=1),
    strict_concat_left_residuals_right=partial(layer_func, strict=True, accum_outputs=-1, concat_outputs=1),

    standard_recursive=partial(layer_func, strict=False),
    standard_concat_left=partial(layer_func, strict=False, concat_outputs=1),
    # concat right doesn't make much sense since it means that, counting from the input, lower layers would have to
    # shift the input more than later layers... but whatever...
    standard_concat_right=partial(layer_func, strict=False, concat_outputs=-1),

    residuals_left=partial(layer_func, strict=False, accum_outputs=1),
    residuals_left_concat_right=partial(layer_func, strict=False, accum_outputs=1, concat_outputs=-1),

    residuals_right=partial(layer_func, strict=False, accum_outputs=-1),
    residuals_right_concat_left=partial(layer_func, strict=False, accum_outputs=-1, concat_outputs=1),

    padded_left=partial(layer_func, strict=False, pad_input=1),
    padded_right=partial(layer_func, strict=False, pad_input=-1),

    padded_left_residuals=partial(layer_func, strict=False, pad_input=1, accum_outputs=1),
    padded_right_residuals=partial(layer_func, strict=False, pad_input=-1, accum_outputs=1),

)


# for some reasons, partials don't get pickled in the hparams, unless they are in a tuple...
for k, v in layer_funcs.items():
    layer_funcs[k] = (v,)


class FreqNet(Model):

    @property
    def strict(self):
        return self.lf.keywords.get("strict", False)

    @property
    def shift(self):
        return sum(2 ** i + i * int(self.strict) for i in self.layers)

    @property
    def receptive_field(self):
        return self.shift - sum(i * int(self.strict) for i in self.layers)

    @property
    def concat_side(self):
        return self.lf.keywords.get("concat_outputs", 0)

    def __init__(self, **kwargs):
        super(FreqNet, self).__init__(**kwargs)

        gate_c, residuals_c, skips_c = self.gate_c, self.residuals_c, self.skips_c
        layer_f, strict, learn_padding = self.lf[0], self.strict, self.learn_padding

        # Input Encoder
        self.inpt = GatedLinearInput(1025, residuals_c)

        # Autoregressive Part
        self.blocks = nn.ModuleList([
            FreqBlock(gate_c, residuals_c, skips_c, n_layers, layer_f,
                      strict=strict, learn_padding=learn_padding)
            for n_layers in self.layers
        ])

        # Output Decoder
        self.outpt = AbsLinearOutput(skips_c, 1025)

    def forward(self, x):
        """
        input x has shape (Batch x Time x Dimensionality)
        depending on the layer_function, output has more, less, or as much time steps
        """
        x = self.inpt(x)
        skips = None
        for block in self.blocks:
            x, skips = block(x, skips)
        x = self.outpt(skips)
        return x

    def training_step(self, batch, batch_idx):
        batch, target = batch

        if self.strict:
            # discard the last time steps of the input to get an output of same length as target
            batch = batch[:, :-sum(i for i in self.layers)]

        output = self.forward(batch)

        # since the output can be shorter, longer or equal to target, we have to make the shapes right...
        n_out, n_target = output.size(1), target.size(1)

        if n_out < n_target:
            target = target[:, :n_out]
        elif n_out > n_target:
            if self.concat_side > 0:
                target = torch.cat((batch[:, :self.shift], target), dim=1)
            else:
                output = output[:, :n_target]

        recon = self.loss_fn(output, target)
        self.ep_losses += [recon.item()]
        return {"loss": recon}

    def generation_slices(self, step_length=1):
        if self.concat_side in (0, -1):
            return slice(-self.receptive_field, None), slice(None, step_length)
        return slice(-self.receptive_field, None), slice(self.shift, self.shift+step_length)
