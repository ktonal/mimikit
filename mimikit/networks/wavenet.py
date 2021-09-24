import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import Optional, Tuple
from itertools import accumulate, chain
import operator as opr
from functools import partial
from argparse import Namespace

from h5mapper import AsSlice
from .generate_utils import *
from ..modules.homs import *
from ..modules.ops import CausalPad, Transpose

__all__ = [
    'WNLayer',
    "WNBlock",
    'WNNetwork',

]


# some HOM helpers

def Chunk(mod: nn.Module, chunks, dim=-1, sig_in="x"):
    out_vars = ", ".join(["x" + str(i) for i in range(chunks)])
    return hom("Chunk",
               f"{sig_in} -> {out_vars}",
               (mod, f"{sig_in} -> _tmp_"),
               (partial(torch.chunk, chunks=chunks, dim=dim), f"_tmp_ -> {out_vars}")
               )


def GatingUnit(act_f: nn.Module, act_g: nn.Module):
    return hom("GatingUnit",
               "x, y -> z",
               (act_f, "x -> x"),
               (act_g, "y -> y"),
               (opr.mul, "x, y -> z")
               )


def Skips(skipper: nn.Module):
    return hom("Skips",
               "x, skips=None -> skips",
               (skipper, "x -> x"),
               (lambda x, s: x if s is None else x + s, "x, skips -> skips")
               )


class WNLayer(HOM):

    def __init__(
            self,
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
    ):
        init_ctx = locals()
        init_ctx.pop("self")
        init_ctx.pop("__class__")
        self.hp = Namespace(**init_ctx)

        cause = (kernel_size - 1) * dilation
        has_gated_units = act_g is not None
        has_residuals = residuals_dim is not None
        if residuals_dim is None:
            # output dim for the gates
            main_dim = residuals_dim = dims_dilated[0]
        else:
            main_dim = residuals_dim
            apply_residuals = True

        def trim_cause(x):
            # remove dilation for generate_fast
            cs = kernel_size - 1 if x.size(2) == kernel_size and self.training else cause
            return x[:, :, slice(cs, None) if pad_side >= 0 else slice(None, -cs)]

        # as many dilated and 1x1 inputs of any size as we want!
        vars_dil = [f"x{i}" for i in range(1, len(dims_dilated))]
        vars_1x1 = [f"h{i}" for i in range(len(dims_1x1))]
        in_sig = ", ".join(["x"] + vars_dil + vars_1x1 + ["skips=None"])
        out_sig = ", ".join(["y"] + vars_dil + vars_1x1 + ["skips"])

        kwargs_dil = dict(kernel_size=(kernel_size,), dilation=dilation, stride=stride, bias=bias, groups=groups)
        kwargs_1x1 = dict(kernel_size=(1,), stride=stride, bias=bias, groups=groups)

        if has_gated_units:
            conv_dil = [Chunk(nn.Conv1d(d, main_dim * 2, **kwargs_dil), 2, dim=1) for d in dims_dilated]
            conv_1x1 = [Chunk(nn.Conv1d(d, main_dim * 2, **kwargs_1x1), 2, dim=1) for d in dims_1x1]
        else:
            conv_dil = [nn.Conv1d(d, main_dim, **kwargs_dil) for d in dims_dilated]
            conv_1x1 = [nn.Conv1d(d, main_dim, **kwargs_1x1) for d in dims_1x1]

        conv_skip = Skips(nn.Conv1d(main_dim, skips_dim, **kwargs_1x1)) if skips_dim is not None else None
        conv_res = nn.Conv1d(main_dim, main_dim, **kwargs_1x1) if has_residuals else None

        def conv_dil_sig(v):
            return f"{v}in -> {v}_f" + (f", {v}_g" if has_gated_units else "")

        def conv_1x1_sig(v):
            return f"{v} -> {v}_f" + (f", {v}_g" if has_gated_units else "")

        HOM.__init__(
            self,
            f"{in_sig} -> {out_sig}",

            # main input
            *Maybe(pad_side,
                   (CausalPad((0, 0, pad_side * cause)), "x -> xin")),
            *Maybe(not pad_side,
                   (lambda x: x, "x -> xin")),
            (conv_dil[0], conv_dil_sig("x")),

            # other dilated inputs
            *flatten([(*Maybe(pad_side,
                              (CausalPad((0, 0, pad_side * cause)), f"{var} -> {var}in")),
                       (conv_dil[i], conv_dil_sig(var)),
                       Sum(f"x_f, {var}_f -> x_f"),
                       *Maybe(has_gated_units,
                              Sum(f"x_g, {var}_g -> x_g")),
                       *Maybe(not pad_side,  # trim input for next layer
                              (trim_cause, f"{var} -> {var}")
                              ))
                      for i, var in enumerate(vars_dil, 1)]),

            # other 1x1 inputs
            *flatten([(*Maybe(not pad_side,
                              (trim_cause, f"{var} -> {var}")),
                       (conv_1x1[i], conv_1x1_sig(var)),
                       Sum(f"x_f, {var}_f -> x_f"),
                       *Maybe(has_gated_units,
                              Sum(f"x_g, {var}_g -> x_g")),
                       ) for i, var in enumerate(vars_1x1)]),

            # non-linearity
            (GatingUnit(act_f, act_g) if has_gated_units else act_f,
             "x_f" + (", x_g" if has_gated_units else "") + " -> y"),

            # with or without skips
            *Maybe(skips_dim is not None,
                   *Maybe(not pad_side,
                          (lambda s: trim_cause(s) if s is not None else s, "skips -> skips")),
                   (conv_skip, "y, skips -> skips")
                   ),

            # with or without residuals
            *Maybe(has_residuals,
                   (conv_res, "y -> y"),
                   ),

            # even if we don't have residual parameters, we can sum inputs + outputs...
            *Maybe(apply_residuals,
                   *Maybe(not pad_side,
                          (trim_cause, "x -> x")),
                   Sum("x, y -> y")),
            # return statement is in the signature ;)
        )


class WNBlock(HOM):

    def __init__(
            self,
            kernel_sizes: Tuple[int] = (2,),
            blocks: Tuple[int] = (),
            dims_dilated: Tuple[int] = (128,),
            dims_1x1: Tuple[int] = (),
            residuals_dim: Optional[int] = None,
            apply_residuals: bool = False,
            skips_dim: Optional[int] = None,
            groups: int = 1,
            act_f: nn.Module = nn.Tanh(),
            act_g: Optional[nn.Module] = nn.Sigmoid(),
            pad_side: int = 1,
            stride: int = 1,
            bias: bool = True,
    ):
        init_ctx = locals()
        init_ctx.pop("self")
        init_ctx.pop("__class__")
        kernel_sizes, dilation = self.get_kernels_and_dilation(kernel_sizes, blocks)
        layers = [
            WNLayer(dims_dilated=dims_dilated, dims_1x1=dims_1x1, residuals_dim=residuals_dim,
                    apply_residuals=apply_residuals, skips_dim=skips_dim,
                    kernel_size=k,
                    groups=groups, act_f=act_f, act_g=act_g, pad_side=pad_side,
                    stride=stride, bias=bias,
                    dilation=d)
            for k, d in zip(kernel_sizes, dilation)
        ]
        if pad_side == 1:
            shift = 1
        else:
            shift = sum(layer.hp.cause for layer in layers) + 1
        in_sig = ", ".join(["x" + str(i) for i in range(len(dims_dilated) + len(dims_1x1))])
        mid_sig = in_sig + (", skips" if skips_dim is not None else "")
        out_sig = "skips" if skips_dim is not None else "x0"
        # make the steps
        layers = zip(layers, [f"{mid_sig} -> {mid_sig}"] * len(layers))
        # all shapes in and out are Batch x Time x Dim
        super().__init__(
            f"{in_sig}, skips=None -> {out_sig}",
            Map(Transpose(1, 2), f"{mid_sig} -> {mid_sig}"),
            *layers,
            (Transpose(1, 2), f"{out_sig} -> {out_sig}"),
        )
        self.shift = shift
        self.hp = Namespace(**init_ctx)

    @staticmethod
    def get_kernels_and_dilation(kernel_sizes, blocks):
        # figure out the layers dilation.
        # User can pass :
        # - 1 kernel, n blocks
        # - 1 block of kernel & n times the length of this block
        # - n kernels for sum(blocks) == n
        # - n kernels & no blocks = 1 block
        if not blocks:
            # single block from kernels
            dilation = accumulate([1, *kernel_sizes], opr.mul)
        else:
            # repetitions of the same block
            if len(set(blocks)) == 1 and set(blocks).pop() == len(kernel_sizes):
                dilation = accumulate([1, *kernel_sizes], opr.mul)
                dilation = chain(*([dilation] * len(blocks)))
                # broadcast!
                kernel_sizes = chain(*([kernel_sizes] * len(blocks)))
            elif len(kernel_sizes) == sum(blocks):
                cum_blocks = list(accumulate(blocks, opr.add))
                dilation = []
                for start, stop in zip([0] + cum_blocks, cum_blocks[1:]):
                    ks = kernel_sizes[start:stop]
                    dilation += list(accumulate([1, *ks], opr.mul))
            elif len(kernel_sizes) == 1:
                k = kernel_sizes[0]
                # all the same
                kernel_sizes = (k for _ in range(sum(blocks)))
                dilation = (k ** i for block in blocks for i in range(block))
            else:
                raise ValueError(f"number of layers and number of kernel sizes not compatible."
                                 " Got kernel_sizes={kernel_sizes} ; blocks={blocks}")
        return kernel_sizes, dilation

    def with_io(self, input_features, output_features):
        inpt_mod = combine("==", None,
                           *(feat.input_module(d)
                             for feat, d in zip(input_features,
                                                chain(self.hp.dims_dilated, self.hp.dims_1x1))))
        out_d = self.hp.skips_dim if self.hp.skips_dim is not None else self.hp.dims_dilated[0]
        outpt_mod = combine("-<", None,
                            *(feat.output_module(out_d) for feat in output_features))
        in_vars = ", ".join([f"x{i}" for i in range(len(input_features))])
        out_vars = ", ".join([f"y{i}" for i in range(len(output_features))])
        # wrap the steps without wrapping the object
        self[:] = [(inpt_mod, f"{in_vars} -> {in_vars}"),
                   (self.elevate(), f"{in_vars} -> y"),
                   (outpt_mod, f"y -> {out_vars}")]
        self.recompile(f"{in_vars} - > {out_vars}")
        return self

    def getters(self, batch_length, stride=1, hop_length=1, shift_error=0):
        if batch_length - self.shift <= 0:
            raise ValueError(f"batch_length must be greater than the receptive field of this network ({self.shift}).")
        return {
            "inputs": AsSlice(shift=0,
                              length=batch_length * hop_length,
                              stride=stride),
            "targets": AsSlice(shift=(self.shift + shift_error) * hop_length,
                               length=(batch_length if self.hp.pad_input else batch_length - self.shift) * hop_length,
                               stride=stride)
        }


@dataclass(init=True, repr=False, eq=False, frozen=False, unsafe_hash=True)
class WNNetwork(nn.Module):

    @staticmethod
    def predict_(outpt, temp):
        if temp is None:
            return nn.Softmax(dim=-1)(outpt).argmax(dim=-1, keepdims=True)
        else:
            return torch.multinomial(nn.Softmax(dim=-1)(outpt / temp.to(outpt)), 1)

    def generate_(self, prompt, n_steps, temperature=0.5, benchmark=False):
        if self.receptive_field <= 64:
            return self.generate_slow(prompt, n_steps, temperature)
        # prompt is a list but generate fast only accepts one tensor prompt...
        return self.generate_fast(prompt[0], n_steps, temperature)

    def generate_slow(self, prompt, n_steps, temperature=0.5):

        output = prepare_prompt(self.device, prompt, n_steps, at_least_nd=2)
        prior_t = prompt[0].size(1)

        rf = self.receptive_field
        _, out_slc = self.generation_slices()

        for t in self.generate_tqdm(range(prior_t, prior_t + n_steps)):
            inputs = tuple(map(lambda x: x[:, t - rf:t] if x is not None else None, output))
            output[0].data[:, t:t + 1] = self.predict_(self.forward(inputs)[:, out_slc], temperature)
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
        pads = {}
        for mod in self.modules():
            if isinstance(mod, nn.Conv1d) and mod.dilation != (1,):
                dilations[mod] = mod.dilation
                mod.dilation = (1,)
            if isinstance(mod, CausalPad):
                pads[mod] = mod.pad
                mod.pad = (0, 0, 0)

        # cache the indices of the inputs for each layer
        lyr_slices = {}
        for l, layer in enumerate(self.layers):
            d, k = layer.hp.dilation, layer.hp.kernel_size
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
                    i = lyr_slices[l][0 if layer.hp.pad_side < 0 else -1].item()
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
        for mod, pad in pads.items():
            mod.pad = pad

        return output
