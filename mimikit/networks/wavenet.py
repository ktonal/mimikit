import torch
import torch.nn as nn
from typing import Optional, Tuple
from itertools import accumulate, chain
import operator as opr

from pytorch_lightning.utilities import AttributeDict

from ..modules.homs import *
from ..modules.misc import CausalPad, Transpose, Chunk
from ..loops.generate import *

__all__ = [
    'WNLayer',
    "WNBlock",
]


# some HOM helpers


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
        cause = (kernel_size - 1) * dilation
        self.hp = AttributeDict(init_ctx)
        self.hp.cause = cause
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
            cs = kernel_size - 1 if x.size(2) == kernel_size and not self.training else cause
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
        rf = sum(layer.hp.cause for layer in layers) + 1
        shift = 1 if pad_side == 1 else rf

        in_sig = ", ".join(["x" + str(i)
                            for i in range(len(dims_dilated) + len(dims_1x1))])
        mid_sig = in_sig + ", skips"
        out_sig = "skips" if skips_dim is not None else "x0"
        # make the steps
        layers = zip(layers, [f"{mid_sig} -> {mid_sig}"] * len(layers))

        # helper to output only predictions in eval mode
        class CausalTrimIfEval(nn.Module):
            def forward(self, x):
                if self.training:
                    return x
                return x[:, slice(-1, None) if pad_side == 1 else slice(0, 1)]

        # all shapes in and out are Batch x Time x Dim
        super().__init__(
            f"{in_sig}, skips=None -> {out_sig}",
            Map(Transpose(1, 2), f"{in_sig} -> {in_sig}"),
            *layers,
            (Transpose(1, 2), f"{out_sig} -> {out_sig}"),
            (CausalTrimIfEval(), f"{out_sig} -> {out_sig}")
        )
        self.shift = shift
        self.rf = rf
        self.output_length = (lambda n: n if (self.hp.pad_side != 0) else (n - self.shift + 1))
        self.hp = AttributeDict(init_ctx)

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

    def with_io(self, input_modules, output_modules):
        inpt_mod = combine("==", None, *input_modules)
        in_pos, in_kw = get_input_signature(inpt_mod.forward)
        self_in = inpt_mod.s.split(" -> ")[1]
        outpt_mod = combine("-<", None, *output_modules)
        out_kwargs = get_input_signature(outpt_mod.forward)[1]
        out_in_vars = ', '.join([v.split("=")[0] for v in out_kwargs.split(',')])
        out_vars = ", ".join([f"y{i}" for i in range(len(output_modules))])
        # initialize the graph without re-initializing the object :
        super().__init__(f"{', '.join([arg for arg in (in_pos, in_kw, out_kwargs) if arg])} -> {out_vars}",
                         (inpt_mod, f"{', '.join([arg for arg in (in_pos, in_kw) if arg])} -> {self_in}"),
                         (self.elevate(), f"{self_in} -> y"),
                         (outpt_mod, f"y, {out_in_vars} -> {out_vars}"))
        return self

    use_fast_generate = False

    def before_generate(self, g_loop, batch, batch_idx):
        if not self.use_fast_generate:
            return

        layers = [m for m in self.modules() if isinstance(m, WNLayer)]

        qs = {i: None for i in range(len(layers))}
        lyr_slices = {}
        dilations = {}
        pads = {}

        for i, layer in enumerate(layers):
            d, k = layer.hp.dilation, layer.hp.kernel_size
            rf = d * k
            if self.hp.pad_side == 0:
                size = (self.rf - layer.hp.cause)
                # indices = [size - n for n in range(0, rf, d)][::-1]
                indices = [*range(-layer.hp.cause-1, 0, d)]
            else:
                size = (self.rf - 1)
                indices = [size - n for n in range(0, rf, d)][::-1]
            lyr_slices[i] = torch.tensor(indices).long().to(self.device)

        handles = []

        for i, layer in enumerate(layers):

            def pre_hook(mod, inpt, i=i):
                if qs[i] is None:
                    qs[i] = inpt
                    return inpt
                if i == 0 and qs[i] is not None:
                    q, _ = qs[0]
                    q = q.roll(-1, 2)
                    q[:, :, -1:] = inpt[0][:, :, -1:]
                    qs[0] = (q, None)
                return tuple(x[:, :, lyr_slices[i]]
                             if x is not None else x for x in inpt)

            handles += [layer.register_forward_pre_hook(pre_hook)]

            def hook(module, inpt, outpt, i=i):
                if not getattr(module, '_fast_mode', False):
                    # turn padding and dilation AFTER first pass
                    for mod in module.modules():
                        if isinstance(mod, nn.Conv1d) and mod.dilation != (1,):
                            dilations[mod] = mod.dilation
                            mod.dilation = (1,)
                        if isinstance(mod, CausalPad):
                            pads[mod] = mod.pad
                            mod.pad = (0,) * len(mod.pad)
                    module._fast_mode = True
                    return outpt
                z, skips = outpt
                if i < len(layers) - 1:
                    # set and roll input for next layer
                    q, skp = qs[i + 1]

                    q = q.roll(-1, 2)
                    q[:, :, -1:] = z[:]
                    if skp is not None:
                        skp = skp.roll(-1, 2)
                        skp[:, :, -1:] = skips

                    qs[i + 1] = (q, skp)
                return qs.get(i + 1, outpt)

            handles += [layer.register_forward_hook(hook)]
        # save those for later
        return dict(handles=handles, dilations=dilations, pads=pads, layers=layers)

    def generate_step(self, t, inputs, ctx):
        return self.forward(*inputs)

    def after_generate(self, final_outputs, ctx, batch_idx):
        if not self.use_fast_generate:
            return
        # reset the layers' parameters
        for mod, d in ctx['dilations'].items():
            mod.dilation = d
        for mod, pad in ctx['pads'].items():
            mod.pad = pad
        # remove the hooks
        for handle in ctx['handles']:
            handle.remove()
        # turn off fast_mode
        for layer in ctx['layers']:
            layer._fast_mode = False
        return final_outputs
