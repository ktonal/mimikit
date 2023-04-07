import torch
import torch.nn as nn
import dataclasses as dtc
from typing import Optional, Tuple, Dict, List, Iterable, Set
from itertools import accumulate, chain
import operator as opr

from .arm import ARM, NetworkConfig
from ..io_spec import IOSpec
from ..features.item_spec import ItemSpec, Step
from ..modules.misc import Chunk, CausalPad, Transpose
from ..modules.activations import ActivationEnum, ActivationConfig

__all__ = [
    "WNLayer",
    "WaveNet"
]


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
            # TODO
            act_skips: Optional[nn.Module] = None,
            act_residuals: Optional[nn.Module] = None,
            dropout: float = 0.,
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
        self.has_residuals = residuals_dim is not None and (input_dim is None or input_dim == residuals_dim)
        if residuals_dim is None:
            main_inner_dim = main_outer_dim = dims_dilated[0]
            in_dim = main_outer_dim if input_dim is None else input_dim
        else:
            main_outer_dim = residuals_dim
            main_inner_dim = dims_dilated[0]
            if self.has_residuals:
                # cannot be false, just as a reminder:
                assert input_dim is None or input_dim == residuals_dim, "input_dim and residuals_dim must be equal if both are not None"
            in_dim = main_outer_dim if input_dim is None else input_dim

        kwargs_dil = dict(kernel_size=(kernel_size,), dilation=dilation, stride=stride, bias=bias, groups=groups)
        kwargs_1x1 = dict(kernel_size=(1,), stride=stride, bias=bias)

        if self.needs_padding:
            self.pad = CausalPad((0, 0, pad_side * self.cause))

        if self.has_gated_units:
            self.conv_dil = nn.ModuleList([
                nn.Sequential(nn.Conv1d(in_dim, d * 2, **kwargs_dil),
                              Chunk(2, dim=1, sum_outputs=False))
                for d in dims_dilated
            ])
            self.conv_1x1 = nn.ModuleList([
                nn.Sequential(nn.Conv1d(d, main_inner_dim * 2, **kwargs_1x1),
                              Chunk(2, dim=1, sum_outputs=False))
                for d in dims_1x1
            ])
        else:
            self.conv_dil = nn.ModuleList([
                nn.Conv1d(in_dim, d, **kwargs_dil) for d in dims_dilated
            ])
            self.conv_1x1 = nn.ModuleList([
                nn.Conv1d(d, main_inner_dim, **kwargs_1x1) for d in dims_1x1
            ])
        if self.has_skips:
            self.conv_skip = nn.Conv1d(main_inner_dim, skips_dim, **kwargs_1x1)
        if self.has_residuals:
            self.conv_res = nn.Conv1d(main_inner_dim, main_outer_dim, **kwargs_1x1)

        # print("***********************")
        # print(f"in_dim={in_dim} main_inner={main_inner_dim} main_outer={main_outer_dim}")
        # for name, mod in self.named_modules():
        #     if isinstance(mod, nn.Conv1d):
        #         print(f"{name} --> {mod}")
        # print("***********************")

    def forward(self,
                inputs_dilated: Tuple[torch.Tensor, ...],
                inputs_1x1: Tuple[torch.Tensor, ...],
                skips: Optional[torch.Tensor] = None
                ):
        """ TODO: each dilated SHOULD BE an independent path, each summed with all 1x1."""
        if self.needs_padding:
            inputs_dilated = tuple(self.pad(x) for x in inputs_dilated)
        if self.has_gated_units:
            # sum the conditioning features
            cond_f, cond_g = 0, 0
            for conv, x in zip(self.conv_1x1, inputs_1x1):
                if not self.needs_padding:
                    x = self.trim_cause(x)
                y_f, y_g = conv(x)
                cond_f += y_f
                cond_g += y_g
            x_f, x_g = self.conv_dil[0](inputs_dilated[0])
            y = self.act_f(x_f + cond_f) * self.act_g(x_g + cond_g)
        else:
            cond = 0
            for conv, x in zip(self.conv_1x1, inputs_1x1):
                if not self.needs_padding:
                    x = self.trim_cause(x)
                cond += conv(x)
            y = self.conv_dil[0](inputs_dilated[0])
            y = self.act_f(y + cond)

        if self.has_skips:
            if not self.needs_padding:
                skips = self.trim_cause(skips) if skips is not None else skips
            if skips is None:
                skips = self.conv_skip(y)
            else:
                skips = self.conv_skip(y) + skips
        if self.has_residuals:
            # either x has been padded, or y is shorter -> we need to trim x!
            x = self.trim_cause(inputs_dilated[0])
            y = x + self.conv_res(y)
        return y, skips

    def trim_cause(self, x):
        cause, pad_side, kernel_size = self.cause, self.pad_side, self.kernel_size
        # remove dilation for generate_fast
        cs = kernel_size - 1 if x.size(2) == kernel_size and not self.training else cause
        return x[:, :, slice(cs, None) if pad_side >= 0 else slice(None, -cs)]


class WaveNet(ARM, nn.Module):
    @dtc.dataclass
    class Config(NetworkConfig):
        io_spec: IOSpec = None
        kernel_sizes: Tuple[int, ...] = (2,)
        blocks: Tuple[int, ...] = (4,)
        dims_dilated: Tuple[int, ...] = (128,)
        dims_1x1: Tuple[int, ...] = ()
        residuals_dim: Optional[int] = None
        apply_residuals: bool = False
        skips_dim: Optional[int] = None
        groups: int = 1
        act_f: ActivationEnum = "Tanh"
        act_g: Optional[ActivationEnum] = "Sigmoid"
        pad_side: int = 0
        stride: int = 1
        bias: bool = True
        use_fast_generate: bool = False
        tie_io_weights: bool = False

    @classmethod
    def get_layers(cls, config: "WaveNet.Config") -> List[WNLayer]:
        kernel_sizes, dilation = cls.get_kernels_and_dilation(config.kernel_sizes, config.blocks)
        return [
            WNLayer(
                input_dim=config.dims_dilated[0],
                dims_dilated=config.dims_dilated, dims_1x1=config.dims_1x1,
                # no residuals for last layer
                residuals_dim=config.residuals_dim if n != sum(config.blocks) - 1 else None,
                apply_residuals=config.apply_residuals and n != 0,
                skips_dim=config.skips_dim,
                kernel_size=k,
                groups=config.groups,
                act_f=ActivationConfig(str(config.act_f)).get(),
                act_g=ActivationConfig(str(config.act_g)).get() if config.act_g is not None else None,
                pad_side=config.pad_side,
                stride=config.stride, bias=config.bias,
                dilation=d
            )
            for n, (k, d) in enumerate(zip(kernel_sizes, dilation))
        ]

    @classmethod
    def from_config(cls, config: "WaveNet.Config") -> "WaveNet":
        layers = cls.get_layers(config)
        all_dims = [*config.dims_dilated, *config.dims_1x1]
        # set Inner Connection
        input_modules = [spec.module.copy()
                             .set(out_dim=h_dim)
                             .module()
                         for spec, h_dim in zip(config.io_spec.inputs, all_dims)]
        if config.skips_dim is not None:
            all_dims[0] = config.skips_dim
        all_dims = len(config.io_spec.targets) * [all_dims[0]]
        output_module = [spec.module.copy()
                             .set(in_dim=h_dim)
                             .module()
                         for spec, h_dim in zip(config.io_spec.targets, all_dims)]
        if config.tie_io_weights:
            for i, o in zip(input_modules, output_module):
                for name, m in i.named_modules():
                    if isinstance(m, nn.Linear):
                        try:
                            m_o = o.get_submodule(name)
                            m_o.weight = nn.Parameter(m.weight.transpose(0, 1))
                        except AttributeError:
                            continue
        return cls(config=config, layers=layers,
                   input_modules=input_modules, output_modules=output_module)

    def __init__(
            self,
            config: "WaveNet.Config",
            layers: List[WNLayer],
            input_modules: List[nn.Module],
            output_modules: List[nn.Module]
    ):
        super(WaveNet, self).__init__()
        self._config = config
        self.input_modules = nn.ModuleList(input_modules)
        self.transpose = Transpose(1, 2)
        self.layers: Iterable[WNLayer] = nn.ModuleList(layers)
        self.has_skips = config.skips_dim is not None
        self.output_modules = nn.ModuleList(output_modules)
        self.eval_slice = slice(-1, None) if config.pad_side == 1 else slice(0, 1)
        self._gen_context = {}

    def forward(self, inputs: Tuple, **parameters):
        inputs = tuple(self.transpose(mod(x)) for mod, x in zip(self.input_modules, inputs))
        dilated, in_1x1, skips = inputs[0], inputs[1:], None
        for layer in self.layers:
            dilated, skips = layer.forward(
                inputs_dilated=(dilated,), inputs_1x1=in_1x1, skips=skips
            )
            if not layer.needs_padding:
                in_1x1 = tuple(layer.trim_cause(x) for x in in_1x1)
        if self.has_skips:
            y = self.transpose(skips)
        else:
            y = self.transpose(dilated)
        if not self.training:
            y = y[:, self.eval_slice]
        return tuple(mod(y, **parameters) for mod in self.output_modules)

    @classmethod
    def get_kernels_and_dilation(cls, kernel_sizes, blocks):
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
                for start, stop in zip([0] + cum_blocks, cum_blocks):
                    ks = kernel_sizes[start:stop - 1]
                    dilation += list(accumulate([1, *ks], opr.mul))
            elif len(kernel_sizes) == 1:
                k = kernel_sizes[0]
                # all the same
                kernel_sizes = (k for _ in range(sum(blocks)))
                dilation = (k ** i for block in blocks for i in range(block))
            else:
                raise ValueError(f"number of layers and number of kernel sizes not compatible."
                                 f" Got kernel_sizes={kernel_sizes} ; blocks={blocks}")
        return kernel_sizes, dilation

    @property
    def config(self) -> Config:
        return self._config

    @property
    def shift(self) -> int:
        return 1 if self.config.pad_side == 1 else self.rf

    @property
    def rf(self) -> int:
        return sum(layer.cause for layer in self.layers) + 1

    def output_length(self, n_input_steps: int) -> int:
        return n_input_steps if (self.config.pad_side != 0) else (n_input_steps - self.shift + 1)

    @property
    def use_fast_generate(self):
        return self._config.use_fast_generate

    def train_batch(self, item_spec: ItemSpec):
        return tuple(
            spec.to_batch_item(
                item_spec
            )
            for spec in self.config.io_spec.inputs
        ), tuple(
            spec.to_batch_item(
                item_spec + ItemSpec(self.shift, self.output_length(0), unit=Step())
            )
            for spec in self.config.io_spec.targets
        )

    def test_batch(self, item_spec: ItemSpec):
        return self.train_batch(item_spec)

    @property
    def generate_params(self) -> Set[str]:
        return getattr(self.output_modules, "sampling_params", {})

    def before_generate(self, prompts: Tuple[torch.Tensor, ...], batch_index: int) -> None:
        if not self.use_fast_generate:
            return

        layers = [m for m in self.modules() if isinstance(m, WNLayer)]

        qs = {i: None for i in range(len(layers))}
        lyr_slices = {}
        dilations = {}
        pads = {}

        for i, layer in enumerate(layers):
            d, k = layer.dilation, layer.kernel_size
            rf = d * k
            if self.config.pad_side == 0:
                # size = (self.rf - layer.cause)
                # indices = [size - n for n in range(0, rf, d)][::-1]
                indices = [*range(-layer.cause - 1, 0, d)]
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
                    q = qs[0][:-1]
                    if isinstance(q, tuple):  # multiple inputs!
                        q = [*q]
                        for l in range(len(q)):
                            q[l] = q[l].roll(-1, 2)
                            q[l][:, :, -1:] = inpt[l][:, :, -1:]
                            qs[0] = (*q, None)
                    else:
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
                z, skips = outpt[:-1], outpt[-1]
                if i < len(layers) - 1:
                    # set and roll input for next layer
                    q, skp = qs[i + 1][:-1], qs[i + 1][-1]
                    q = [*q]
                    for l in range(len(q)):
                        q[l] = q[l].roll(-1, 2)
                        q[l][:, :, -1:] = z[l][:, :, -1:]
                    if skp is not None:
                        skp = skp.roll(-1, 2)
                        skp[:, :, -1:] = skips[:, :, -1:]

                    qs[i + 1] = (*q, skp)
                return qs.get(i + 1, outpt)

            handles += [layer.register_forward_hook(hook)]
        # save those for later
        self._gen_context = dict(handles=handles, dilations=dilations, pads=pads, layers=layers)

    def generate_step(
            self,
            inputs: Tuple[torch.Tensor, ...], *,
            t: int = 0,
            **parameters: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, ...]:
        return self.forward(inputs, **parameters)

    def after_generate(self, final_outputs: Tuple[torch.Tensor, ...], batch_index: int) -> None:
        if not self.use_fast_generate:
            return
        # reset the layers' parameters
        ctx = self._gen_context
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
        self._gen_context = {}
