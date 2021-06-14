from dataclasses import dataclass
from functools import wraps

import torch.nn as nn
import torch
from copy import deepcopy

__all__ = [
    'HOM',
    'HOMSequential',
    'SequentialAdd',
    'SequentialMul',
    'SequentialCollect',
    'Paths',
    'FPaths',
    'AddPaths',
    'MulPaths',
    'block',
    'Tiers',
    'Skips',
    'GatedUnit'
]


# *******************************************************************
# ************************ HOM **************************************
# *******************************************************************


class NotAModule(nn.Module):
    """
    the nn.Module equivalent of None
    """
    pass


class HOM(nn.Sequential, nn.ModuleList):

    @staticmethod
    def replace_none(module):
        return module if module is not None else NotAModule()

    @staticmethod
    def output_or_none(mod, x):
        if type(mod) is NotAModule or x is None:
            return None
        return mod(x)

    @staticmethod
    def flat_inputs(*inputs):
        return inputs[0] if len(inputs) == 1 and isinstance(inputs[0], tuple) else inputs

    def __init__(self, *modules):
        super().__init__(*(self.replace_none(m) for m in modules))

    def forward(self, *inputs):
        raise NotImplementedError


# *******************************************************************
# ******************* HOMSequential *********************************
# *******************************************************************


class HOMSequential(HOM):
    ops_ = None

    def __init__(self, *modules):
        nn.Sequential.__init__(self, *[self.replace_none(m) for m in modules])

    def forward(self, input):
        out = None
        for mod in self:
            out = self.output_or_none(mod, input)
            if out is not None:
                if self.ops_ is not None:
                    self.ops_(input, out)
                else:
                    input = out
        return out


class SequentialAdd(HOMSequential):
    @property
    def ops_(self):
        return torch.Tensor.add_


class SequentialMul(HOMSequential):
    @property
    def ops_(self):
        return torch.Tensor.mul_


class SequentialCollect(HOMSequential):

    def forward(self, input):
        collected = []
        for mod in self:
            out = self.output_or_none(mod, input)
            if out is not None:
                if self.ops_ is not None:
                    self.ops_(input, out)
                else:
                    input = out
            collected += [out]
        return tuple(collected)


# *******************************************************************
# ************************* Paths ***********************************
# *******************************************************************


class Paths(HOM):

    def __init__(self, *modules):
        super(Paths, self).__init__(*modules)

    def forward(self, *inputs):
        inputs = self.flat_inputs(*inputs)
        return tuple(self.output_or_none(mod, x) for mod, x in zip(self, inputs))


class FPaths(Paths):
    @property
    def ops(self):
        return None

    def forward(self, *inputs):
        # flatten ((x, x, ....), ) to (x, x, ...)
        inputs = self.flat_inputs(*inputs)
        inputs = (x for x in inputs)
        modules = (m for m in self)
        out = None
        # if the firsts modules were Nones
        while out is None:
            out = self.output_or_none(next(modules), (next(inputs)))
        # now accum the outputs
        for mod, x in zip(modules, inputs):
            output = self.output_or_none(mod, x)
            if output is not None:
                # all output shapes have to be the same!...
                self.ops(out, output)
        return out


class AddPaths(FPaths):
    @property
    def ops(self):
        return torch.Tensor.add


class MulPaths(FPaths):
    @property
    def ops(self):
        return torch.Tensor.mul


# *******************************************************************
# ************************ block ************************************
# *******************************************************************


def block(name, layer_func, container_cls=HOMSequential):
    def init(self, n_layers, *args, **kwargs):
        container_cls.__init__(self, *(layer_func(i, *args, **kwargs)
                                       for sub_block in n_layers for i in range(sub_block)))

    return type(name, (container_cls,), {"__init__": init})


class Tiers(HOMSequential):

    def forward(self, tiers_inputs):
        output = None
        for tier, inpt in zip(self, tiers_inputs):
            output = tier(inpt, output)
        return output


# *******************************************************************
# ************************* MISC ************************************
# *******************************************************************

class Skips(nn.Module):
    def __init__(self, skips):
        super().__init__()
        self.skp = skips

    def forward(self, x, skips=None):
        h = self.skp(x)
        if skips is None:
            skips = torch.zeros_like(h, device=h.device)
        skips.add_(h)
        return skips


class GatedUnit(MulPaths):

    def __new__(cls, module):
        class GU(MulPaths):
            def forward(self, inputs):
                return super(MulPaths, self).forward(inputs, inputs)

        return GU(nn.Sequential(module, nn.Tanh()),
                  nn.Sequential(deepcopy(module), nn.Sigmoid()))


def propped(*bases):
    def wrapper(cls):
        cls = dataclass(cls, init=True, repr=False, eq=False, frozen=False)

        @wraps(cls.__init__)
        def new_init(self, *args, **kwargs):
            for base in bases:
                base.__init__(self)
            cls.__init__(self, *args, **kwargs)

        cls.__init__ = new_init
        return cls

    return wrapper
