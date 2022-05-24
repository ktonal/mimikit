from functools import partial
import torch.nn.functional as F
from torch import nn
import torch

from .homs import hom, HOM, Maybe, Sum

__all__ = [
    'Abs',
    'Chunk',
    'Flatten',
    'Transpose',
    'CausalPad',
    'ScaledActivation',
    'ScaledSigmoid',
    'ScaledTanh',
    'ScaledAbs'
]


class Abs(nn.Module):

    def forward(self, x):
        return x.abs_()


class Transpose(nn.Module):

    def __init__(self, dim1, dim2):
        super(Transpose, self).__init__()
        self.dims = (dim1, dim2)

    def forward(self, *args):
        if len(args) > 1:
            return tuple(x.transpose(*self.dims).contiguous() if x is not None else x for x in args)
        return args[0].transpose(*self.dims).contiguous() if args[0] is not None else None


class CausalPad(nn.Module):

    @staticmethod
    def int_to_lr(i):
        return (i, 0) if i >= 0 else (0, abs(i))

    def __init__(self, pad, **kwargs):  # TODO: learn=True
        super().__init__()
        # for whatever reason torch decided dimensions should be in the reversed order...
        self.pad = tuple(i for p in reversed(pad) for i in self.int_to_lr(p))
        self.kwargs = kwargs

    def forward(self, x):
        return F.pad(x, self.pad, **self.kwargs)


class ScaledActivation(nn.Module):
    def __init__(self, activation, dim, with_range=True):
        super(ScaledActivation, self).__init__()
        self.activation = activation
        # self.scales = nn.Parameter(torch.rand(dim, ) * 100, )
        self.scales = nn.Linear(dim, dim)
        self.dim = dim

    def forward(self, x):
        # s = self.scales.to(x).view(*(d if d == self.dim else 1 for d in x.size()))
        # return self.activation(self.rg * x / self.scales) * self.scales
        return self.activation(x) / self.activation(self.scales(x))


class ScaledSigmoid(ScaledActivation):
    def __init__(self, dim, with_range=True):
        super(ScaledSigmoid, self).__init__(nn.Sigmoid(), dim, with_range)


class ScaledTanh(ScaledActivation):
    def __init__(self, dim, with_range=True):
        super(ScaledTanh, self).__init__(nn.Tanh(), dim, with_range)


class ScaledAbs(ScaledActivation):
    def __init__(self, dim, with_range=True):
        super(ScaledAbs, self).__init__(Abs(), dim, with_range)


class Chunk(HOM):
    def __init__(self, mod: nn.Module, chunks, dim=-1, sig_in="x", sum_out=False):
        out_vars = ", ".join(["x" + str(i) for i in range(chunks)])
        super(Chunk, self).__init__(
            f"{sig_in} -> {out_vars if not sum_out else 'out'}",
            (mod, f"{sig_in} -> _tmp_"),
            (partial(torch.chunk, chunks=chunks, dim=dim), f"_tmp_ -> {out_vars}"),
            *Maybe(sum_out,
                   Sum(f"{out_vars} -> out"))
        )


class Flatten(HOM):
    """flatten `n_dims` dimensions of a tensor counting from the last"""

    def __init__(self, n_dims):
        super(Flatten, self).__init__(
            "x -> x",
            (lambda x: x.view(*x.shape[:-n_dims], -1), 'x -> x')
        )
