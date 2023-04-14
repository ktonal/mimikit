from typing import Tuple

import torch.nn.functional as F
from torch import nn
import torch


__all__ = [
    'Chunk',
    'Flatten',
    'Transpose',
    'CausalPad',
    'Unsqueeze',
    'Unfold',
    'ShapeWrap'
]


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

    def __init__(self, pad, **kwargs):
        super().__init__()
        # for whatever reason torch decided dimensions should be in the reversed order...
        self.pad = tuple(i for p in reversed(pad) for i in self.int_to_lr(p))
        self.kwargs = kwargs

    def forward(self, x):
        return F.pad(x, self.pad, **self.kwargs)


class Chunk(nn.Module):
    def __init__(
            self,
            chunks: int,
            dim: int = -1,
            sum_outputs: bool = False
    ):
        super(Chunk, self).__init__()
        self.chunks = chunks
        self.dim = dim
        self.sum_outputs = sum_outputs

    def forward(self, x):
        x = torch.chunk(x, chunks=self.chunks, dim=self.dim)
        if self.sum_outputs:
            return sum(x)
        return x


class Flatten(nn.Module):
    """flatten `n_dims` dimensions of a tensor (firsts n if n_dims > 0, else n lasts)"""

    def __init__(self, n_dims):
        super(Flatten, self).__init__()
        self.n_dims = n_dims

    def forward(self, x):
        if self.n_dims < 0:
            return x.reshape(*x.shape[:self.n_dims], -1)
        else:
            return x.reshape(-1, *x.shape[self.n_dims:])


class Unsqueeze(nn.Module):

    def __init__(self, dim):
        super(Unsqueeze, self).__init__()
        self.dim = dim

    def forward(self, x):
        return x.unsqueeze(self.dim)


class Unfold(nn.Module):

    def __init__(self, dim=-1, size=1, step=1):
        super(Unfold, self).__init__()
        self.params = (dim, size, step)

    def forward(self, x):
        return x.unfold(*self.params)


class ShapeWrap(nn.Module):
    def __init__(self, module: nn.Module,
                 in_view: Tuple[int, ...],
                 out_view: Tuple[int, ...]):
        super(ShapeWrap, self).__init__()
        self.m = module
        self.in_view = in_view
        self.out_view = out_view

    def forward(self, x):
        B = x.size(0)
        x = self.m(x.view(*self.in_view)).squeeze()
        return x.view(B, *self.out_view)
