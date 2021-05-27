import torch.nn.functional as F
from torch import nn

__all__ = [
    'Abs',
    'Transpose',
    'CausalPad',
    'Slice',
    'Clone'
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


class Slice(nn.Module):

    def __init__(self, slc):
        super(Slice, self).__init__()
        self.slc = slc

    def forward(self, x):
        return x[self.slc].contiguous()


class Clone(nn.Module):
    def forward(self, x):
        return x.clone()