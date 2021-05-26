import dataclasses as dtc
from typing import Callable

import torch
from numpy.lib.stride_tricks import as_strided as np_as_strided

__all__ = [
    'Getter',
    'AsSlice',
    'AsFramedSlice',
    'Input',
    'Target',
]


@dtc.dataclass
class Getter:
    n_examples: int = None

    def __call__(self, feat_data, item):
        return feat_data[item]

    def __len__(self):
        return self.n_examples


@dtc.dataclass
class AsSlice(Getter):
    shift: int = 0
    length: int = 1
    stride: int = 1

    def __call__(self, feat_data, item):
        i = item * self.stride
        return feat_data[slice(i + self.shift, i + self.shift + self.length)]

    def __len__(self):
        return (self.n_examples - self.shift + self.length + 1) // self.stride


@dtc.dataclass
class AsFramedSlice(Getter):
    shift: int = 0
    length: int = 1
    stride: int = 1
    frame_size: int = 1
    as_strided: bool = False

    def __post_init__(self):
        self.asslice = AsSlice(self.n_examples, self.shift, self.length, self.stride)

    def __call__(self, feat_data, item):
        sliced = self.asslice(feat_data, item)
        if self.as_strided:
            if type(feat_data) is not torch.Tensor:
                itemsize = sliced.dtype.itemsize
                as_strided = lambda arr: np_as_strided(arr,
                                                       shape=(self.length, self.frame_size),
                                                       strides=(itemsize, itemsize))
            else:
                as_strided = lambda tensor: torch.as_strided(tensor,
                                                             size=(self.length, self.frame_size),
                                                             stride=(1, 1))

            with torch.no_grad():
                return as_strided(sliced)
        else:
            return sliced.reshape(-1, self.frame_size)

    def __len__(self):
        return len(self.asslice)


@dtc.dataclass
class Input:
    db_key: str = ''
    getter: Getter = Getter()
    transform: Callable = lambda x: x

    def __len__(self):
        return len(self.getter)


class Target(Input):
    """exactly equivalent to Input, just makes code simpler to read."""
    pass
