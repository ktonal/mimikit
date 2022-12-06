import torch.nn as nn
from functools import wraps
import dataclasses as dtc


def dataclass_module(module: type):

    module = dtc.dataclass(unsafe_hash=True, repr=False)(module)
    dtc_init = module.__init__

    @wraps(dtc_init)
    def wrapped(self, *args, **kwargs):
        nn.Module.__init__(self)
        dtc_init(self, *args, **kwargs)

    module.__init__ = wrapped

    return module


@dataclass_module
class mymod(nn.Module):
    param: int
    child: nn.Linear

    def __post_init__(self):
        pass

if __name__ == '__main__':
    from typing import Tuple, TypeVar, Generic, NewType, Type

    T = NewType("T", nn.Module)
    V = TypeVar("V", nn.Module, T)


    @dtc.dataclass
    class ModuleArg:
        module: nn.Module


    @dtc.dataclass
    class Promise:
        _value: nn.Module

        def __call__(self) -> nn.Module:
            return self._value


    @dtc.dataclass
    class tst(nn.Module):
        _fc: ModuleArg
        _act: Promise = Promise(nn.Tanh())

        def __post_init__(self):
            nn.Module.__init__(self)
            self.act = self._act()
            self.fc = self._fc.module
            self.f = nn.Sequential(self.fc, self.act)


    mod = tst(ModuleArg("o"), Promise(nn.ReLU()))

    mod2 = tst(ModuleArg(nn.LSTM()), Promise(9879456))
