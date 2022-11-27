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
    mymod()

