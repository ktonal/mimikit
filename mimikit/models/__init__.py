import pytorch_lightning as pl
from inspect import signature
import dataclasses as dtc


def _filter_cls_kwargs(cls, kwargs):
    valids = signature(cls.__init__).parameters.keys()
    return {k: v for k, v in kwargs.items() if k in valids}


def model(cls):
    """

    Parameters
    ----------
    cls

    Returns
    -------

    """
    bases = cls.__bases__

    # get the __init__ signatures of the bases
    sigs = [dict(signature(d.__init__).parameters) for d in bases
            if d is not pl.LightningModule]

    # collect positional arguments for validation
    posits = {k: v for s in sigs for k, v in s.items() if v.default is v.empty}
    if any(k for k in posits.keys() if k != "self"):
        raise ValueError("All dependencies should have kwargs only")

    # collect keywords arguments
    kws = {k: v for s in sigs for k, v in s.items() if v.default is not v.empty}

    # build fields for make_dataclass
    fields = [(k, type(v.default), dtc.field(default=v.default)) for k, v in kws.items()]

    # method to initialize all the bases
    def __post_init__(self):
        super(pl.LightningModule, self).__init__()
        for b in self.__parts__:
            if hasattr(b, '__post_init__'):
                # avoid recursion if b is a dataclass already
                b.__post_init__(self)
            else:
                b.__init__(self, **_filter_cls_kwargs(b, self.__dict__))
        self._set_hparams({k: getattr(self, k) for k in self.__args__})

    # namespace of the class
    ns = {
        "__post_init__": __post_init__,
        '__args__': tuple(kws.keys()),
    }
    return dtc.make_dataclass(cls.__qualname__,
                              fields,
                              bases=(cls,
                                     *bases,
                                     *((pl.LightningModule,) if pl.LightningModule not in bases else tuple())),
                              namespace=ns,
                              init=True, repr=False, eq=False, frozen=False, unsafe_hash=True
                              )


class Model(type):
    def __new__(mcs, name, bases, namespace):
        cls = super(Model, mcs).__new__(mcs, name, bases, namespace)
        return model(cls)

