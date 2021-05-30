import dataclasses as dtc
from inspect import signature

import pytorch_lightning as pl


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
    bases = [cls, *bases]

    # get the __init__ signatures of the bases
    sigs = [dict(signature(d.__init__).parameters) for d in bases
            if d is not pl.LightningModule]
    # collect positional arguments for validation
    posits = {k: v for s in sigs for k, v in s.items() if v.default is v.empty}
    if any(k for k in posits.keys() if k != "self"):
        for b in bases:
            posits = {k: v for k, v in dict(signature(b.__init__).parameters).items() if v.default is v.empty}
            if any(k for k in posits.keys() if k != "self"):
                raise ValueError("All dependencies should have kwargs only."
                                 " Got following positional args: " + str([k for k in posits.keys() if k != "self"]) +\
                                 "for the following parent class: " + str(b))

    # collect keywords arguments
    kws = {k: v for s in sigs for k, v in s.items() if v.default is not v.empty}

    # build fields for make_dataclass
    fields = [(k, type(v.default), dtc.field(default=v.default)) for k, v in kws.items()]

    # method to initialize all the bases
    def __post_init__(self):
        super(pl.LightningModule, self).__init__()
        for b in bases:
            if hasattr(b, '__post_init__'):
                # avoid recursion if b is a dataclass already
                b.__post_init__(self)
            else:
                b.__init__(self, **_filter_cls_kwargs(b, self.__dict__))

        # set hparams manually since lightning will not be able to infer them...
        self._set_hparams({k: getattr(self, k) for k in self.__args__})

    # namespace of the class
    ns = dict(cls.__dict__)
    ns.update({
        "__post_init__": __post_init__,
        '__args__': tuple(kws.keys()),
    })
    return dtc.make_dataclass(cls.__qualname__,
                              fields,
                              bases=(*(b for b in bases if b is not cls),
                                     *((pl.LightningModule,) if pl.LightningModule not in bases else tuple())),
                              namespace=ns,
                              init=True, repr=False, eq=False, frozen=False, unsafe_hash=True
                              )


class Model(type):
    def __new__(mcs, name, bases, namespace):
        cls = super(Model, mcs).__new__(mcs, name, bases, namespace)
        return model(cls)