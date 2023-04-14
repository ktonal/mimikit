import abc
import sys
from copy import deepcopy
from omegaconf import OmegaConf, ListConfig, DictConfig
from typing import List, Tuple, Union, Dict, Any
import dataclasses as dtc
from functools import reduce, partial

__all__ = [
    "private_runtime_field",
    "Config",
    "Configurable",
]


def private_runtime_field(default):
    return dtc.field(init=False, repr=False, metadata=dict(omegaconf_ignore=True), default_factory=lambda: default)


# noinspection PyTypeChecker
def _get_type_object(type_) -> type:
    if ":" in type_:
        module, qualname = type_.split(":")
    else:
        module, qualname = "mimikit", type_
    try:
        m = sys.modules[module]
        return reduce(lambda o, a: getattr(o, a), qualname.split("."), m)
    except AttributeError or KeyError:
        raise ImportError(f"could not find class '{qualname}' from module {module} in current environment")


STATIC_TYPED_KEYS = {
    "dataset": "DatasetConfig",
    "io_spec": "IOSpec",
    "inputs": "InputSpec",
    "targets": "TargetSpec",
    "objective": "Objective",
    "feature": "Feature",
    "extractor": "Extractor",
    "activation": "ActivationConfig"
}


@dtc.dataclass
class Config:
    ## type: str = dtc.field(init=False, repr=False, default="mimikit.config:Config")

    @classmethod
    def __init_subclass__(cls, type_field=True, **kwargs):
        """add type info to subclass"""
        if type_field:
            default = f"{cls.__qualname__}"
            if not cls.__module__.startswith("mimikit"):
                default = f"{cls.__module__}:{default}"
            setattr(cls, "type", dtc.field(init=False, default=default, repr=False))
            if "__annotations__" in cls.__dict__:
                # put the type first for nicer serialization...
                ann = cls.__dict__["__annotations__"].copy()
                for k in ann:
                    cls.__dict__["__annotations__"].pop(k)
                cls.__dict__["__annotations__"].update({"type": str, **ann})
            else:
                setattr(cls, "__annotations__", {"type": str})

    @staticmethod
    def validate_class(cls: type):
        if "__dataclass_fields__" not in cls.__dict__:
            if not issubclass(cls, (tuple, list)):
                raise TypeError("Please decorate your Config class with @dataclass"
                                " so that it can be (de)serialized")

    @property
    def owner_class(self):
        module, type_ = type(self).__module__, type(self).__qualname__
        type_ = ".".join(type_.split(".")[:-1]) if "." in type_ else type_
        type_ = f"{module}:{type_}"
        return _get_type_object(type_)

    def serialize(self):
        self.validate_class(type(self))
        cfg = OmegaConf.structured(self)
        return OmegaConf.to_yaml(cfg)

    @staticmethod
    def deserialize(raw_yaml, as_type=None):
        cfg = OmegaConf.create(raw_yaml)
        return Config.object(cfg, as_type)

    @staticmethod
    def object(cfg: Union[ListConfig, DictConfig, Dict, List, Tuple, Any], as_type=None):
        if isinstance(cfg, (DictConfig, Dict)):
            for k, v in cfg.items():
                if k in STATIC_TYPED_KEYS:
                    cls = _get_type_object(STATIC_TYPED_KEYS[k])
                    setattr(cfg, k, Config.object(v, as_type=cls))
                elif k == "extractors":
                    setattr(cfg, k, tuple(map(partial(Config.object, as_type=_get_type_object("Extractor")), v)))
                elif isinstance(v, (ListConfig, DictConfig, Dict, List, Tuple)):
                    setattr(cfg, k, Config.object(v))
            if as_type is not None:
                cls = as_type
            elif "type" in cfg:
                cls = _get_type_object(cfg.type)
            else:  # untyped raw dict
                return cfg
            if isinstance(cfg, DictConfig):
                cfg._metadata.object_type = cls
                return OmegaConf.to_object(cfg)
            else:
                return cls(**cfg)

        elif isinstance(cfg, (ListConfig, List, Tuple)):
            return OmegaConf.to_object(OmegaConf.structured([*map(partial(Config.object, as_type=as_type), cfg)]))
        # any other kind of value
        return cfg

    def dict(self):
        """caution! nested configs are also converted!"""
        return dtc.asdict(self)

    def copy(self):
        return deepcopy(self)

    def validate(self) -> Tuple[bool, str]:
        return True, ''


class Configurable(abc.ABC):

    @classmethod
    @abc.abstractmethod
    def from_config(cls, config: Config):
        ...

    @property
    @abc.abstractmethod
    def config(self) -> Config:
        ...


