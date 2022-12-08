import sys
from omegaconf import OmegaConf
from typing import Protocol, List, Optional, Tuple, AnyStr
import dataclasses as dtc
from functools import reduce


__all__ = [
    "Config",
    "ParameterConfig",
    'ModuleConfig',
    'FeatureConfig',
    'ModelConfig',
    'OptimConfig',
    'TrainingConfig'
]

# type check done by OmegaConf when calling serialize():
# if field is not assigned -> ValidationError!
NOT_NULL = lambda: dtc.field(init=False, default=None)


# noinspection PyTypeChecker
def _get_mmk_type(class_) -> type:
    try:
        mmk = sys.modules["mimikit"]
        return reduce(lambda o, a: getattr(o, a), class_.split("."), mmk)
    except AttributeError:
        raise ImportError(f"class '{class_}' could not be found in mimikit")


class Config(Protocol):

    @staticmethod
    def validate_class(cls: type):
        if len(cls.__qualname__.split(".")) < 2:
            raise TypeError(f"Please define your Config class *within*"
                            f" a Checkpointable class so that it can be saved and loaded")
        if "__dataclass_fields__" not in cls.__dict__:
            raise TypeError("Please decorate your Config class with @dataclass"
                            " so that it can be (de)serialized")

    @property
    def owner_class(self):
        qualname = type(self).__qualname__
        return _get_mmk_type(qualname.split(".")[0])

    def serialize(self):
        self.validate_class(type(self))
        cfg = OmegaConf.structured(self)
        OmegaConf.set_struct(cfg, False)
        cfg.class_ = type(self).__qualname__
        return OmegaConf.to_yaml(cfg)

    @staticmethod
    def deserialize(raw_yaml):
        cfg = OmegaConf.create(raw_yaml)
        cls = _get_mmk_type(cfg.class_)
        cfg.pop("class_")
        return OmegaConf.to_object(OmegaConf.merge(cls(), cfg))

    @classmethod
    def from_file(cls, path: str):
        with open(path, 'r') as f:
            return cls.deserialize(f.read())

    def validate(self) -> Tuple[bool, AnyStr]:
        return True, ''



class ParameterConfig(Config):
    pass


@dtc.dataclass
class ModuleConfig(Config):
    module_class: str = NOT_NULL()

    def to_impl(self):
        cls = _get_mmk_type(self.module_class)
        hp = dtc.asdict(self)
        hp.pop("module_class")
        return cls(**hp)


@dtc.dataclass
class FeatureConfig(Config):
    pass


@dtc.dataclass
class ModelConfig(Config):
    inputs: List[FeatureConfig] = NOT_NULL()
    outputs: List[FeatureConfig] = NOT_NULL()
    network: ModuleConfig = NOT_NULL()


@dtc.dataclass
class OptimConfig(Config):
    optimizer: Config = NOT_NULL()
    scheduler: Optional[Config] = None


@dtc.dataclass
class TrainingConfig(Config):
    batch: Config = NOT_NULL()
    optim: OptimConfig = NOT_NULL()
    options: Config = NOT_NULL()


@dtc.dataclass
class DatasetConfig(Config):
    sources: List[str] = NOT_NULL()


@dtc.dataclass
class MMKConfig(Config):
    dataset: DatasetConfig = NOT_NULL()
    model: ModelConfig = NOT_NULL()
    training: TrainingConfig = NOT_NULL()
