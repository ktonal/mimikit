from copy import deepcopy
from enum import Enum

import numpy as np
import torch
import dataclasses as dtc
import abc
from typing import Protocol, Union, Tuple, Optional
from omegaconf import OmegaConf, DictConfig
import h5mapper as h5m

from ..config import Config, private_runtime_field

__all__ = [
    "SequenceSpec",
    "TimeUnit",
    "Feature",
    "DiscreteFeature",
    "RealFeature",
    "Batch",
    'IFeature'
]


@dtc.dataclass
class SequenceSpec:
    sr: Optional[int] = None
    shift: int = 0
    length: int = 0
    frame_size: Optional[int] = None
    hop_length: Optional[int] = None
    center: bool = False


def samples2hops(x, spec: SequenceSpec):
    return x // spec.hop_length


def hops2samples(x, spec: SequenceSpec):
    return x * spec.hop_length


def samples2frames(x, spec: SequenceSpec):
    extra = (spec.frame_size - spec.hop_length)
    return samples2hops(x - extra, spec)


def frames2samples(x, spec: SequenceSpec):
    return hops2samples(x, spec) + (spec.frame_size - spec.hop_length)


def steps2samples(x, spec: SequenceSpec):
    pass


def samples2steps(x, spec: SequenceSpec):
    pass


def samples2seconds(x, spec: SequenceSpec):
    return x / spec.sr


def seconds2samples(x, spec: SequenceSpec):
    return int(x * spec.sr)


class TimeUnit(Enum):
    sample = 0
    frame = 1
    hop = 2
    step = 3
    second = 4

    @classmethod
    def from_name(cls, name):
        if isinstance(name, str) and hasattr(TimeUnit, name):
            return getattr(TimeUnit, name)
        elif name in TimeUnit:
            return name
        else:
            raise TypeError(f"unit '{name}' is not a TimeUnit member")

    @classmethod
    def to_samples(cls, x, unit: Union["TimeUnit", str], spec: SequenceSpec):
        if unit is cls.sample:
            return x
        if unit is cls.frame:
            return frames2samples(x, spec)
        if unit is cls.hop:
            return hops2samples(x, spec)
        if unit is cls.step:
            return x
        if unit is cls.second:
            return seconds2samples(x, spec)


# @dtc.dataclass
class Feature(abc.ABC, Config):
    sr: dtc.InitVar[Optional[int]] = None
    seq_spec: SequenceSpec = private_runtime_field(None)

    def __post_init__(self, sr):
        self.seq_spec = SequenceSpec(sr)

    @abc.abstractmethod
    def t(self, inputs):
        ...

    @abc.abstractmethod
    def inv(self, inputs):
        ...

    @property
    @abc.abstractmethod
    def h5m_type(self) -> h5m.Feature:
        ...

    def add_shift(self, s: int, unit: Union[TimeUnit, str] = TimeUnit.sample):
        unit = TimeUnit.from_name(unit)
        if unit is TimeUnit.step and hasattr(self, "hop_length"):
            unit = TimeUnit.hop
        s = TimeUnit.to_samples(s, unit, self.seq_spec)
        self.seq_spec.shift += s
        return self

    def add_length(self, l: int, unit: Union[TimeUnit, str] = TimeUnit.sample):
        unit = TimeUnit.from_name(unit)
        if unit is TimeUnit.step and hasattr(self, "hop_length"):
            unit = TimeUnit.hop
        v = TimeUnit.to_samples(abs(l), unit, self.seq_spec)
        self.seq_spec.length += v if l > 0 else -v
        return self

    def set_frame(self, fs: int, hl: int):
        self.seq_spec.frame_size = fs
        self.seq_spec.hop_length = hl
        return self

    def batch_item(self, length=0, unit=TimeUnit.step, downsampling=1) -> h5m.Input:
        self.add_length(length, unit)
        s = self.seq_spec
        if hasattr(self, "hop_length"):
            s.length += s.frame_size - s.hop_length
        if not getattr(self, "as_framed_slice", False):
            getter = h5m.AsSlice(shift=s.shift, length=s.length, downsampling=downsampling)
        else:
            # TODO: either Spectrogram must use rfft as transform or this must be removed!
            getter = h5m.AsFramedSlice(shift=s.shift, length=s.length,
                                       frame_size=s.frame_size, hop_length=s.hop_length,
                                       center=s.center,
                                       downsampling=downsampling)
        return h5m.Input(data="snd", getter=getter, transform=self.t)

    def seconds_to_steps(self, duration):
        steps = seconds2samples(duration, self.seq_spec)
        if hasattr(self, "hop_length"):
            return samples2frames(steps, self.seq_spec)
        return steps


class DiscreteFeature(Feature, abc.ABC):
    @property
    @abc.abstractmethod
    def class_size(self) -> int:
        ...

    @property
    @abc.abstractmethod
    def vector_dim(self) -> int:
        ...


class RealFeature(Feature, abc.ABC):
    @property
    @abc.abstractmethod
    def out_dim(self) -> int:
        ...

    @property
    @abc.abstractmethod
    def support(self) -> Tuple[float, float]:
        ...


class Batch(Config):
    inputs: Tuple[Feature, ...] = (),
    targets: Tuple[Feature, ...] = (),

    def serialize(self):
        feats = {"inputs": [], "targets": []}
        for i, inpt in enumerate(self.inputs):
            feats[f"inputs"].append({i: inpt.dict()})
        for i, trgt in enumerate(self.targets):
            feats[f"targets"].append({i: trgt.dict()})
        return OmegaConf.to_yaml(OmegaConf.create(feats))

    @staticmethod
    def deserialize(raw_yaml):
        cfg: DictConfig = OmegaConf.create(raw_yaml)
        inputs = []
        targets = []

        for data in cfg.inputs:
            data = [*data.values()][0]
            inputs += [Feature.object(data)]
        for data in cfg.targets:
            data = [*data.values()][0]
            targets += [Feature.object(data)]

        return Batch(inputs=inputs, targets=targets)

    def __repr__(self):
        inpt = ',\n\t'.join([repr(x) for x in self.inputs])
        trgt = ',\n\t'.join([repr(x) for x in self.targets])
        return f"<Batch\n" \
               f"   inputs=(\n\t{inpt})\n" \
               f"   targets=(\n\t{trgt})>"

    def __eq__(self, other):
        if not isinstance(other, Batch):
            return False
        feats_self, feats_other = [*self.inputs, *self.targets], [*other.inputs, *other.targets]
        if len(feats_self) != len(feats_other):
            return False
        for f_self, f_other in zip(feats_self, feats_other):
            if f_self != f_other:
                return False
        return True


DataType = Union[np.ndarray, torch.Tensor]


class IFeature(Protocol):
    """template for ML data lifecycle"""

    @abc.abstractmethod
    def load(self, path: str) -> DataType:
        raise NotImplementedError

    @abc.abstractmethod
    def pre_process(self, x: DataType) -> DataType:
        raise NotImplementedError

    @abc.abstractmethod
    def getter(self, *args, **kwargs):
        raise NotImplementedError

    @abc.abstractmethod
    def transform(self, x: DataType) -> DataType:
        raise NotImplementedError

    @abc.abstractmethod
    def augment(self, x: DataType) -> DataType:
        raise NotImplementedError

    @abc.abstractmethod
    def input_module(self, *args, **kwargs) -> torch.nn.Module:
        raise NotImplementedError

    @abc.abstractmethod
    def output_module(self, *args, **kwargs) -> torch.nn.Module:
        raise NotImplementedError

    @abc.abstractmethod
    def loss_fn(self, x: DataType, y: DataType) -> DataType:
        raise NotImplementedError

    @abc.abstractmethod
    def inverse_transform(self, x: DataType) -> DataType:
        raise NotImplementedError

    @abc.abstractmethod
    def setter(self, *args, **kwargs):
        raise NotImplementedError

    @abc.abstractmethod
    def display(self, x: DataType):
        raise NotImplementedError

    @abc.abstractmethod
    def save(self, path: str, x: DataType) -> str:
        raise NotImplementedError
