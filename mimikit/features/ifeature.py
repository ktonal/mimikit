import numpy as np
import torch
import dataclasses as dtc
import abc
from typing import Protocol, Union, Tuple, Optional
from omegaconf import OmegaConf, DictConfig
import h5mapper as h5m

from ..config import Config

__all__ = [
    "Feature",
    "DiscreteFeature",
    "RealFeature",
    "Batch",
    'IFeature'
]


@dtc.dataclass
class SequenceSpec:
    data_key: str = "snd"
    shift: int = 0
    length: int = 0
    frame_size: Optional[int] = None
    hop_length: Optional[int] = None
    center: bool = False

    def add_shift(self, s: int):
        self.shift += s
        return self

    def add_length(self, l: int):
        self.length += l
        return self

    def set_frame(self, fs: int, hl: int):
        self.frame_size = fs
        self.hop_length = hl
        return self


# @dtc.dataclass
class Feature(abc.ABC, Config):
    @property
    @abc.abstractmethod
    def feature_dtype(self) -> h5m.Feature:
        ...

    train_spec: SequenceSpec = dtc.field(
        init=False, repr=False, default=SequenceSpec()
    )
    infer_spec: SequenceSpec = dtc.field(
        init=False, repr=False, default=SequenceSpec()
    )


class DiscreteFeature(abc.ABC, Feature):
    class_size: int = dtc.field(init=False, repr=False, default=2)
    vector_dim: int = dtc.field(init=False, repr=False, default=1)


class RealFeature(abc.ABC, Feature):
    out_dim: int = dtc.field(init=False, repr=False, default=1)
    support: Tuple[float, float] = dtc.field(
        init=False, repr=False, default=(-float("inf"), float("inf")))


class Batch(Config):
    inputs: Tuple[Feature] = (),
    targets: Tuple[Feature] = (),

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
