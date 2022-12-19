from enum import auto
from typing import Tuple, Optional, Union
import torch.nn as nn
import dataclasses as dtc

from ..modules.targets import CategoricalSampler
from ..utils import AutoStrEnum
from ..config import Config
from ..features.ifeature import Feature, DiscreteFeature, RealFeature
from ..modules.io import IOFactory
from ..modules.loss_functions import MeanL1Prop


__all__ = [
    "InputSpec",
    "ObjectiveType",
    "Objective",
    "TargetSpec",
    "IOSpec"
]


class InputSpec(Config):
    feature: Feature
    module: IOFactory

    def __post_init__(self):
        # wire feature -> module
        if isinstance(self.feature, DiscreteFeature):
            self.module.set(class_size=self.feature.class_size)
        elif isinstance(self.feature, RealFeature):
            self.module.set(in_dim=self.feature.out_dim)


class ObjectiveType(AutoStrEnum):
    reconstruction = auto()
    categorical_dist = auto()
    bernoulli_dist = auto()
    continuous_bernoulli = auto()
    logistic_dist = auto()
    gaussian_dist = auto()
    gaussian_elbo = auto()


class Objective(Config):
    objective_type: ObjectiveType
    n_components: Optional[int] = None
    n_params: int = dtc.field(
        init=False, repr=False, default=1
    )
    support: Union[int, Tuple[float, float]] = dtc.field(
        init=False, repr=False, default=2
    )


class TargetSpec(Config):
    feature: Feature
    module: IOFactory
    objective: Objective

    def __post_init__(self):
        # wire feature, objective, module
        if self.objective.objective_type == "reconstruction":
            assert isinstance(self.feature, RealFeature)
            self.module.set(out_dim=self.feature.out_dim)
            self._loss_fn = self.mean_l1_prop
        elif self.objective.objective_type == "categorical_dist":
            assert isinstance(self.feature, DiscreteFeature)
            self.module.set(out_dim=self.feature.class_size, sampler=CategoricalSampler())
            self._loss_fn = self.cross_entropy

    @staticmethod
    def cross_entropy(output, target):
        criterion = nn.CrossEntropyLoss(reduction="mean")
        L = criterion(output.view(-1, output.size(-1)), target.view(-1))
        return {"loss": L}

    @staticmethod
    def mean_l1_prop(output, target):
        criterion = MeanL1Prop()
        return {"loss": criterion(output, target)}

    @property
    def loss_fn(self):
        return self._loss_fn


class IOSpec(Config):
    inputs: Tuple[InputSpec, ...]
    targets: Tuple[TargetSpec, ...]

