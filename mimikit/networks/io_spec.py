from enum import auto
from typing import Tuple, Optional, Union, List
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

    @property
    def sr(self):
        srs = {i.feature.sr for i in [*self.inputs, *self.targets]}
        if len(srs) > 1:
            # it is the responsibility of the user to have consistent sr
            raise RuntimeError(f"Expected to find a single sample_rate "
                               f"but found several: '{srs}'")
        return srs.pop()

    @property
    def input_features(self) -> List[Feature]:
        return [i.feature for i in self.inputs]

    @property
    def target_features(self) -> List[Feature]:
        return [t.feature for t in self.targets]

    @property
    def matching_io(self) -> Tuple[List[Optional[int]], List[Optional[int]]]:
        inputs = [None] * len(self.inputs)
        targets = [None] * len(self.targets)
        trg_feats = self.target_features
        for n, i in enumerate(self.input_features):
            try:
                idx = trg_feats.index(i)
            except ValueError:
                idx = None
            if idx is not None:
                inputs[n] = idx
                targets[idx] = n
        return inputs, targets

    @property
    def is_auxiliary_target(self) -> List[bool]:
        in_feats = self.input_features
        return [f not in in_feats for f in self.target_features]

    @property
    def is_fixture_input(self) -> List[bool]:
        target_feats = self.target_features
        return [f not in target_feats for f in self.input_features]

