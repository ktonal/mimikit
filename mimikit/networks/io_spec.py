from enum import auto
from typing import Tuple, Optional, Union
import torch.nn as nn
import dataclasses as dtc

from ..utils import AutoStrEnum
from ..config import Config
from ..features.ifeature import Feature
from ..modules.io import IOFactory
from ..modules.loss_functions import MeanL1Prop


__all__ = [
    "InputSpec",
    "TargetSpec",
    "IOSpec"
]


class InputSpec(Config):
    var_name: str
    feature: Feature
    module: IOFactory

    def __post_init__(self):
        # wire feature -> module
        pass


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
    var_name: str
    feature: Feature
    module: IOFactory
    objective: Objective

    def __post_init__(self):
        # wire feature, objective, module
        pass

    @staticmethod
    def cross_entropy(output, target):
        criterion = nn.CrossEntropyLoss(reduction="mean")
        L = criterion(output.view(-1, output.size(-1)), target.view(-1))
        return {"loss": L}

    @staticmethod
    def mean_l1_prop(output, target):
        criterion = MeanL1Prop()
        return {"loss": criterion(output, target)}


class IOSpec(Config):
    inputs: Tuple[InputSpec, ...]
    targets: Tuple[TargetSpec, ...]

    def batch(self):
        inputs = {spec.var_name: spec.feature for spec in self.inputs}
        targets = {spec.var_name: spec.feature for spec in self.targets}
        return inputs, targets

    def loss_fn(self):
        funcs = {str(trgt.objective) for trgt in self.targets}
        assert len(funcs) == 1, "only one objective per IOSpec supported"
        return getattr(TargetSpec, funcs.pop())
