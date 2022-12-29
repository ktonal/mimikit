from enum import auto
from typing import Tuple, Optional, List
import torch.nn as nn
import dataclasses as dtc
import h5mapper as h5m

from ..modules.targets import CategoricalSampler, MoLSampler, MoLLoss
from ..utils import AutoStrEnum
from ..config import Config
from ..features.ifeature import Feature, DiscreteFeature, RealFeature, TimeUnit
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
    logistic_dist = auto()
    bernoulli_dist = auto()
    continuous_bernoulli = auto()
    gaussian_dist = auto()
    gaussian_elbo = auto()


class Objective(Config):
    objective_type: ObjectiveType
    n_components: Optional[int] = None
    n_params: int = dtc.field(
        init=False, repr=False, default=1
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
        elif self.objective.objective_type == 'logistic_dist':
            self.module.set(out_dim=1, sampler=MoLSampler(self.objective.n_components),
                            n_params=3, n_components=self.objective.n_components)
            self._loss_fn = MoLLoss(self.objective.n_components, 'mean')

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


EXTRACTOR_TYPE_2_DATASET_ATTR = dict(
    Sound='snd', SignalEnvelop='env'
)


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
    def hop_length(self):
        hops = {i.feature.hop_length for i in [*self.inputs, *self.targets]}
        if len(hops) > 1:
            # it is the responsibility of the user to have consistent sr
            raise RuntimeError(f"Expected to find a single hop_length "
                               f"but found several: '{hops}'")
        return hops.pop()

    @property
    def unit(self) -> TimeUnit:
        units = {i.feature.time_unit for i in [*self.inputs, *self.targets]}
        if len(units) > 1:
            # it is the responsibility of the user to have consistent unit
            raise RuntimeError(f"Expected to find a single time unit "
                               f"but found several: '{units}'")
        return units.pop()

    @property
    def loss_fn(self):
        def func(output, target):
            out = {}
            L = 0.
            for spec, o, t in zip(self.targets, output, target):
                d = spec.loss_fn(o, t)
                L += d["loss"]
                out[str(type(spec.feature))] = d["loss"]
            out["loss"] = L
            return out
        return func

    def dataset_class(self):
        feats = [*self.input_features, *self.target_features]
        attrs = {}
        for i, f in enumerate(feats):
            extractor = f.h5m_type
            tp = type(extractor).__qualname__.split(".")[0]
            attr = EXTRACTOR_TYPE_2_DATASET_ATTR[tp]
            attrs[attr] = extractor
            f.dataset_attr = attr
        return type("Dataset", (h5m.TypedFile, ), attrs)

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

