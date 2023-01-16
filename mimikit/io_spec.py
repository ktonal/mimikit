from enum import auto
from typing import Tuple, Optional, List, Dict
import torch.nn as nn
import dataclasses as dtc
import h5mapper as h5m

from .utils import AutoStrEnum
from .config import Config
from .features.dataset import DatasetConfig
from .features.extractor import Extractor
from .features.item_spec import Unit, Sample, Frame, ItemSpec
from .features.functionals import Continuous, Discrete, Functional
from .modules.targets import CategoricalSampler,\
    MixOfLogisticsSampler, MixOfLogisticsLoss, MixOfGaussianLoss, MixOfGaussianSampler,\
    MixOfLaplaceLoss, MixOfLaplaceSampler
from .modules.io import IOFactory
from .modules.loss_functions import MeanL1Prop

__all__ = [
    "InputSpec",
    "ObjectiveType",
    "Objective",
    "TargetSpec",
    "IOSpec"
]


@dtc.dataclass
class _FeatureSpec(Config, type_field=False):
    extractor_name: str
    transform: Functional
    module: IOFactory
    extractor: Extractor = dtc.field(init=False, repr=False, metadata=dict(omegaconf_ignore=True))
    # logger: None = None

    def bind_to(self, extractor: Extractor):
        self.extractor = extractor

    @property
    def units(self):
        return [f.unit for f in [self.extractor.functional, self.transform] if f.unit is not None]

    @property
    def unit(self):
        return self.units[-1]

    @property
    def elem_type(self):
        el = tuple(f.elem_type for f in [self.extractor.functional, self.transform] if f.elem_type is not None)
        return el[-1]

    @property
    def sr(self):
        sr = [f.unit.sr for f in [self.extractor.functional, self.transform]
              if isinstance(f.unit, Sample) and f.unit.sr is not None]
        return sr[-1] if any(sr) else None

    @property
    def hop_length(self):
        hops = [f.unit.hop_length for f in [self.extractor.functional, self.transform]
                if isinstance(f.unit, Frame)]
        return hops[-1] if any(hops) else None

    def to_batch_item(self, item_spec: ItemSpec):
        item_spec = item_spec.to(self.extractor.functional.unit)
        return h5m.Input(
            data=self.extractor.name,
            getter=h5m.AsSlice(
                dim=0, shift=item_spec.shift,
                length=item_spec.length,
                downsampling=item_spec.stride
            ),
            transform=self.transform
        )

    @property
    def inv(self):
        return self.transform.inv


@dtc.dataclass
class InputSpec(_FeatureSpec, type_field=False):

    def bind_to(self, extractor: Extractor):
        super(InputSpec, self).bind_to(extractor)
        # wire feature -> module
        if isinstance(self.elem_type, Discrete):
            self.module.set(class_size=self.elem_type.class_size)
        elif isinstance(self.elem_type, Continuous):
            self.module.set(in_dim=self.elem_type.vector_size)
        return self


class ObjectiveType(AutoStrEnum):
    reconstruction = auto()
    categorical_dist = auto()
    logistic_dist = auto()
    gaussian_dist = auto()
    laplace_dist = auto()
    logistic_vector = auto()
    bernoulli_dist = auto()
    continuous_bernoulli = auto()
    gaussian_elbo = auto()


@dtc.dataclass
class Objective(Config, type_field=False):
    objective_type: ObjectiveType
    params: Dict = dtc.field(default_factory=lambda: {})

    def get_criterion(self):
        if self.objective_type == "reconstruction":
            return MeanL1Prop()
        elif self.objective_type == "categorical_dist":
            return self.cross_entropy
        elif self.objective_type == "logistic_dist":
            return MixOfLogisticsLoss(**self.params)

    @staticmethod
    def cross_entropy(output, target):
        criterion = nn.CrossEntropyLoss(reduction="mean")
        return criterion(output.view(-1, output.size(-1)), target.view(-1))


@dtc.dataclass
class TargetSpec(_FeatureSpec, type_field=False):
    objective: Objective

    def bind_to(self, extractor: Extractor):
        super(TargetSpec, self).bind_to(extractor)
        # wire feature, objective, module
        if self.objective.objective_type == "reconstruction":
            assert isinstance(self.elem_type, Continuous)
            self.module.set(out_dim=self.elem_type.vector_size)
            self._loss_fn = self.mean_l1_prop
        elif self.objective.objective_type == "categorical_dist":
            assert isinstance(self.elem_type, Discrete)
            self.module.set(out_dim=self.elem_type.class_size, sampler=CategoricalSampler())
            self._loss_fn = self.cross_entropy
        elif self.objective.objective_type == 'logistic_dist':
            n_components = self.objective.params.get("n_components", 1)
            self.module.set(out_dim=1, sampler=MixOfLogisticsSampler(**self.objective.params),
                            n_params=3, n_components=n_components)
            self._loss_fn = lambda o, t: {"loss": MixOfLogisticsLoss(**self.objective.params)(o, t)}
        return self

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


@dtc.dataclass
class IOSpec(Config, type_field=False):
    inputs: Tuple[InputSpec, ...]
    targets: Tuple[TargetSpec, ...]

    def bind_to(self, dataset_config: DatasetConfig):
        schema = dataset_config.schema
        for f in [*self.inputs, *self.targets]:
            f.bind_to(schema[f.extractor_name])
        return self

    @property
    def sr(self):
        srs = {i.sr for i in [*self.inputs, *self.targets]}
        if len(srs) > 1:
            # it is the responsibility of the user to have consistent sr
            raise RuntimeError(f"Expected to find a single sample_rate "
                               f"but found several: '{srs}'")
        return srs.pop()

    @property
    def hop_length(self):
        hops = {i.hop_length for i in [*self.inputs, *self.targets]}
        if len(hops) > 1:
            # it is the responsibility of the user to have consistent sr
            raise RuntimeError(f"Expected to find a single hop_length "
                               f"but found several: '{hops}'")
        return hops.pop()

    @property
    def unit(self) -> Unit:
        units = {i.unit for i in [*self.inputs, *self.targets]}
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
            for i, (spec, o, t) in enumerate(zip(self.targets, output, target)):
                d = spec.loss_fn(o, t)
                L += d["loss"]
                out[f"output_{i}_{spec.objective.objective_type}"] = d["loss"]
            out["loss"] = L
            return out
        return func

    @property
    def matching_io(self) -> Tuple[List[Optional[int]], List[Optional[int]]]:
        # TODO: broken...
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

