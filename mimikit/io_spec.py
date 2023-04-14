from enum import auto
from typing import Tuple, Dict
from typing_extensions import Literal
import torch.nn as nn
import dataclasses as dtc
import h5mapper as h5m

from .utils import AutoStrEnum
from .config import Config
from .features.dataset import DatasetConfig
from .features.extractor import Extractor
from .features.item_spec import Unit, Sample, Frame, ItemSpec
from .features.functionals import *
from .modules.targets import *
from .modules.io import *
from .modules.activations import *
from .modules.loss_functions import MeanL1Prop

__all__ = [
    "InputSpec",
    "ObjectiveType",
    "Objective",
    "TargetSpec",
    "IOSpec",
]


@dtc.dataclass
class _FeatureSpec(Config, type_field=False):
    extractor_name: str
    transform: Functional
    module: IOModule
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
            self.module.set(class_size=self.elem_type.size)
        elif isinstance(self.elem_type, Continuous):
            self.module.set(in_dim=self.elem_type.size)
        return self


class ObjectiveType(AutoStrEnum):
    reconstruction = auto()
    categorical_dist = auto()
    logistic_dist = auto()
    gaussian_dist = auto()
    laplace_dist = auto()
    logistic_vector = auto()
    gaussian_vector = auto()
    laplace_vector = auto()


@dtc.dataclass
class Objective(Config, type_field=False):
    objective_type: ObjectiveType
    params: Dict = dtc.field(default_factory=lambda: {})

    def get_criterion(self):
        if self.objective_type == "reconstruction":
            return MeanL1Prop(eps=1.)
        elif self.objective_type == "categorical_dist":
            return self.cross_entropy

    def get_sampler(self):
        if self.objective_type == "reconstruction":
            return None
        elif self.objective_type == "categorical_dist":
            return CategoricalSampler()

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
        sampler = self.objective.get_sampler()
        if self.objective.objective_type == "reconstruction":
            assert isinstance(self.elem_type, Continuous)
            self.module.set(out_dim=self.elem_type.size)
        elif self.objective.objective_type == "categorical_dist":
            assert isinstance(self.elem_type, Discrete)
            self.module.set(out_dim=self.elem_type.size,
                            sampler=sampler)
        self.criterion = self.objective.get_criterion()
        return self

    def loss_fn(self, output, target):
        return {"loss": self.criterion(output, target)}


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
                # out.update(d)
            out["loss"] = L
            return out

        return func

    @dtc.dataclass
    class MuLawIOConfig(Config):
        sr: int = 16000
        q_levels: int = 256
        compression: float = 1.
        input_module_type: Literal['framed_linear', 'embedding'] = 'framed_linear'
        mlp_dim: int = 128
        n_mlp_layers: int = 0
        min_temperature: float = 1e-4

    @staticmethod
    def mulaw_io(
            config: MuLawIOConfig,
            extractor: Extractor = None,
    ):
        c = config
        if extractor is None:
            extractor = Extractor(
                "signal", Compose(
                    FileToSignal(c.sr), Normalize(), RemoveDC()
                )
            )
        mu_law = MuLawCompress(c.q_levels, c.compression)
        if config.input_module_type == "framed_linear":
            module_type = FramedLinearIO
        elif config.input_module_type == "embedding":
            module_type = EmbeddingIO
        else:
            raise ValueError(f"Unimplemented input_module_type: '{config.input_module_type}'")
        return IOSpec(
            inputs=(InputSpec(
                extractor_name=extractor.name,
                transform=mu_law,
                module=module_type()).bind_to(extractor),
                    ),
            targets=(TargetSpec(
                extractor_name=extractor.name,
                transform=mu_law,
                module=MLPIO(
                        hidden_dim=c.mlp_dim, n_hidden_layers=c.n_mlp_layers,
                        min_temperature=c.min_temperature
                    ),
                objective=Objective("categorical_dist")
            ).bind_to(extractor),))

    @dtc.dataclass
    class MagSpecIOConfig(Config):
        sr: int = 22050
        n_fft: int = 2048
        hop_length: int = 512
        activation: str = "Abs"

    @staticmethod
    def magspec_io(
            config: MagSpecIOConfig,
            extractor=None,
    ):
        c = config
        if extractor is None:
            extractor = Extractor("signal", Compose(
                FileToSignal(c.sr), Normalize(), RemoveDC()
            ))
        return IOSpec(
            inputs=(InputSpec(
                extractor_name=extractor.name,
                transform=MagSpec(c.n_fft, c.hop_length, center=False, window='hann'),
                module=ChunkedLinearIO(n_chunks=1)).bind_to(extractor),),
            targets=(TargetSpec(
                extractor_name=extractor.name,
                transform=MagSpec(c.n_fft, c.hop_length, center=False, window='hann'),
                module=ChunkedLinearIO(n_chunks=1,
                                       activation=ActivationConfig(
                                           act=c.activation,
                                       )),
                objective=Objective("reconstruction")
            ).bind_to(extractor),))

