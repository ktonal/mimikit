from typing import Tuple

from ..config import Config
from ..features.ifeature import Feature
from ..modules.io import ModuleFactory


__all__ = [
    "InputSpec",
    "TargetSpec",
    "IOSpec"
]


class InputSpec(Config):
    var_name: str
    data_key: str
    feature: Feature
    module: ModuleFactory.Config

    def __post_init__(self):
        # wire feature and module
        params = getattr(self.module, "module_params", {})
        if hasattr(self.feature, "class_size") and hasattr(params, "class_size"):
            params.class_size = self.feature.class_size
        if hasattr(self.feature, "out_dim") and hasattr(params, "in_dim"):
            params.in_dim = self.feature.out_dim


class TargetSpec(Config):
    var_name: str
    data_key: str
    feature: Feature
    module: ModuleFactory.Config
    objective: str


class IOSpec(Config):
    inputs: Tuple[InputSpec, ...]
    targets: Tuple[InputSpec, ...]
