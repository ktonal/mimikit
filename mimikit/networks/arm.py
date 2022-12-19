import abc
from typing import Tuple, Dict, Callable
import torch
import h5mapper as h5m

from ..features.ifeature import TimeUnit
from ..config import Configurable, Config
from .io_spec import IOSpec

__all__ = [
    "ARMConfig",
    "ARM",
    "ARMWithHidden"
]


class ARMConfig(Config, abc.ABC):

    @property
    @abc.abstractmethod
    def io_spec(self) -> IOSpec:
        ...


class ARM(Configurable, torch.nn.Module):
    """ Interface for Auto Regressive Networks """

    @property
    def device(self):
        return next(self.parameters()).device

    @property
    @abc.abstractmethod
    def config(self) -> ARMConfig:
        ...

    @property
    @abc.abstractmethod
    def rf(self):
        ...

    @abc.abstractmethod
    def train_batch(self, length=1, unit=TimeUnit.step, downsampling=1)\
            -> Tuple[Tuple[h5m.Input, ...], Tuple[h5m.Input, ...]]:
        ...

    @abc.abstractmethod
    def before_generate(self,
                        prompts: Tuple[torch.Tensor, ...],
                        batch_index: int
                        ) -> None:
        ...

    @abc.abstractmethod
    def generate_step(self,
                      inputs: Tuple[torch.Tensor, ...], *,
                      t: int = 0,
                      **parameters: Dict[str, torch.Tensor]
                      ) -> Tuple[torch.Tensor, ...]:
        ...

    @abc.abstractmethod
    def after_generate(self,
                       final_outputs: Tuple[torch.Tensor, ...],
                       batch_index: int
                       ) -> None:
        ...


class ARMWithHidden(ARM, abc.ABC):

    @abc.abstractmethod
    def reset_hidden(self) -> None:
        ...
