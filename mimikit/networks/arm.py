import abc
import dataclasses as dtc
from typing import Tuple, Dict, Set
import torch
import h5mapper as h5m

from ..features.item_spec import ItemSpec
from ..config import Config, Configurable
from ..io_spec import IOSpec

__all__ = [
    "NetworkConfig",
    "ARM",
    "ARMWithHidden"
]


@dtc.dataclass
class NetworkConfig(Config, abc.ABC):

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
    def config(self) -> NetworkConfig:
        ...

    @property
    @abc.abstractmethod
    def rf(self):
        ...

    @abc.abstractmethod
    def train_batch(self, item_spec: ItemSpec)\
            -> Tuple[Tuple[h5m.Input, ...], Tuple[h5m.Input, ...]]:
        ...

    @abc.abstractmethod
    def test_batch(self, item_spec: ItemSpec)\
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

    @property
    @abc.abstractmethod
    def generate_params(self) -> Set[str]:
        ...


class ARMWithHidden(ARM, abc.ABC):

    @abc.abstractmethod
    def reset_hidden(self) -> None:
        ...