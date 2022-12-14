import abc
from typing import Optional, Tuple, Dict, ClassVar
import torch

from ..features.ifeature import Batch
from ..config import Configurable, Config

__all__ = [
    "ARMConfig",
    "ARM",
    "ARMWithHidden"
]


class ARMConfig(Config):
    batch: Batch


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
    def time_axis(self) -> int:
        ...

    @property
    @abc.abstractmethod
    def shift(self) -> int:
        """difference in # of steps
         between first predicted and first received steps"""
        ...

    @property
    @abc.abstractmethod
    def rf(self) -> int:
        """# of steps in the receptive field"""
        ...

    @property
    @abc.abstractmethod
    def hop_length(self) -> Optional[int]:
        """# of predicted steps in 1 inference pass"""
        ...

    @abc.abstractmethod
    def output_length(self, n_input_steps: int) -> int:
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
