import sys
import abc
from typing import Optional, Tuple, Dict
import torch

from ..config import Config

__all__ = [
    "Checkpointable",
    "ARM",
    "ARMWithHidden"
]


class Checkpointable(abc.ABC):
    @classmethod
    def __init_subclass__(cls, **kwargs):
        """
        add custom class to the mimikit namespace
        for saving/loading checkpoints of custom models
        """
        super().__init_subclass__()
        # print("__main__" in sys.modules.keys(), cls.__module__)
        # todo: store the class original module when saving configs!
        mmk = sys.modules["mimikit"]
        attr = cls.__qualname__
        # check that class was defined OUTSIDE of mimikit
        if "mimikit" not in cls.__module__ and not hasattr(mmk, attr):
            setattr(mmk, attr, cls)

    @classmethod
    @abc.abstractmethod
    def from_config(cls, config: Config):
        ...

    @property
    @abc.abstractmethod
    def config(self) -> Config:
        ...


class ARM(Checkpointable, torch.nn.Module):
    """ Interface for Auto Regressive Networks """

    @property
    def device(self):
        return next(self.parameters()).device

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
