import torch.nn as nn
from dataclasses import dataclass
from typing import Optional

from .wavenet import WNNetwork, WaveNetLayer
from ..modules import homs as H, ops as Ops

__all__ = [
    'FreqNetNetwork'
]


@dataclass(init=True, repr=False, eq=False, frozen=False, unsafe_hash=True)
class FreqNetNetwork(WNNetwork):
    """
    adapts Wavenet to the frequency-domain

    Parameters
    ----------
    n_layers: tuple

    input_dim: int

    n_cin_classes: Optional[int]

    cin_dim: Optional[int]

    n_gin_classes: Optional[int]

    gin_dim: Optional[int]

    gate_dim: int

    kernel_size: int

    groups: int

    accum_outputs: int

    pad_input: int

    skip_dim: Optional[int]

    residuals_dim: Optional[int]


    """
    n_layers: tuple = (4,)
    input_dim: int = 256
    n_cin_classes: Optional[int] = None
    cin_dim: Optional[int] = None
    n_gin_classes: Optional[int] = None
    gin_dim: Optional[int] = None
    gate_dim: int = 128
    kernel_size: int = 2
    groups: int = 1
    accum_outputs: int = 0
    pad_input: int = 0
    skip_dim: Optional[int] = None
    residuals_dim: Optional[int] = None

    def inpt_(self):
        return H.Paths(
            nn.Sequential(
                H.GatedUnit(nn.Linear(self.input_dim, self.gate_dim)), Ops.Transpose(1, 2)),
            # conditioning parameters :
            nn.Sequential(
                nn.Embedding(self.n_cin_classes, self.cin_dim), Ops.Transpose(1, 2)) if self.cin_dim else None,
            nn.Sequential(
                nn.Embedding(self.n_gin_classes, self.gin_dim), Ops.Transpose(1, 2)) if self.gin_dim else None
        )

    def layers_(self):
        return nn.Sequential(*[
            WaveNetLayer(i,
                         gate_dim=self.gate_dim,
                         skip_dim=self.skip_dim,
                         residuals_dim=self.residuals_dim,
                         kernel_size=self.kernel_size,
                         cin_dim=self.cin_dim,
                         gin_dim=self.gin_dim,
                         groups=self.groups,
                         pad_input=self.pad_input,
                         accum_outputs=self.accum_outputs,
                         )
            for block in self.n_layers for i in range(block)
        ])

    def outpt_(self):
        return nn.Sequential(
            Ops.Transpose(1, 2),
            nn.Linear(self.gate_dim if self.skip_dim is None else self.skip_dim, self.input_dim), Ops.Abs()
        )

    @staticmethod
    def predict_(outpt, temp=None):
        return outpt
