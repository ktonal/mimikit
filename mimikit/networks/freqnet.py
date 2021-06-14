import torch
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
    """
    n_layers: tuple = (4,)
    n_input_heads: int = 1
    n_output_heads: int = 1
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
    gated_units: bool = True
    skip_dim: Optional[int] = None
    residuals_dim: Optional[int] = None
    reverse_dilation_order: bool = False

    def inpt_(self):
        return H.Paths(
            nn.Sequential(
                nn.Linear(self.input_dim, self.gate_dim * self.n_input_heads),
                FreqNetNetwork.ChunkSum(self.n_input_heads),
                Ops.Transpose(1, 2)
            ),
            # conditioning parameters :
            nn.Sequential(
                nn.Embedding(self.n_cin_classes, self.cin_dim), Ops.Transpose(1, 2)) if self.cin_dim else None,
            nn.Sequential(
                nn.Embedding(self.n_gin_classes, self.gin_dim), Ops.Transpose(1, 2)) if self.gin_dim else None
        )

    def outpt_(self):
        return nn.Sequential(
            Ops.Transpose(1, 2),
            nn.Linear(self.gate_dim if self.skip_dim is None else self.skip_dim, self.input_dim * self.n_output_heads),
            FreqNetNetwork.ChunkSum(self.n_output_heads),
            Ops.Abs()
        )

    @staticmethod
    def predict_(outpt, temp=None):
        return outpt

    class ChunkSum(nn.Module):

        def __init__(self, n_chunks):
            super(FreqNetNetwork.ChunkSum, self).__init__()
            self.n_chunks = n_chunks

        def forward(self, x):
            return sum(torch.chunk(x, self.n_chunks, dim=-1))