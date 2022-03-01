import torch.nn as nn
import torch

from ..modules import *
from ..networks import *

__all__ = [
    "qx_io",
    "mag_spec_io",
    "pol_spec_io"
]


def qx_io(q_levels, net_dim, mlp_dim, mlp_activation=nn.ReLU()):
    return nn.Embedding(q_levels, net_dim), \
           SingleClassMLP(net_dim, mlp_dim, q_levels, mlp_activation)


def mag_spec_io(spec_dim, net_dim, in_chunks, out_chunks, scaled_activation=False):
    return Chunk(nn.Linear(spec_dim, net_dim * in_chunks),
                 in_chunks, sum_out=True), \
           nn.Sequential(Chunk(nn.Linear(net_dim, spec_dim * out_chunks),
                               out_chunks, sum_out=True),
                         ScaledSigmoid(spec_dim, with_range=False) if scaled_activation else Abs())


def pol_spec_io(spec_dim, net_dim, in_chunks, out_chunks, scaled_activation=False, phs='a'):
    pi = torch.acos(torch.zeros(1)).item()
    # act_phs = ScaledTanh(out_dim, with_range=False) if scaled_activation else nn.Tanh()
    # act_phs = nn.Tanh()
    act_phs = nn.Identity() if phs in ("a", "b") else nn.Tanh()
    act_mag = ScaledSigmoid(spec_dim, with_range=False) if scaled_activation else Abs()

    class ScaledPhase(HOM):
        def __init__(self):
            super(ScaledPhase, self).__init__(
                "x -> phs",
                *(Maybe(phs == "b",
                        (lambda self, x: torch.cos(self.psis.to(x) * x) * pi, "self, x -> x"))),
                (nn.Sequential(Chunk(nn.Linear(net_dim, spec_dim * out_chunks), out_chunks, sum_out=True), act_phs),
                 'x -> phs'),
                *(Maybe(phs == "a",
                        (lambda self, phs: torch.cos(
                            phs * self.psis.to(phs).view(*([1] * (len(phs.shape) - 1)), -1)) * pi,
                         'self, phs -> phs'))),
                *(Maybe(phs in ("b", "c"),
                        (lambda self, phs: phs * pi, 'self, phs -> phs'))),
            )
            self.psis = nn.Parameter(torch.ones(*((spec_dim,) if phs in ("a", "c") else (1, net_dim))))

    return HOM('x -> x',
               (Flatten(2), 'x -> x'),
               (Chunk(nn.Linear(spec_dim * 2, net_dim * in_chunks), in_chunks, sum_out=True), 'x -> x')), \
           HOM("x -> y",
               # phase module
               (ScaledPhase(), 'x -> phs'),
               # magnitude module
               (nn.Sequential(Chunk(nn.Linear(net_dim, spec_dim * out_chunks), out_chunks, sum_out=True), act_mag),
                'x -> mag'),
               (lambda mag, phs: torch.stack((mag, phs), dim=-1), "mag, phs -> y")
               )
