import torch.nn as nn
import torch

from ..modules import *
from ..networks import *

__all__ = [
    "qx_io",
    "wn_qx_io",
    "mag_spec_io",
    "pol_spec_io"
]


def qx_io(q_levels, net_in_dim, net_out_dim, mlp_dim, mlp_activation=nn.ReLU()):
    return nn.Embedding(q_levels, net_in_dim), \
           SingleClassMLP(net_out_dim, mlp_dim, q_levels, mlp_activation)


def wn_qx_io(q_levels, net_in_dim, net_out_dim, mlp_dim, mlp_activation=nn.ReLU()):
    class Dropout1d(nn.Dropout):

        def forward(self, input):
            # if self.training:
            # N, L, D = input.size()
            # ones = super(Dropout1d, self).forward(torch.ones(N * L).to(input.device))
            # out = (input.view(-1, D) * ones.unsqueeze(1)).view(N, L, D)
            input = ((input.float() / q_levels) - .5) * 4
            return input.unsqueeze(-1)
            # return input

    inpt_mod = HOM("x -> y1",
                   (nn.Sequential(
                       # nn.Embedding(q_levels, net_in_dim),
                       Dropout1d(0), ), "x -> y1")
                   )
    # embd = nn.utils.weight_norm(nn.Embedding(q_levels, net_in_dim), "weight")
    # return inpt_mod, ConvMLP(net_out_dim, mlp_dim, q_levels)
    return inpt_mod, SingleClassMLP(net_out_dim, mlp_dim, q_levels, learn_temperature=True, n_hidden_layers=3)


def mag_spec_io(spec_dim, net_dim, in_chunks, out_chunks, scaled_activation=False, with_sampler=False):
    return Chunk(nn.Linear(spec_dim, net_dim * in_chunks, bias=False),
                 in_chunks, sum_out=True), \
           nn.Sequential(Chunk(nn.Linear(net_dim, spec_dim * out_chunks, bias=False),
                               out_chunks, sum_out=True),
                         *((ParametrizedGaussian(spec_dim, spec_dim, False, False),) if with_sampler else ()),
                         ScaledSigmoid(spec_dim, with_range=False) if scaled_activation else Abs())
    # HOM("x -> y",
    #     *(Maybe(with_sampler,
    #             (ParametrizedGaussian(net_dim, spec_dim, True, False), "x -> f"))),
    #     *(Maybe(not with_sampler,
    #             (Chunk(nn.Linear(net_dim, spec_dim * out_chunks, bias=False),
    #                    out_chunks, sum_out=True), "x -> f"))),
    #     (nn.Sequential(nn.Linear(net_dim, spec_dim), nn.Sigmoid()), "x -> temp"),
    #     (lambda f, temp: (f.abs_() / temp), "f, temp -> y")
    #     )


def pol_spec_io(spec_dim, net_dim, in_chunks, out_chunks, scaled_activation=False, phs='a', with_sampler=False):
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
                (nn.Sequential(Chunk(nn.Linear(net_dim, spec_dim * out_chunks, bias=False), out_chunks, sum_out=True),
                               act_phs),
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
               (Chunk(nn.Linear(spec_dim * 2, net_dim * in_chunks, bias=False), in_chunks, sum_out=True), 'x -> x')), \
           HOM("x -> y",
               *(Maybe(with_sampler,
                       (ParametrizedGaussian(net_dim, net_dim, False, False), "x -> x"))),
               # phase module
               (ScaledPhase(), 'x -> phs'),
               # magnitude module
               (nn.Sequential(Chunk(nn.Linear(net_dim, spec_dim * out_chunks, bias=False), out_chunks, sum_out=True),
                              act_mag),
                'x -> mag'),
               (lambda mag, phs: torch.stack((mag, phs), dim=-1), "mag, phs -> y")
               )
    # HOM("x -> y",
    #     (HOM("x -> mag",
    #          (Chunk(nn.Linear(net_dim, spec_dim * out_chunks, bias=False),
    #                 out_chunks, sum_out=True), "x -> f"),
    #          (nn.Sequential(nn.Linear(net_dim, spec_dim), nn.Sigmoid()), "x -> temp"),
    #          (lambda f, temp: f / temp, "f, temp -> mag")
    #          ), "x -> mag"),
    #     (HOM("x -> phs",
    #          (Chunk(nn.Linear(net_dim, spec_dim * out_chunks, bias=False),
    #                 out_chunks, sum_out=True), "x -> f"),
    #          # (nn.Sequential(nn.Linear(net_dim, spec_dim), nn.Sigmoid()), "x -> temp"),
    #          (lambda f: nn.Hardtanh()(f) * pi * 1.1, "f -> phs")
    #          ), "x -> phs"),
    #     (lambda mag, phs: torch.stack((mag, phs), dim=-1), "mag, phs -> y")
    #     )
