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
    class OHEmbedding(nn.Module):
        def __init__(self, n_classes, net_in_dim):
            super(OHEmbedding, self).__init__()
            self.n_classes = n_classes
            self.cv = nn.Conv1d(n_classes, net_in_dim, kernel_size=(1,), bias=False)

        def forward(self, x):
            out = nn.functional.one_hot(x, self.n_classes).to(x.device).float()
            if self.training:
                return nn.Dropout2d(1 / 3)(out)
            return out
            # return self.cv(y.transpose(-1, -2).contiguous()).transpose(-1, -2).contiguous()

    class ConvMLP(nn.Module):
        def __init__(self, net_out_dim, mlp_dim, n_classes):
            super(ConvMLP, self).__init__()
            self.cv1 = nn.Sequential(
                nn.Linear(net_out_dim, n_classes, bias=False),
            )
            self.cv2 = nn.Sequential(
                nn.Linear(n_classes, 1, bias=False),
            )
            # self.cv3 = nn.Linear(n_classes, 1, bias=False)
            # self.bn = nn.BatchNorm1d(n_classes)
            self.Q = n_classes
            self.top_k = None
            self.top_p = None

        def forward(self, x, temperature=None):
            # x = x.transpose(-1, -2).contiguous()
            outpt = nn.Mish()(self.cv1(x))
            outpt = torch.sin(self.cv2(outpt))
            # outpt = outpt.transpose(-1, -2).contiguous()
            return outpt.squeeze(-1)
            # Roundable Sigmoid :
            # outpt = (self.bn(self.cv2(outpt) / (self.Q/4))).transpose(-1, -2).contiguous() * self.Q
            # return outpt.squeeze() if self.training else outpt.squeeze(-1)
            # Standard Pr_Y
            # outpt = (self.bn(outpt + self.cv2(outpt))).transpose(-1, -2).contiguous()
            # if self.training:
            #     return outpt
            # if temperature is None:
            #     return outpt.argmax(dim=-1)
            # else:
            #     if not isinstance(temperature, torch.Tensor):
            #         temperature = torch.Tensor([temperature]).reshape(*([1] * (len(outpt.size()))))
            #     probas = outpt.squeeze() / temperature.to(outpt)
            #     if self.top_k is not None:
            #         indices_to_remove = probas < torch.topk(probas, self.top_k)[0][..., -1, None]
            #         probas[[indices_to_remove]] = - float("inf")
            #         probas = nn.Softmax(dim=-1)(probas)
            #     elif self.top_p is not None:
            #         sorted_logits, sorted_indices = torch.sort(probas, descending=True)
            #         cumulative_probs = torch.cumsum(nn.Softmax(dim=-1)(sorted_logits), dim=-1)
            #
            #         # Remove tokens with cumulative probability above the threshold
            #         sorted_indices_to_remove = cumulative_probs > self.top_p
            #         # Shift the indices to the right to keep also the first token above the threshold
            #         sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            #         sorted_indices_to_remove[..., 0] = 0
            #
            #         indices_to_remove = sorted_indices[sorted_indices_to_remove]
            #         probas[indices_to_remove] = - float("inf")
            #         probas = nn.Softmax(dim=-1)(probas)
            #     else:
            #         probas = nn.Softmax(dim=-1)(probas)
            #     if probas.dim() > 2:
            #         o_shape = probas.shape
            #         probas = probas.view(-1, o_shape[-1])
            #         return torch.multinomial(probas, 1).reshape(*o_shape[:-1])
            #     return torch.multinomial(probas, 1)

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
    return inpt_mod, SingleClassMLP(net_out_dim, mlp_dim, q_levels)


def mag_spec_io(spec_dim, net_dim, in_chunks, out_chunks, scaled_activation=False, with_sampler=False):
    return Chunk(nn.Linear(spec_dim, net_dim * in_chunks, bias=False),
                 in_chunks, sum_out=True), \
           nn.Sequential(Chunk(nn.Linear(net_dim, spec_dim * out_chunks, bias=False),
                               out_chunks, sum_out=True),
                         *((ParametrizedGaussian(spec_dim, spec_dim, False, False), ) if with_sampler else ()),
                         ScaledSigmoid(spec_dim, with_range=False) if scaled_activation else Abs())


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
                (nn.Sequential(Chunk(nn.Linear(net_dim, spec_dim * out_chunks, bias=False), out_chunks, sum_out=True), act_phs),
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
               (nn.Sequential(Chunk(nn.Linear(net_dim, spec_dim * out_chunks, bias=False), out_chunks, sum_out=True), act_mag),
                'x -> mag'),
               (lambda mag, phs: torch.stack((mag, phs), dim=-1), "mag, phs -> y")
               )
