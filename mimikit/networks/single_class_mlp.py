from typing import Optional, Tuple, Literal

import torch.nn as nn
import torch
from torch import distributions as D

from ..modules.homs import HOM

__all__ = [
    "SingleClassMLP"
]

"""
mod(h) -> Categorical (temperature=...)
mod(h) -> (Mixture) Logistic
mod(h) -> (Mixture) Gaussian
mod(h) -> (Continuous) Bernoulli(s) 
-----> L = - distribution.log_prob(target)
-----> prediction = distribution.sample()

activation( mod( h ) ) -> Y / Envelope
activation( mod( h ) ) -> (complex) FFT / MelSpec / MFCC / DCT
----> L = reconstruction(output, target)
----> prediction = output


"""


class MLP(nn.Module):

    def __init__(
            self,
            in_dim: int,
            hidden_dim: int,
            out_dim: int,
            n_hidden_layers: int = 0,
            activation: nn.Module = nn.Mish(),
            bias: bool = True,
            dropout: Optional[float] = None,
            dropout1d: Optional[float] = None,
    ):
        super(MLP, self).__init__()
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.n_hidden_layers = n_hidden_layers
        self.activation = activation
        self.bias = bias
        self.dropout = dropout
        self.dropout1d = dropout1d

        assert not (dropout is not None and dropout1d is not None), "only on of dropout and dropout1d can be a float"
        if dropout is not None:
            self.dp = nn.Dropout(dropout)
        elif dropout1d is not None:
            # TODO: torch>=1.12
            self.dp = nn.Dropout1d(dropout1d)
        else:
            self.dp = None

        fc = [
            nn.Linear(in_dim, hidden_dim, bias=bias), self.activation,
            *((self.dp, ) if self.dp else ())
        ]
        fc += [
            *((nn.Linear(hidden_dim, hidden_dim, bias=bias), self.activation,
               *((self.dp, ) if self.dp else ())) * n_hidden_layers)
        ]
        self.fc = nn.Sequential(
            *fc, nn.Linear(hidden_dim, out_dim, bias=bias)
        )

    def forward(self, x: torch.Tensor):
        return self.fc(x)


class RelaxedCategorical(nn.Module):

    def __init__(
            self,
            learn_temperature: bool = True,
    ):
        super(RelaxedCategorical, self).__init__()
        self.learn_temperature = learn_temperature
        if learn_temperature:
            self.sigmoid = nn.Sigmoid()

    def forward(self,
                params: Tuple[torch.Tensor, Optional[torch.Tensor]],
                *, temperature=None):
        logits, temp = params
        if self.learn_temperature and temp is not None:
            logits = logits / (self.sigmoid(temp))
        if self.training:
            return logits
        if temperature is None:
            return logits.argmax(dim=-1)
        if not isinstance(temperature, torch.Tensor):
            if isinstance(temperature, torch.types.Number):
                temperature = [temperature]
            temperature = torch.tensor(temperature)
        if temperature.ndim != logits.ndim:
            temperature = temperature.view(*temperature.shape, *([1] * (logits.ndim - temperature.ndim)))
        logits = logits / temperature.to(logits.device)
        logits = logits - logits.logsumexp(-1, keepdim=True)
        if logits.dim() > 2:
            o_shape = logits.shape
            logits = logits.view(-1, o_shape[-1])
            return torch.multinomial(logits, 1).reshape(*o_shape[:-1], 1)
        return torch.multinomial(logits, 1)


def Logistic(loc, scale) -> D.TransformedDistribution:
    """credits to https://github.com/pytorch/pytorch/issues/7857"""
    return D.TransformedDistribution(
        D.Uniform(0, 1),
        [D.SigmoidTransform().inv, D.AffineTransform(loc, scale)]
    )


class MixtureOfLogistics(nn.Module):
    # TODO: LossModule & InferModule
    def __init__(
            self,
            k_components: int,
            reduction: Literal["sum", "mean", "none"],
            clamp_samples: Tuple[float, float] = (-1., 1.)
    ):
        super(MixtureOfLogistics, self).__init__()
        self.k_components = k_components
        self.reduction = reduction
        self.clamp = clamp_samples

    def forward(self,
                params: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
                targets: Optional[torch.Tensor] = None
                ):
        weight, loc, scale = params
        o_shape, K = loc.shape, self.k_components
        assert weight.size(-1) == loc.size(-1) == scale.size(-1) == K
        weight, loc, scale = weight.view(-1, K), loc.view(-1, K), scale.view(-1, K)
        mixture = D.MixtureSameFamily(
            D.Categorical(logits=weight), Logistic(loc, scale)
        )
        if targets is not None:
            probs = mixture.log_prob(targets.view(-1))
            if self.reduction != "none":
                return getattr(torch, self.reduction)(probs)
            return probs
        return mixture.sample((1,)).clamp(*self.clamp).reshape(*o_shape[:-1])



class SingleClassMLP(nn.Module):

    def __init__(self, d_in, d_mid, d_out,
                 act=nn.Mish(), bias=True,
                 learn_temperature=True, n_hidden_layers=0,
                 net_type="mlp"  # one of ['mlp', 'highway', 'residuals']
                 ):
        super(SingleClassMLP, self).__init__()
        self.learn_temp = learn_temperature
        if net_type == 'mlp':
            self.mlp = nn.Sequential(
                nn.Linear(d_in, d_mid, bias=bias), act,
                *((nn.Linear(d_mid, d_mid, bias=bias), act) * n_hidden_layers),
                nn.Linear(d_mid, d_out + int(learn_temperature), bias=bias)
            )
        elif net_type == 'highway':
            self.mlp = HOM("x -> y",
                           (nn.Sequential(nn.Linear(d_in, d_mid, bias=bias), act, ), "x -> x"),
                           (lambda x, y: torch.cat((x, y), dim=-1), "x, x -> xh"),
                           *(
                                (nn.Sequential(nn.Linear(d_mid * 2, d_mid, bias=bias), act), "xh -> y"),
                                (lambda x, y: torch.cat((x, y), dim=-1), "x, y -> xh"),
                            ) * n_hidden_layers,
                           (nn.Linear(d_mid, d_out + int(learn_temperature), bias=bias), "y -> y"),
                           )
        elif net_type == 'residuals':
            self.mlp = HOM("x -> y",
                           (nn.Sequential(nn.Linear(d_in, d_mid, bias=bias), act, ), "x -> xh"),
                           *(
                                (nn.Sequential(nn.Linear(d_mid, d_mid, bias=bias), act), "xh -> y"),
                                (lambda x, y: x + y, "xh, y -> xh"),
                            ) * n_hidden_layers,
                           (nn.Linear(d_mid, d_out + int(learn_temperature), bias=bias), "xh -> y"),
                           )
        else:
            raise ValueError(f"Expected `net_type` to be one of ['mlp', 'highway', 'residuals']. Got {net_type}")
        self.d_out = d_out

    def forward(self, x, temperature=None):
        outpt = self.mlp(x)
        if self.learn_temp:
            # last output feature is the learned temperature
            outpt = outpt[..., :-1] / (nn.Sigmoid()(outpt[..., -1:]))
        if self.training:
            return outpt
        if temperature is None:
            return outpt.argmax(dim=-1)
        else:
            if not isinstance(temperature, torch.Tensor):
                temperature = torch.Tensor([temperature]).reshape(*([1] * (len(outpt.size()))))
            probas = outpt.squeeze() / temperature.to(outpt)
            probas = nn.Softmax(dim=-1)(probas)
            if probas.dim() > 2:
                o_shape = probas.shape
                probas = probas.view(-1, o_shape[-1])
                return torch.multinomial(probas, 1).reshape(*o_shape[:-1])
            return torch.multinomial(probas, 1)
