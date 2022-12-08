from typing import Optional

import torch.nn as nn
import torch
import abc

from torch.types import Number

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

        self.fc_in = nn.Sequential(
            nn.Linear(in_dim, hidden_dim, bias=bias), self.activation,
            *((self.dp, ) if self.dp else ())
        )
        self.fc_hidden = nn.Sequential(
            *((nn.Linear(hidden_dim, hidden_dim, bias=bias), self.activation,
               *((self.dp, ) if self.dp else ())) * n_hidden_layers)
        )
        self.fc_out = nn.Linear(hidden_dim, out_dim, bias=bias)

    def forward(self, x: torch.Tensor):
        return self.fc_out(self.fc_hidden(self.fc_in(x)))


class AsRelaxedCategorical(nn.Module):

    def __init__(
            self,
            module: nn.Module,
            learn_temperature: bool = True,
    ):
        super(AsRelaxedCategorical, self).__init__()
        self.module = module
        self.learn_temperature = learn_temperature
        if learn_temperature:
            self.sigmoid = nn.Sigmoid()

    def forward(self, inputs: torch.Tensor, *, temperature=None):
        output = self.module(inputs)
        if self.learn_temperature:
            output = output[..., :-1] / (self.sigmoid(output[..., -1:]))
        if self.training:
            return output
        if temperature is None:
            return output.argmax(dim=-1)
        if not isinstance(temperature, torch.Tensor):
            if isinstance(temperature, torch.types.Number):
                temperature = [temperature]
            temperature = torch.tensor(temperature)
        if temperature.size() != output.size():
            temperature = temperature.view(*temperature.shape, *([1] * (output.ndim - temperature.ndim)))
        output = output / temperature.to(output.device)
        output = output - output.logsumexp(-1, keepdim=True)
        if output.dim() > 2:
            o_shape = output.shape
            output = output.view(-1, o_shape[-1])
            return torch.multinomial(output, 1).reshape(*o_shape[:-1], 1)
        return torch.multinomial(output, 1)


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
