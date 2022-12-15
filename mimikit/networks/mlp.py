from typing import Optional

import torch.nn as nn
import torch


__all__ = [
    "MLP"
]


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
            learn_temperature: bool = True,
    ):
        super(MLP, self).__init__()
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim + int(learn_temperature)
        self.n_hidden_layers = n_hidden_layers
        self.activation = activation
        self.bias = bias
        self.dropout = dropout
        self.dropout1d = dropout1d
        self.learn_temperature = learn_temperature

        assert not (dropout is not None and dropout1d is not None), "only one of dropout and dropout1d can be a float"
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
            *fc, nn.Linear(hidden_dim, self.out_dim, bias=bias)
        )
        if self.learn_temperature:
            self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor):
        logits = self.fc(x)
        if self.learn_temperature:
            logits = logits[..., :-1] / self.sigmoid(logits[..., -1:])
        return logits
