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
            dropout: float = 0.,
            dropout1d: float = 0.,
            min_temperature: Optional[float] = 1e-4,
    ):
        super(MLP, self).__init__()
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim + int(min_temperature is not None)
        self.n_hidden_layers = n_hidden_layers
        self.activation = activation
        self.bias = bias
        self.dropout = dropout
        self.dropout1d = dropout1d
        self.learn_temperature = min_temperature is not None

        dp = []
        if dropout > 0.:
            dp += [nn.Dropout(dropout)]
        if dropout1d > 0.:
            dp += [nn.Dropout1d(dropout1d)]

        fc = [
            nn.Linear(in_dim, hidden_dim, bias=bias), self.activation,
            *dp
        ]
        fc += [
            *((nn.Linear(hidden_dim, hidden_dim, bias=bias), self.activation,
               *dp) * n_hidden_layers)
        ]
        self.fc = nn.Sequential(
            *fc, nn.Linear(hidden_dim, self.out_dim, bias=bias)
        )
        if self.learn_temperature:
            self.sigmoid = nn.Sigmoid()
            self.register_buffer("min_temp", torch.tensor(min_temperature))

    def forward(self, x: torch.Tensor):
        logits = self.fc(x)
        if self.learn_temperature:
            temp = self.sigmoid(logits[..., -1:])
            logits = logits[..., :-1] / torch.maximum(temp, self.min_temp)
        return logits
