import torch.nn as nn
import torch

__all__ = [
    "SingleClassMLP"
]


class SingleClassMLP(nn.Module):

    def __init__(self, d_in, d_mid, d_out, act=nn.ReLU()):
        super(SingleClassMLP, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(d_in, d_mid), act,
            nn.Linear(d_mid, d_mid), act,
            nn.Linear(d_mid, d_out),
        )

    def forward(self, x, temperature=None):
        if self.training:
            return self.mlp(x)
        outpt = self.mlp(x)
        if temperature is None:
            return nn.Softmax(dim=-1)(outpt).argmax(dim=-1, keepdims=True)
        else:
            return torch.multinomial(nn.Softmax(dim=-1)(outpt / temperature.to(outpt)), 1)
