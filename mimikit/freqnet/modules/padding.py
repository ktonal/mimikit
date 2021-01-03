import torch
from torch import nn as nn


class LearnablePad1d(nn.Module):
    def __init__(self, input_dim, amount, learn_padding=False):
        super(LearnablePad1d, self).__init__()
        self.pad = None
        self.side = torch.sign(torch.tensor(amount)).item()
        amount = abs(amount)
        if amount > 0:
            if learn_padding:
                self.pad = nn.Parameter(torch.zeros(input_dim, amount), requires_grad=True)
            else:
                self.pad = torch.zeros(input_dim, amount)

    def forward(self, x):
        if self.pad is None:
            return x
        self.pad = self.pad.to(x)
        return torch.cat((torch.stack([self.pad] * x.size(0)), x)[::self.side], dim=-1)


    def receptive_field(self):
        shifts = []
        for i, layer in enumerate(self.layers):
            if shifts[-1] > layer.shift():
                shifts += [shifts[-1] + layer.shift()]
            else:
                shifts += [layer.shift()]
        return sum(shifts)