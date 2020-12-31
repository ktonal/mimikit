import torch.nn as nn
from .gated_units import GatedLinear


class PermuteTF(nn.Module):
    def forward(self, x):
        return x.transpose(1, 2).contiguous()


class GatedLinearInput(GatedLinear):
    def __init__(self, in_dim, out_dim, permute_tf=True):
        super(GatedLinearInput, self).__init__(in_dim, out_dim)
        self.post = PermuteTF() if permute_tf else nn.Identity()

    def forward(self, x):
        x = super(GatedLinearInput, self).forward(x)
        return self.post(x)


class AbsLinearOutput(nn.Module):
    def __init__(self, in_dim, out_dim, permute_tf=True, **kwargs):
        super(AbsLinearOutput, self).__init__()
        self.fc = nn.Sequential(
            PermuteTF() if permute_tf else nn.Identity(),
            nn.Linear(in_dim, out_dim, **kwargs))

    def forward(self, x):
        return self.fc(x).abs()
