import torch.nn as nn
import torch
import torch.nn.functional as F

__all__ = [
    "TiedAE"
]


class TiedAE(nn.Module):

    def __init__(self, **kwargs):
        super(TiedAE, self).__init__()

        B, K = 8, 4

        self.inpt = nn.Conv1d(1025, B, K, padding=1, bias=False)
        self.permute = lambda x: x.transpose(1, 2)

    def forward(self, x):
        x = self.permute(x)

        x = F.conv1d(x, self.inpt.weight, padding=1).abs()
        wwt = torch.matmul(self.inpt.weight.sum(dim=2), self.inpt.weight.sum(dim=2).t())
        sim2 = 1 * F.l1_loss(wwt,
                             torch.eye(wwt.size(0)).to(wwt) * 1)

        x = F.conv_transpose1d(x, self.inpt.weight, padding=1)
        x = self.permute(x).abs()
        return x, 0 + sim2
