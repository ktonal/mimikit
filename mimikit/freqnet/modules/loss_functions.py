from torch import nn as nn


def mean_L1_prop(output, target):
    L = nn.L1Loss(reduction="none")(output, target).sum(dim=(0, -1), keepdim=True)
    return 100 * (L / target.abs().sum(dim=(0, -1), keepdim=True)).mean()