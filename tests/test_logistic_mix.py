from mimikit.modules.logistic_mix import *
import pytest
import torch
import matplotlib.pyplot as plt


def test_nothing():
    B, C, T, num_classes = 1, 3, 64, 32
    y_hat, y = torch.zeros(B, C, T), torch.linspace(-1, 1, B*T).view(B, T, 1)
    y_hat[:, 1] = torch.linspace(1., -1, T).expand_as(y_hat[:, 1])
    y_hat[:, 2] = .1
    probs = discretized_mix_logistic_loss(y_hat, y, num_classes, reduce=False)

    plt.figure()
    plt.plot(y_hat[:, 1].squeeze().detach().cpu().numpy())
    plt.plot(y[0].squeeze().detach().cpu().numpy())
    plt.plot((-probs).squeeze().detach().cpu().numpy())
    plt.show()
