import torch.nn as nn
import torch
import torch.nn.functional as F
from scipy.signal import lfilter
from torch.autograd import Variable
from dataclasses import dataclass
from typing import Optional
import numpy as np


# from .wavenet import WNNetwork, WaveNetLayer
# from ..modules import homs as H, ops as Ops


def peakdetector(sig, att, rel):
    """
    Envelope follower
    Parameters
    ----------
    sig : array_like
        the input signal.  For audio signals the abs should be taken or the signal
        should be raised to the power of two
    att : float
        attack coefficient
    rel : float
        release coefficient
    """
    lev = 0.0
    outsig = np.zeros_like(sig)
    for i in range(len(sig)):
        x = sig[i]
        if x > lev:
            lev = lev + att * (x - lev)
        else:
            lev = lev + rel * (x - lev)
        outsig[i] = lev
    return outsig


def extract_env(S):
    mags = abs(S)
    env = 0.1 * peakdetector(np.sum(mags, 0), 0.75, 0.1)
    env_deriv = lfilter([3, 0, -3], [1], env)[2:]
    env = np.stack([env[1: -1], env_deriv])
    return env.T.astype(np.float32)


@dataclass(init=True, repr=False, eq=False, frozen=False, unsafe_hash=True)
class PhaseNetwork(nn.Module):
    input_dim: int = 513
    dim1x1: int = 64
    dim2x3: int = 64
    n_1x1layers: int = 3
    n_2x3layers: int = 2
    groups: int = 1

    def __post_init__(self):
        nn.Module.__init__(self)
        dim1x1 = self.dim1x1
        dim2x3 = self.dim2x3
        self.first_phslayer = nn.Conv2d(5, dim2x3 - 5, kernel_size=(3, 3), padding=(0, 1))
        self.phs_layers2x3 = nn.ModuleList(
            [nn.Conv2d(dim2x3, dim2x3, kernel_size=(2, 3), padding=(0, 1), groups=self.groups)
             for _ in range(self.n_2x3layers)])
        self.gate_layers2x3 = nn.ModuleList(
            [nn.Conv2d(dim2x3, dim2x3, kernel_size=(2, 3), padding=(0, 1), groups=self.groups)
             for _ in range(self.n_2x3layers)])
        phs_layers1x1 = [nn.Conv2d(dim2x3, dim1x1, kernel_size=(1, 1), groups=self.groups)]
        gate_layers1x1 = [nn.Conv2d(dim2x3, dim1x1, kernel_size=(1, 1), groups=self.groups)]
        for _ in range(self.n_1x1layers - 1):
            phs_layers1x1 += [nn.Conv2d(dim1x1, dim1x1, kernel_size=(1, 1), groups=self.groups)]
            gate_layers1x1 += [nn.Conv2d(dim1x1, dim1x1, kernel_size=(1, 1), groups=self.groups)]
        self.phs_layers1x1 = nn.ModuleList(phs_layers1x1)
        self.gate_layers1x1 = nn.ModuleList(gate_layers1x1)
        self.last_phslayer = nn.Conv2d(dim1x1, 1, kernel_size=(1, 1))
        self.center_adv = Variable(self.principarg(torch.from_numpy(np.arange(self.input_dim) * 2 * np.pi * 0.25)),
                                   requires_grad=False).float()

    def forward(self, x, predicted_mags):
        phs = x[:, 1:]
        shift = x.shape[2] - predicted_mags.shape[1]
        phs_net_shift = shift - 2 - self.n_2x3layers
        # phase gradients in frequency and time
        pgf = self.principarg(F.pad(x[:, 1:, phs_net_shift:, 2:] - x[:, 1:, phs_net_shift:, :-2],
                                     (1, 1, 0, 0), mode='reflect'))
        pgt = self.principarg(x[:, 1:, phs_net_shift - 1:-1] - x[:, 1:, phs_net_shift:])
        pgt = self.principarg(pgt - self.center_adv)  # demodulate
        log_mags = torch.cat([self.safe_log(x[:, 0:1, phs_net_shift:]),
                              self.safe_log(predicted_mags[:, -1:]).unsqueeze(1)], 2)
        # log magnitude gradients
        tgt = log_mags[:, :, 1:] - log_mags[:, :, :-1]
        log_mags = log_mags[:, :, 1:]
        tgf = F.pad(log_mags[:, :, :, 2:] - log_mags[:, :, :, :-2], (1, 1, 0, 0), mode='reflect')
        dphs = torch.cat([log_mags, tgf, tgt, pgf, pgt], 1)
        dphs = torch.cat([dphs[:, :, 2:], F.tanh(self.first_phslayer(dphs))], 1)
        for l, g in zip(self.phs_layers2x3, self.gate_layers2x3):
            dphs = torch.tanh(l(dphs)) * F.relu(g(dphs)) + dphs[:, :, 1:]
        dphs = torch.tanh(self.phs_layers1x1[0](dphs)) * F.relu(self.gate_layers1x1[0](dphs))
        dphs[:, :self.dim2x3] += dphs[:, :self.dim2x3]
        for l, g in zip(self.phs_layers1x1[1:], self.gate_layers1x1[1:]):
            dphs = torch.tanh(l(dphs)) * F.relu(g(dphs)) + dphs
        dphs = self.last_phslayer(dphs)
        return self.principarg(phs[:, :, shift:] + self.center_adv + dphs)

    @staticmethod
    def principarg(x):
        return x - 2.0 * np.pi * torch.round(x / (2.0 * np.pi))

    @staticmethod
    def safe_log(x):
        return torch.log(torch.maximum(x, torch.tensor(0.00001)))


WNNetwork = None


@dataclass(init=True, repr=False, eq=False, frozen=False, unsafe_hash=True)
class PocoNetNetwork(WNNetwork, nn.Module):
    n_layers: tuple = (4,)
    input_dim: int = 256
    n_cin_classes: Optional[int] = None
    cin_dim: Optional[int] = None
    n_gin_classes: Optional[int] = None
    gin_dim: Optional[int] = None
    gate_dim: int = 128
    kernel_size: int = 2
    groups: int = 1
    accum_outputs: int = 0
    pad_input: int = 0
    skip_dim: Optional[int] = None
    residuals_dim: Optional[int] = None
    dim1x1: int = 64
    dim2x3: int = 64
    n_1x1layers: int = 3
    n_2x3layers: int = 2
    phs_groups: int = 1
    amp_env_dim: int = 32
    amp_gate_dim: int = 256
    amp_env_layers: int = 1

    def inpt_(self):
        return H.Paths(
            nn.Sequential(
                H.GatedUnit(nn.Linear(self.input_dim, self.gate_dim)), Ops.Transpose(1, 2)),
            # conditioning parameters :
            nn.Sequential(
                nn.Embedding(self.n_cin_classes, self.cin_dim), Ops.Transpose(1, 2)) if self.cin_dim else None,
            nn.Sequential(
                nn.Embedding(self.n_gin_classes, self.gin_dim), Ops.Transpose(1, 2)) if self.gin_dim else None
        )

    def layers_(self):
        return nn.Sequential(*[
            WaveNetLayer(i,
                         gate_dim=self.gate_dim,
                         skip_dim=self.skip_dim,
                         residuals_dim=self.residuals_dim,
                         kernel_size=self.kernel_size,
                         cin_dim=self.cin_dim,
                         gin_dim=self.gin_dim,
                         groups=self.groups,
                         pad_input=self.pad_input,
                         accum_outputs=self.accum_outputs,
                         )
            for block in self.n_layers for i in range(block)
        ])

    def amp_env_(self):
        # amp env input to get an additive and a multiplicative tensor to modify the input to the net
        mul_net = [nn.Linear(2, self.amp_env_dim)]
        add_net = [nn.Linear(2, self.amp_env_dim)]
        for _ in range(self.amp_env_layers):
            mul_net += [nn.Linear(self.amp_env_dim, self.amp_env_dim), nn.ReLU()]
            add_net += [nn.Linear(self.amp_env_dim, self.amp_env_dim), nn.ReLU()]
        mul_net += [nn.Linear(self.amp_env_dim, self.amp_gate_dim), nn.ReLU(), Ops.Transpose(1, 2)]
        add_net += [nn.Linear(self.amp_env_dim, self.amp_gate_dim), nn.ReLU(), Ops.Transpose(1, 2)]

        return H.Paths(
            nn.Sequential(*mul_net),
            nn.Sequential(*add_net)
        )

    def outpt_(self):
        return nn.Sequential(
            Ops.Transpose(1, 2),
            nn.Linear(self.gate_dim if self.skip_dim is None else self.skip_dim, self.input_dim), Ops.Abs()
        )

    def __post_init__(self):
        WNNetwork.__post_init__(self)
        self.phs_network = PhaseNetwork(input_dim=self.input_dim, dim1x1=self.dim1x1, dim2x3=self.dim2x3,
                                        n_1x1layers=self.n_1x1layers, n_2x3layers=self.n_2x3layers,
                                        groups=self.phs_groups)
        self.amp_env = self.amp_env_()

    def forward(self, xi, env, cin=None, gin=None):
        x, cin, gin = self.inpt(xi[:, 0], cin, gin)
        aa, am = self.amp_env(env, env)
        xm = torch.cat([am * x[:, :self.amp_gate_dim] + aa, x[:, self.amp_gate_dim:]], 1)
        y, _, _, skips = self.layers((xm, cin, gin, None))
        predicted_mags = self.outpt(skips if skips is not None else y)
        phs = self.phs_network(xi, predicted_mags)
        return torch.cat([predicted_mags.unsqueeze(1), phs], 1)

    def cuda(self, device=None):
        self = super().cuda(device)
        self.phs_network.center_adv = self.phs_network.center_adv.cuda(device)
        return self

    def to(self, *args, **kwargs):
        self = super().to(*args, **kwargs)
        self.phs_network.center_adv = self.phs_network.center_adv.to(*args, **kwargs)
        return self


def l1_loss_with_phs(output, target):
    """PocoNet loss"""
    y = target
    x = output
    norm = target[:, 0].abs().sum(dim=(0, -1), keepdim=True)
    cd = (torch.cos(y[:, 1]) - torch.cos(x[:, 1]))
    sd = (torch.sin(y[:, 1]) - torch.sin(x[:, 1]))
    phserr = torch.mean(torch.norm(torch.stack((sd, cd)) * torch.sqrt((y[:, 0] / norm + 0.01)), dim=0))
    # phserr = torch.mean(cd*cd + sd*sd + 0.00001))
    L = nn.L1Loss(reduction="none")(output[:, 0], target[:, 0]).sum(dim=(0, -1), keepdim=True)
    mag_loss, phs_loss = 100 * (L / norm).mean(), 100 * phserr
    return {"loss": mag_loss+phs_loss, "mag_loss": mag_loss, "phs_loss": phs_loss}