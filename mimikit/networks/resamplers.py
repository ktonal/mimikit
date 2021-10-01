import torch.nn as nn
import numpy as np
from functools import partial

__all__ = [
    'LinearResampler',
    'Conv1dResampler',
    'LSTMResampler'
]


class LinearResampler(nn.Module):
    def __init__(self, in_d, t_factor, d_factor, **kwargs):
        super().__init__()

        self.fc = nn.Linear(in_d, int(in_d * t_factor * d_factor), **kwargs)
        self.tf, self.df = t_factor, d_factor

    def forward(self, x):
        B, T, D = x.size()
        x = self.fc(x)
        return x.reshape(B, int(T * self.tf), int(D * self.df))


class Conv1dResampler(nn.Module):

    def __init__(self, in_d, t_factor, d_factor, **kwargs):
        super().__init__()
        mod = nn.Conv1d if t_factor <= 1 else partial(nn.ConvTranspose1d, stride=t_factor)
        self.cv = mod(in_d, int(in_d * d_factor),
                      t_factor if t_factor >= 1 else int(1 / t_factor),
                      **kwargs)
        self.tf, self.df = t_factor, d_factor

    def forward(self, x):
        if len(x.size()) > 3:
            x = x.view(x.size(0), np.prod(x.shape[1:-1]), x.size(-1))
        B, T, D = x.size()
        if self.tf <= 1:
            x = x.view(-1, int(1 / self.tf), D).transpose(1, 2)
            x = self.cv(x).squeeze(-1).reshape(B, int(D * self.df), -1)
        else:
            x = x.transpose(1, 2)
            x = self.cv(x)
        return x.transpose(1, 2)


class LSTMResampler(nn.Module):

    def __init__(self, in_d, t_factor, d_factor, **kwargs):
        super().__init__()
        self.lstm = nn.LSTM(in_d, int(in_d * t_factor * d_factor), batch_first=True, **kwargs)
        self.tf, self.df = t_factor, d_factor

    def forward(self, x, hidden=None):
        B, T, D = x.size()
        x, hidden = self.lstm(x, hidden)
        return x.reshape(B, int(T * self.tf), int(D * self.df)), hidden