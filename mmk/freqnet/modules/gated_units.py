import torch.nn as nn


class GatedLinear(nn.Module):
    def __init__(self, in_dim, out_dim, **kwargs):
        super(GatedLinear, self).__init__()
        self.fcg = nn.Linear(in_dim, out_dim, **kwargs)
        self.fcf = nn.Linear(in_dim, out_dim, **kwargs)

    def forward(self, x):
        f = nn.Tanh()(self.fcf(x))
        g = nn.Sigmoid()(self.fcg(x))
        return f * g


class GatedConv(nn.Module):
    def __init__(self, c_in, c_out, kernel_size=2, transpose=False, **kwargs):
        super(GatedConv, self).__init__()
        mod = nn.Conv1d if not transpose else nn.ConvTranspose1d
        self.conv_f = mod(c_in, c_out, kernel_size, **kwargs)
        self.conv_g = mod(c_in, c_out, kernel_size, **kwargs)
        self.tanh = nn.Tanh()
        self.sig = nn.Sigmoid()

    def forward(self, x):
        f = self.tanh(self.conv_f(x))
        g = self.sig(self.conv_g(x))
        return f * g
