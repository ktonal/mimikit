import torch.nn as nn


class Conditioned(nn.Module):

    def __init__(self, mod, *conds):
        super().__init__()
        self.mod = mod
        self.conds = nn.ModuleList([c for c in conds if c])

    def forward(self, x, *conditions):
        # all output shapes have to be the same!...
        conditions = (c for c in conditions if c is not None)
        return self.mod(x) + sum(mod(c) for mod, c in zip(self.conds, conditions))


class GatedLinearUnit(nn.Module):

    def __init__(self, in_dim, out_dim, local_cond_dim=None, global_cond_dim=None, **kwargs):
        super(GatedLinearUnit, self).__init__()
        self.fcg = Conditioned(nn.Linear(in_dim, out_dim, **kwargs),
                               # conditioning parameters :
                               nn.Linear(local_cond_dim, out_dim, **kwargs) if local_cond_dim else None,
                               nn.Linear(global_cond_dim, out_dim, **kwargs) if global_cond_dim else None)
        self.fcf = Conditioned(nn.Linear(in_dim, out_dim, **kwargs),
                               nn.Linear(local_cond_dim, out_dim, **kwargs) if local_cond_dim else None,
                               nn.Linear(global_cond_dim, out_dim, **kwargs) if global_cond_dim else None)
        self.tanh = nn.Tanh()
        self.sig = nn.Sigmoid()

    def forward(self, x, local_c=None, global_c=None):
        f = self.fcf(x, local_c, global_c)
        g = self.fcg(x, local_c, global_c)
        return self.tanh(f) * self.sig(g)


class GatedConv1d(nn.Module):

    def __init__(self, c_in, c_out, kernel_size=2, transpose=False,
                 local_cond_dim=None, global_cond_dim=None, **kwargs):
        super(GatedConv1d, self).__init__()
        mod = nn.Conv1d if not transpose else nn.ConvTranspose1d
        self.conv_f = Conditioned(
            mod(c_in, c_out, kernel_size, **kwargs),
            # conditioning parameters :
            nn.Conv1d(local_cond_dim, c_out, kernel_size=1, **kwargs) if local_cond_dim else None,
            nn.Conv1d(global_cond_dim, c_out, kernel_size=1, **kwargs) if global_cond_dim else None,
        )
        self.conv_g = Conditioned(
            mod(c_in, c_out, kernel_size, **kwargs),
            nn.Conv1d(local_cond_dim, c_out, kernel_size=1, **kwargs) if local_cond_dim else None,
            nn.Conv1d(global_cond_dim, c_out, kernel_size=1,
                      **kwargs) if global_cond_dim else None,
        )
        self.tanh = nn.Tanh()
        self.sig = nn.Sigmoid()

    def forward(self, x, local_c=None, global_c=None):
        f = self.conv_f(x, local_c, global_c)
        g = self.conv_g(x, local_c, global_c)
        return self.tanh(f) * self.sig(g)
