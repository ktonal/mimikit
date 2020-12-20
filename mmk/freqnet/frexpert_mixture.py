import torch
import torch.nn as nn

from .modules.loss_functions import mean_L1_prop
from .base import FreqNetModel
from .freqnet import FreqNet


class FrexpertMixture(FreqNetModel):
    """
    Use the Base FreqNet class to load a bunch of FreqNets for a mixture
    """

    def __init__(self, loss_fn=mean_L1_prop, **data_optim_kwargs):
        super(FrexpertMixture, self).__init__(**data_optim_kwargs)
        self.loss_fn = loss_fn
        self.save_hyperparameters()
        data_obj = data_optim_kwargs.get("data_object", None)
        # we use nets instead of layers, but we still store them in the layers attributes!
        self.nets = nn.ModuleList([
            FreqNet(data_object=data_obj,
                    model_dim=512,
                    groups=1,
                    n_layers=(3,),
                    strict=False,
                    accum_outputs=1),
            FreqNet(data_object=data_obj,
                    model_dim=1024,
                    groups=2,
                    n_layers=(3,),
                    strict=False,
                    accum_outputs=-1)
        ])
        # a simple MLP to output the weight of each net given the input
        self.proba_f = nn.Sequential(nn.Linear(1025, 1025), nn.Tanh(),
                                     nn.Linear(1025, 2), nn.Softmax(dim=-1))

    def forward(self, x):
        ys = torch.stack([net(x) for net in self.nets], dim=2)
        # we crop the begining of x to have matching shapes
        pr_ys = self.proba_f(x[:, -ys.size(1):]).unsqueeze(-1)
        # sum by the "net-dim"
        y = (pr_ys * ys).sum(dim=2)
        return y

    def targets_shifts_and_lengths(self, input_length):
        return self.nets[0].targets_shifts_and_lengths(input_length)