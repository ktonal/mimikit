import torch.nn as nn
import torch

from ..modules.homs import HOM

__all__ = [
    "SingleClassMLP"
]


class SingleClassMLP(nn.Module):

    def __init__(self, d_in, d_mid, d_out,
                 act=nn.Mish(), bias=True,
                 learn_temperature=True, n_hidden_layers=0,
                 net_type="mlp"  # one of ['mlp', 'highway', 'residuals']
                 ):
        super(SingleClassMLP, self).__init__()
        self.learn_temp = learn_temperature
        if net_type == 'mlp':
            self.mlp = nn.Sequential(
                nn.Linear(d_in, d_mid, bias=bias), act,
                *((nn.Linear(d_mid, d_mid, bias=bias), act) * n_hidden_layers),
                nn.Linear(d_mid, d_out + int(learn_temperature), bias=bias)
            )
        elif net_type == 'highway':
            self.mlp = HOM("x -> y",
                           (nn.Sequential(nn.Linear(d_in, d_mid, bias=bias), act, ), "x -> x"),
                           (lambda x, y: torch.cat((x, y), dim=-1), "x, x -> xh"),
                           *(
                                (nn.Sequential(nn.Linear(d_mid * 2, d_mid, bias=bias), act), "xh -> y"),
                                (lambda x, y: torch.cat((x, y), dim=-1), "x, y -> xh"),
                            ) * n_hidden_layers,
                           (nn.Linear(d_mid, d_out + int(learn_temperature), bias=bias), "y -> y"),
                           )
        elif net_type == 'residuals':
            self.mlp = HOM("x -> y",
                           (nn.Sequential(nn.Linear(d_in, d_mid, bias=bias), act, ), "x -> xh"),
                           *(
                                (nn.Sequential(nn.Linear(d_mid, d_mid, bias=bias), act), "xh -> y"),
                                (lambda x, y: x + y, "xh, y -> xh"),
                            ) * n_hidden_layers,
                           (nn.Linear(d_mid, d_out + int(learn_temperature), bias=bias), "xh -> y"),
                           )
        else:
            raise ValueError(f"Expected `net_type` to be one of ['mlp', 'highway', 'residuals']. Got {net_type}")
        self.d_out = d_out

    def forward(self, x, temperature=None):
        outpt = self.mlp(x)
        if self.learn_temp:
            # last output feature is the learned temperature
            outpt = outpt[..., :-1] / (nn.Sigmoid()(outpt[..., -1:]))
        if self.training:
            return outpt
        if temperature is None:
            return outpt.argmax(dim=-1)
        else:
            if not isinstance(temperature, torch.Tensor):
                temperature = torch.Tensor([temperature]).reshape(*([1] * (len(outpt.size()))))
            probas = outpt.squeeze() / temperature.to(outpt)
            probas = nn.Softmax(dim=-1)(probas)
            if probas.dim() > 2:
                o_shape = probas.shape
                probas = probas.view(-1, o_shape[-1])
                return torch.multinomial(probas, 1).reshape(*o_shape[:-1])
            return torch.multinomial(probas, 1)
