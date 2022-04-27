import torch.nn as nn
import torch

__all__ = [
    "SingleClassMLP"
]


class SingleClassMLP(nn.Module):

    def __init__(self, d_in, d_mid, d_out, act=nn.Mish(), bias=True, learn_temperature=True):
        super(SingleClassMLP, self).__init__()
        self.learn_temp = learn_temperature
        self.mlp = nn.Sequential(
            nn.Linear(d_in, d_mid, bias=bias), act,
            # nn.Linear(d_mid, d_mid, bias=bias), act,
            nn.Linear(d_mid, d_out + int(learn_temperature), bias=bias),
        )

    def forward(self, x, temperature=None):
        outpt = self.mlp(x)
        if self.learn_temp:
        # last output feature is the learned temperature
            outpt = outpt[..., :-1] / nn.Sigmoid()(outpt[..., -1:])
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
