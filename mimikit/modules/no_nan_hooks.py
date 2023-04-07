import torch
import torch.nn as nn

import h5mapper as h5m

__all__ = [
    "no_nan_hooks"
]


def raise_on_nan(msg):
    def _hook(x):
        if torch.any(torch.isnan(x)):
            raise RuntimeError(msg)
    return _hook


def no_nan_hooks(network):

    def frw_hook(module, inpt, output):
        is_tensor = lambda x: isinstance(x, torch.Tensor)
        h5m.process_batch(inpt, is_tensor, raise_on_nan(f"input is NAN for {module}"))
        h5m.process_batch(output, is_tensor, raise_on_nan(f"output is NAN for {module}"))
        return output

    def grad_hook(module, inpt, output):
        is_tensor = lambda x: isinstance(x, torch.Tensor)
        h5m.process_batch(inpt, is_tensor, raise_on_nan(f"input GRAD is NAN for {module}"))
        h5m.process_batch(output, is_tensor,  raise_on_nan(f"output GRAD is NAN for {module}"))
        return inpt

    for mod in network.modules():
        mod.register_full_backward_hook(grad_hook)
        mod.register_forward_hook(frw_hook)
    #
    # def hook(grad):
    #     is_nans = torch.isnan(grad)
    #     if torch.any(is_nans):
    #         return torch.where(is_nans, torch.zeros_like(grad), grad)
    #     return grad
    #
    # for p in self.parameters():
    #     p.register_hook(hook)