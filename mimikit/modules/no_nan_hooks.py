import torch
import torch.nn as nn

import h5mapper as h5m

__all__ = [
    "NoNanHooks"
]


class NoNanHooks(nn.Module):

    def attach_hooks(self):

        def frw_hook(module, inpt, output):
            is_tensor = lambda x: isinstance(x, torch.Tensor)
            h5m.process_batch(inpt, is_tensor,
                              lambda x: print("INPUT NAN", module) if torch.any(torch.isnan(x)) else None
                              )
            h5m.process_batch(output, is_tensor,
                              lambda x: print("OUTPUT NAN", module) if torch.any(torch.isnan(x)) else None
                              )
            return output

        def grad_hook(module, inpt, output):
            is_tensor = lambda x: isinstance(x, torch.Tensor)
            h5m.process_batch(inpt, is_tensor,
                              lambda x: print("INPUT GRAD NAN", module) if torch.any(torch.isnan(x)) else None
                              )
            h5m.process_batch(output, is_tensor,
                              lambda x: print("OUTPUT GRAD NAN", module) if torch.any(torch.isnan(x)) else None
                              )
            return inpt

        for mod in self.modules():
            mod.register_full_backward_hook(grad_hook)
            mod.register_forward_hook(frw_hook)

        def hook(grad):
            is_nans = torch.isnan(grad)
            if torch.any(is_nans):
                return torch.where(is_nans, torch.zeros_like(grad), grad)
            return grad
        for p in self.parameters():
            p.register_hook(hook)