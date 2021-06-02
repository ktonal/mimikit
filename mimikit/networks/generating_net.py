import torch
import torch.nn as nn
import numpy as np

from ..models.parts import tqdm
from ..data import process_batch

__all__ = [
    'GeneratingNetwork'
]


class GeneratingNetwork(nn.Module):

    def prepare_prompt(self, prompt, n_steps, at_least_nd=2):
        def _prepare(prmpt):
            if isinstance(prmpt, np.ndarray):
                prmpt = torch.from_numpy(prmpt)
            while len(prmpt.shape) < at_least_nd:
                prmpt = prmpt.unsqueeze(0)
            prmpt = prmpt.to(self.device)
            blank_shapes = prmpt.size(0), n_steps, *prmpt.size()[2:]
            return torch.cat((prmpt, torch.zeros(*blank_shapes).to(prmpt)), dim=1)

        return process_batch(prompt, lambda x: isinstance(x, (np.ndarray, torch.Tensor)), _prepare)

    @staticmethod
    def generate_tqdm(rng):
        return tqdm(rng, desc="Generate", dynamic_ncols=True,
                    leave=False, unit="step", mininterval=0.25)
