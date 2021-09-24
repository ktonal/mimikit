import torch
import torch.nn as nn
import numpy as np

from ..loops.callbacks import tqdm
from h5mapper import process_batch

__all__ = [
    'prepare_prompt',
    'generate_tqdm',
    "PromptsManager"
]


def prepare_prompt(device, prompt, n_steps, at_least_nd=2):
    def _prepare(prmpt):
        if isinstance(prmpt, np.ndarray):
            prmpt = torch.from_numpy(prmpt)
        while len(prmpt.shape) < at_least_nd:
            prmpt = prmpt.unsqueeze(0)
        prmpt = prmpt.to(device)
        blank_shapes = prmpt.size(0), n_steps, *prmpt.size()[2:]
        return torch.cat((prmpt, torch.zeros(*blank_shapes).to(prmpt)), dim=1)

    return process_batch(prompt, lambda x: isinstance(x, (np.ndarray, torch.Tensor)), _prepare)


def generate_tqdm(rng):
    return tqdm(rng, desc="Generate", dynamic_ncols=True,
                leave=False, unit="step", mininterval=0.25)


class PromptsManager:

    def __init__(self, prompts, dims_to_set):
        self.tensors = prompts
        self.dims = [tuple([slice(None)] * d) for d in dims_to_set]

    def __getitem__(self, item):
        return tuple(x[(*d, item)] for x, d in zip(self.tensors, self.dims))

    def __setitem__(self, item, values):
        for x, d, v in zip(self.tensors, self.dims, values):
            x.data[(*d, item)] = v
        return self
