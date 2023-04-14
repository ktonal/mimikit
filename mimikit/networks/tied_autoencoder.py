from typing import Tuple, Optional, Set, Dict

import h5mapper as h5m
import torch.nn as nn
import torch
import torch.nn.functional as F
import dataclasses as dtc

__all__ = [
    "TiedAE"
]

from .arm import ARM, NetworkConfig
from ..features.item_spec import ItemSpec
from ..io_spec import IOSpec


class TiedAE(ARM, nn.Module):

    @dtc.dataclass
    class Config(NetworkConfig):
        io_spec: IOSpec = None
        kernel_sizes: Tuple[int, ...] = (3,)
        dims: Tuple[int, ...] = (16,)
        non_negative_latent: bool = False
        independence_reg: Optional[float] = None

    @classmethod
    def from_config(cls, config: Config):
        input_dim = config.io_spec.inputs[0].module.out_dim
        weights = [nn.Conv1d(d_in, d_out, k, bias=False).weight
                   for (d_in, d_out), k in
                   zip((input_dim, *config.dims[:-1]), config.dims, config.kernel_sizes)]
        return cls(config, *weights)

    def __init__(self, config, *cv_weights):
        super(TiedAE, self).__init__()
        self._config = config
        self.padding = [k // 2 for k in config.kernel_sizes]
        self.weights = nn.ParameterList(cv_weights)
        self.permute = lambda x: x.transpose(1, 2)

    def forward(self, x):
        x = self.permute(x)
        indp = 0
        for w, p in zip(self.weights, self.padding):
            x = F.conv1d(x, w, padding=p)
            if self._config.non_negative_latent:
                x = x.abs()
            # x = F.conv1d(x, w, padding=p).abs()
        for w, p in zip(reversed(self.weights), reversed(self.padding)):
            x = F.conv_transpose1d(x, w, padding=p)
            if self._config.independence_reg is not None:
                wwt = torch.matmul(w.sum(dim=2), w.sum(dim=2).t())
                indp += F.l1_loss(wwt, torch.eye(wwt.size(0)).to(wwt))

        x = self.permute(x)
        # x = self.permute(x).abs()
        return x, indp

    @property
    def config(self) -> NetworkConfig:
        return self._config

    @property
    def rf(self):
        return 0

    def train_batch(self, item_spec: ItemSpec) -> Tuple[Tuple[h5m.Input, ...], Tuple[h5m.Input, ...]]:
        pass

    def test_batch(self, item_spec: ItemSpec) -> Tuple[Tuple[h5m.Input, ...], Tuple[h5m.Input, ...]]:
        pass

    def before_generate(self, prompts: Tuple[torch.Tensor, ...], batch_index: int) -> None:
        pass

    def generate_step(self, inputs: Tuple[torch.Tensor, ...], *, t: int = 0, **parameters: Dict[str, torch.Tensor]) -> \
    Tuple[torch.Tensor, ...]:
        pass

    def after_generate(self, final_outputs: Tuple[torch.Tensor, ...], batch_index: int) -> None:
        pass

    @property
    def generate_params(self) -> Set[str]:
        return set()