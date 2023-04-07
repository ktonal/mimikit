from typing import Tuple, Dict, Set
import dataclasses as dtc

import pytest
import torch
from assertpy import assert_that

import numpy as np
import h5mapper as h5m
from torch import nn

from mimikit import ARM, IOSpec
from mimikit.networks.arm import NetworkConfig

__all__ = [
    "TestDB",
    "tmp_db",
    "TestARM",
]

from mimikit.features.item_spec import ItemSpec


class RandSignal(h5m.Feature):

    def load(self, source):
        return (np.random.rand(32000) * 2 - 1).astype(np.float32)


class RandLabel(h5m.Feature):

    def load(self, source):
        return np.random.randint(0, 256, (32000,))


class TestDB(h5m.TypedFile):
    signal = RandSignal()
    label = RandLabel()


@pytest.fixture
def tmp_db(tmp_path):
    root = (tmp_path / "dbs")
    root.mkdir()

    def create_func(filename) -> TestDB:
        TestDB.create(
            str(root / filename),
            sources=tuple(map(str, range(2))),
            mode="w", keep_open=False, parallelism='none'
        )
        return TestDB(str(root / filename))

    return create_func


def test_fixture_db(tmp_db):
    db = tmp_db("temp")

    assert_that(db.signal).is_not_none()
    assert_that(db.signal[:32]).is_instance_of(np.ndarray)


class TestARM(ARM, nn.Module):
    @dtc.dataclass
    class Config(NetworkConfig):
        io_spec: IOSpec = None

    @property
    def config(self) -> NetworkConfig:
        return self._config

    @property
    def rf(self):
        return 8

    def train_batch(self, item_spec: ItemSpec) -> \
            Tuple[Tuple[h5m.Input, ...], Tuple[h5m.Input, ...]]:
        return tuple(
            feat.to_batch_item(item_spec)
            for feat in self.config.io_spec.inputs
        ), tuple(
            feat.to_batch_item(item_spec)
            for feat in self.config.io_spec.targets
        )

    def test_batch(self, item_spec: ItemSpec) ->\
            Tuple[Tuple[h5m.Input, ...], Tuple[h5m.Input, ...]]:
        return self.train_batch(item_spec)

    @property
    def generate_params(self) -> Set[str]:
        return set()

    def before_generate(self, prompts: Tuple[torch.Tensor, ...], batch_index: int) -> None:
        return

    def generate_step(self, inputs: Tuple[torch.Tensor, ...], *, t: int = 0, **parameters: Dict[str, torch.Tensor]) -> \
            Tuple[torch.Tensor, ...]:
        return tuple(self.forward(i) for i in inputs)

    def after_generate(self, final_outputs: Tuple[torch.Tensor, ...], batch_index: int) -> None:
        return

    @classmethod
    def from_config(cls, config: NetworkConfig):
        return cls(config)

    def __init__(self, config: NetworkConfig):
        super(TestARM, self).__init__()
        self._config = config
        self.fc = nn.Linear(1, 1)

    def forward(self, inputs):
        if self.training:
            if isinstance(inputs, (tuple, list)):
                return tuple(self.fc(x.unsqueeze(-1)).squeeze() for x in inputs)
            return self.fc(inputs.unsqueeze(-1)).squeeze()
        else:
            if isinstance(inputs, (tuple, list)):
                return tuple(x[:, -1:] for x in inputs)
            return inputs[:, -1:]
