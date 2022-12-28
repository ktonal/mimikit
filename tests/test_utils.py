from typing import Tuple, Dict

import pytest
import torch
from assertpy import assert_that

import numpy as np
import h5mapper as h5m
from torch import nn

from mimikit import ARMConfig, ARM, TimeUnit, IOSpec


__all__ = [
    "TestDB",
    "tmp_db",
    "TestARM",
]


class RandSignal(h5m.Feature):

    def load(self, source):
        return (np.random.rand(32000) * 2 - 1).astype(np.float32)


class RandLabel(h5m.Feature):

    def load(self, source):
        return np.random.randint(0, 256, (32000, ))


class TestDB(h5m.TypedFile):
    snd = RandSignal()
    label = RandLabel()


@pytest.fixture
def tmp_db(tmp_path):
    root = (tmp_path / "dbs")
    root.mkdir()

    def create_func(filename) -> TestDB:
        TestDB.create(
            root / filename,
            sources=tuple(map(str, range(2))),
            mode="w", keep_open=False, parallelism='none'
        )
        return TestDB(root / filename)

    return create_func


def test_fixture_db(tmp_db):
    db = tmp_db("temp")

    assert_that(db.snd).is_not_none()
    assert_that(db.snd[:32]).is_instance_of(np.ndarray)


class TestARM(ARM, nn.Module):
    class Config(ARMConfig):
        io_spec: IOSpec = None

        def __init__(self, io_spec):
            self.io_spec = io_spec

    @property
    def config(self) -> ARMConfig:
        return self._config

    @property
    def rf(self):
        return 8

    def train_batch(self, length=1, unit=TimeUnit.step, downsampling=1) -> \
            Tuple[Tuple[h5m.Input, ...], Tuple[h5m.Input, ...]]:
        return tuple(
            feat.feature.copy()
                .batch_item(length, unit)
            for feat in self.config.io_spec.inputs
        ), \
            tuple(
                feat.feature.copy()
                    .batch_item(length, unit)
                for feat in self.config.io_spec.targets
            )

    def before_generate(self, prompts: Tuple[torch.Tensor, ...], batch_index: int) -> None:
        return

    def generate_step(self, inputs: Tuple[torch.Tensor, ...], *, t: int = 0, **parameters: Dict[str, torch.Tensor]) -> \
    Tuple[torch.Tensor, ...]:
        return tuple(self.forward(i) for i in inputs)

    def after_generate(self, final_outputs: Tuple[torch.Tensor, ...], batch_index: int) -> None:
        return

    @classmethod
    def from_config(cls, config: ARMConfig):
        return cls(config)

    def __init__(self, config: ARMConfig):
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
