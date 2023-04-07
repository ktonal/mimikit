import pytest
import dataclasses as dtc

import torch
import torch.nn as nn
from assertpy import assert_that

import mimikit as mmk
import mimikit.config
import mimikit.networks.arm


class MyCustom(mimikit.config.Configurable, nn.Module):
    @dtc.dataclass
    class CustomConfig(mimikit.networks.arm.NetworkConfig):
        io_spec: mmk.IOSpec = mmk.IOSpec(
            inputs=(mmk.InputSpec(
                extractor_name="signal",
                transform=mmk.Normalize(),
                module=mmk.IOFactory(module_type="linear",
                                     params=mmk.LinearParams())
            ).bind_to(mmk.Extractor("signal", mmk.FileToSignal(16000))),),
            targets=(mmk.TargetSpec(
                extractor_name="signal",
                transform=mmk.Normalize(),
                module=mmk.IOFactory(module_type="linear",
                                     params=mmk.LinearParams()),
                objective=mmk.Objective(objective_type="reconstruction")
            ).bind_to(mmk.Extractor("signal", mmk.FileToSignal(16000))),)
        )
        x: int = 1

    @classmethod
    def from_config(cls, config: "MyCustom.CustomConfig"):
        return cls(config, nn.Linear(config.x, config.x))

    def __init__(self, config: "MyCustom.CustomConfig", module: nn.Module):
        super().__init__()
        self._config = config
        self.mod = module

    def forward(self, x):
        return self.mod(x)

    @property
    def config(self): return self._config


def test_should_save_and_load_class_defined_outside_mmk(tmp_path_factory):
    model = MyCustom.from_config(MyCustom.CustomConfig())

    output = model(torch.randn(2, 1, 1))

    assert_that(type(output)).is_equal_to(torch.Tensor)

    root = str(tmp_path_factory.mktemp("ckpt"))
    ckpt = mmk.Checkpoint(id="123", epoch=1, root_dir=root)

    ckpt.create(network=model)
    loaded = ckpt.network

    assert_that(type(loaded)).is_equal_to(MyCustom)
