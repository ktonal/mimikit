import abc
import dataclasses as dtc
from typing import Optional
from functools import cached_property
import torch.nn as nn

import h5mapper as h5m
import os

from .config import Config, Configurable, TrainingConfig, NetworkConfig
from .dataset import DatasetConfig


__all__ = [
    'Checkpoint',
    'CheckpointBank'
]


class ConfigurableModule(Configurable, nn.Module, abc.ABC):
    pass


class CheckpointBank(h5m.TypedFile):
    network = h5m.TensorDict()
    optimizer = h5m.TensorDict()

    @classmethod
    def save(cls,
             filename: str,
             network: ConfigurableModule,
             training_config: TrainingConfig,
             optimizer: Optional[nn.Module] = None
             ) -> "CheckpointBank":

        net_dict = network.state_dict()
        opt_dict = optimizer.state_dict() if optimizer is not None else {}
        cls.network.set_ds_kwargs(net_dict)
        if optimizer is not None:
            cls.optimizer.set_ds_kwargs(opt_dict)
        os.makedirs(os.path.split(filename)[0], exist_ok=True)

        bank = cls(filename, mode="w")
        bank.network.attrs["config"] = network.config.serialize()
        bank.network.add("state_dict", h5m.TensorDict.format(net_dict))

        if optimizer is not None:
            bank.optimizer.add("state_dict", h5m.TensorDict.format(opt_dict))

        bank.attrs["data"] = training_config.data.serialize()
        bank.attrs["training"] = training_config.training.serialize()
        bank.flush()
        bank.close()
        return bank


@dtc.dataclass
class Checkpoint:
    id: str
    epoch: int
    root_dir: str = "models/"

    def create(self,
               network: ConfigurableModule,
               training_config: TrainingConfig,
               optimizer: Optional[nn.Module] = None):
        CheckpointBank.save(self.os_path, network, training_config, optimizer)
        return self

    @staticmethod
    def get_id_and_epoch(path):
        id_, epoch = path.split("/")[-2:]
        return id_.strip("/"), int(epoch.split(".h5")[0].split("=")[-1])

    @staticmethod
    def from_path(path):
        basename = os.path.dirname(os.path.dirname(path))
        return Checkpoint(*Checkpoint.get_id_and_epoch(path), root_dir=basename)

    @property
    def os_path(self):
        return os.path.join(self.root_dir, f"{self.id}/epoch={self.epoch}.h5")

    def delete(self):
        os.remove(self.os_path)

    @cached_property
    def bank(self) -> CheckpointBank:
        return CheckpointBank(self.os_path, 'r')

    @cached_property
    def network_config(self) -> NetworkConfig:
        return Config.deserialize(self.bank.network.attrs["config"])

    @cached_property
    def training_config(self) -> TrainingConfig:
        bank = CheckpointBank(self.os_path, 'r')
        return Config.deserialize(bank.attrs["training"])

    @cached_property
    def data_config(self) -> DatasetConfig:
        return Config.deserialize(self.bank.attrs["data"])

    @cached_property
    def network(self) -> ConfigurableModule:
        cfg: NetworkConfig = self.network_config
        cls = cfg.owner_class
        state_dict = self.bank.network.get("state_dict")
        net = cls.from_config(cfg)
        net.load_state_dict(state_dict, strict=True)
        return net

    @cached_property
    def dataset(self):
        dataset: DatasetConfig = self.data_config
        if os.path.exists(dataset.filename):
            return dataset.get(mode="r")
        return dataset.create(mode="w")

    # Todo: method to add state_dict mul by weights -> def average(self, *others)
