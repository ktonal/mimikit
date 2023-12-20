import abc
import dataclasses as dtc
from typing import Optional
from typing_extensions import Protocol
from omegaconf import OmegaConf, ListConfig, DictConfig

try:
    from functools import cached_property
except ImportError:  # python<3.8
    def cached_property(f):
        return property(f)

import torch.nn as nn
from torch.optim import Adam
import torch

import h5mapper as h5m
import os

from .config import Config, Configurable
from .networks.arm import NetworkConfig
from .features.dataset import DatasetConfig

__all__ = [
    'Checkpoint',
    'CheckpointBank',
    "CheckpointsMix"
]


class ConfigurableModule(Configurable, nn.Module, abc.ABC):
    @property
    @abc.abstractmethod
    def config(self) -> NetworkConfig:
        ...


class TrainingConfig(Protocol):
    @property
    def dataset(self) -> DatasetConfig:
        ...

    @property
    def network(self) -> NetworkConfig:
        ...

    @property
    def training(self) -> Config:
        ...


class CheckpointBank(h5m.TypedFile):
    network = h5m.TensorDict()
    optimizer = h5m.TensorDict()

    @classmethod
    def save(cls,
             filename: str,
             network: ConfigurableModule,
             training_config: Optional[TrainingConfig] = None,
             optimizer: Optional[nn.Module] = None,
             trainer_state: Optional[dict] = None
             ) -> "CheckpointBank":

        net_dict = network.state_dict()
        opt_dict = optimizer.state_dict() if optimizer is not None else {}
        cls.network.set_ds_kwargs(net_dict)
        # if optimizer is not None:
        #    cls.optimizer.set_ds_kwargs(opt_dict)
        os.makedirs(os.path.split(filename)[0], exist_ok=True)

        bank = cls(filename, mode="w")
        bank.network.attrs["config"] = network.config.serialize()
        bank.network.add("state_dict", h5m.TensorDict.format(net_dict))

        if optimizer is not None:
            # bank.optimizer.add("state_dict", h5m.TensorDict.format(opt_dict))
            torch.save(opt_dict, os.path.splitext(filename)[0] + ".opt")

        if training_config is not None:
            bank.attrs["dataset"] = training_config.dataset.serialize()
            bank.attrs["training"] = training_config.training.serialize()
        else:
            # make a dataset config for being able to at least load the network later
            features = [*network.config.io_spec.inputs, *network.config.io_spec.targets]
            schema = {f.extractor_name: f.extractor for f in features}
            bank.attrs["dataset"] = DatasetConfig(filename="unknown", sources=(),
                                                  extractors=tuple(schema.values())).serialize()
        if trainer_state is not None:
            bank.attrs["trainer_state"] = OmegaConf.to_yaml(OmegaConf.structured(trainer_state))

        bank.flush()
        bank.close()
        return bank


@dtc.dataclass
class Checkpoint:
    id: str
    epoch: int
    root_dir: str = "./"
    weight: float = 1.

    def create(self,
               network: ConfigurableModule,
               training_config: Optional[TrainingConfig] = None,
               optimizer: Optional[nn.Module] = None,
               trainer_state: Optional[dict] = None):
        CheckpointBank.save(self.os_path, network, training_config, optimizer, trainer_state)
        return self

    @staticmethod
    def get_id_and_epoch(path):
        id_, epoch = path.split("/")[-2:]
        return id_.strip("/"), int(epoch.split(".ckpt")[0].split("=")[-1])

    @staticmethod
    def from_path(path):
        basename = os.path.dirname(os.path.dirname(path))
        return Checkpoint(*Checkpoint.get_id_and_epoch(path), root_dir=basename)

    @property
    def os_path(self):
        return os.path.join(self.root_dir, f"{self.id}/epoch={self.epoch}.ckpt")

    def delete(self):
        os.remove(self.os_path)

    @cached_property
    def bank(self) -> CheckpointBank:
        return CheckpointBank(self.os_path, 'r')

    @cached_property
    def dataset_config(self) -> DatasetConfig:
        return Config.deserialize(self.bank.attrs["dataset"], as_type=DatasetConfig)

    @cached_property
    def network_config(self) -> NetworkConfig:
        return Config.deserialize(self.bank.network.attrs["config"])

    @cached_property
    def training_config(self) -> TrainingConfig:
        bank = CheckpointBank(self.os_path, 'r')
        return Config.deserialize(bank.attrs["training"])

    @cached_property
    def network_state_dict(self):
        state_dict = self.bank.network.get("state_dict")
        if self.weight != 1.:
            state_dict = {k: v * self.weight for k, v in state_dict.items()}
        return state_dict

    @cached_property
    def network(self) -> ConfigurableModule:
        cfg: NetworkConfig = self.network_config
        cfg.io_spec.bind_to(self.dataset_config)
        cls = cfg.owner_class
        state_dict = self.bank.network.get("state_dict")
        net = cls.from_config(cfg)
        net.load_state_dict(state_dict, strict=True)
        return net

    @cached_property
    def dataset(self) -> h5m.TypedFile:
        dataset: DatasetConfig = self.dataset_config
        if os.path.exists(dataset.filename):
            return dataset.get(mode="r")
        return dataset.create(mode="w")

    @cached_property
    def optimizer_state(self):
        opt_path = os.path.join(self.root_dir, f"{self.id}/epoch={self.epoch}.opt")
        if os.path.isfile(opt_path):
            return torch.load(opt_path)
        return None

    @cached_property
    def trainer_state(self):
        state = self.bank.attrs.get('trainer_state', None)
        if state is not None:
            return OmegaConf.create(state)
        return None


def CheckpointsMix(*checkpoints: Checkpoint):
    net_cfg = checkpoints[0].network_config
    net_cfg.io_spec.bind_to(checkpoints[0].dataset_config)
    cls = net_cfg.owner_class
    net_state_dict = checkpoints[0].network_state_dict
    for ckpt in checkpoints[1:]:
        other_dict = ckpt.network_state_dict
        for k, v in other_dict.items():
            net_state_dict.update({k: net_state_dict[k] + v})
    net = cls.from_config(net_cfg)
    net.load_state_dict(net_state_dict, strict=True)
    return net
