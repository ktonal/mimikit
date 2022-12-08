import dataclasses as dtc
import h5mapper as h5m
import os

from .config import Config


__all__ = [
    'Checkpoint',
    'CheckpointBank'
]


class CheckpointBank(h5m.TypedFile):
    network = h5m.TensorDict()
    optimizer = h5m.TensorDict()

    @classmethod
    def save(cls,
             filename: str,
             model_config: Config,
             network,
             optimizer
             ) -> "CheckpointBank":
        net_dict = network.state_dict()
        opt_dict = optimizer.state_dict() if optimizer is not None else {}
        cls.network.set_ds_kwargs(net_dict)
        if optimizer is not None:
            cls.optimizer.set_ds_kwargs(opt_dict)
        os.makedirs(os.path.split(filename)[0], exist_ok=True)
        bank = cls(filename, mode="w")
        bank.attrs["config"] = model_config.serialize()
        bank.network.add("state_dict", h5m.TensorDict.format(net_dict))
        if optimizer is not None:
            bank.optimizer.add("state_dict", h5m.TensorDict.format(opt_dict))
        bank.flush()
        bank.close()
        return bank


@dtc.dataclass
class Checkpoint:
    id: str
    epoch: int
    root_dir: str = "models/"

    def create(self, model_config: Config, network, optimizer=None):
        CheckpointBank.save(self.os_path, model_config, network, optimizer)
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

    @property
    def network(self):
        bank = CheckpointBank(self.os_path, 'r')
        hp: Config = Config.deserialize(bank.attrs["config"])
        cls = hp.owner_class
        state_dict = bank.network.get("state_dict")
        net = cls.from_config(hp)
        net.load_state_dict(state_dict, strict=True)
        return net

    @property
    def feature(self):
        bank = CheckpointBank(self.os_path, 'r')
        hp = bank.network.load_hp()
        return hp['feature']

    @property
    def train_hp(self):
        return load_trainings_hp(os.path.join(self.root_dir, self.id))

    # Todo: method to add state_dict mul by weights -> def average(self, *others)
