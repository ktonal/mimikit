import dataclasses as dtc
import h5mapper as h5m
import json
import os
from .s2s import Seq2SeqLSTM, Seq2SeqLSTMv0
from .wavenets import WaveNetFFT, WaveNetQx
from .srnns import SampleRNN

__all__ = [
    'find_checkpoints',
    'load_trainings_hp',
    'load_network_cls',
    'load_feature',
    'Checkpoint'
]

def find_checkpoints(root="trainings"):
    return h5m.FileWalker(r"\.h5", root)


def load_trainings_hp(dirname):
    return json.loads(open(os.path.join(dirname, "hp.json"), 'r').read())


class CkptBank(h5m.TypedFile):
    ckpt = h5m.TensorDict()


def load_feature(s):
    import mimikit as mmk
    loc = dict()
    exec(f"feature = {s}", mmk.__dict__, loc)
    return loc["feature"]


def load_network_cls(s):
    loc = dict()
    exec(f"cls = {s}", globals(), loc)
    return loc["cls"]


@dtc.dataclass
class Checkpoint:
    id: str
    epoch: int
    root_dir: str = "./"

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
        bank = CkptBank(self.os_path, 'r')
        hp = bank.ckpt.load_hp()
        return bank.ckpt.load_checkpoint(hp["cls"], "state_dict")

    @property
    def feature(self):
        bank = CkptBank(self.os_path, 'r')
        hp = bank.ckpt.load_hp()
        return hp['feature']

    @property
    def train_hp(self):
        return load_trainings_hp(os.path.join(self.root_dir, self.id))
