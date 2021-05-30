from pytorch_lightning import LightningModule
from typing import Iterable
import dataclasses as dtc
from torch.utils.data import Dataset


from ...data import Database, DataModule

__all__ = [
    'IData'
]


class IData(LightningModule):

    def __init__(self):
        LightningModule.__init__(self)

    @property
    def dm(self) -> DataModule:
        return self.trainer.data_connector.datamodule

    @classmethod
    def schema(cls, *args, **kwargs) -> dict:
        raise NotImplementedError

    @classmethod
    def dependant_hp(cls, db: Database) -> dict:
        return {}

    def batch_signature(self, stage):
        raise NotImplementedError

    def loader_kwargs(self, stage, datamodule) -> dict:
        raise NotImplementedError
