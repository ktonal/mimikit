from pytorch_lightning import LightningModule
from typing import Iterable
import dataclasses as dtc
from torch.utils.data import Dataset


from ...data import Database, DataModule

__all__ = []


class DataPart(LightningModule):
    """
    creates and attaches a DataModule to a Model.

    Note that this part has no argument in its constructor but a **kwargs placeholder,
    thus allowing you to add explicit data-hyperparameters to your model and to control how they
    affect db-creation (e.g. schema) and/or serving the db (e.g. dataset_cls, batch_signature)

    Examples
    --------

    import mimikit as K

    class MyDataPart(K.models.DataPart):
        def __init__(self, sr=16000, q_levels=256):

            # override the class attribute in this instance :
            self.schema = {"qx": K.audios.MuLawSignal(sr=sr, q_levels=q_levels)

            # call DataPart's init for building the DataModule :
            K.models.DataPart.__init__(self)


    """

    db: [Database, Dataset, str] = None
    keep_open: bool = False
    sources: [str, Iterable[str]] = ''
    schema: dict = dtc.field(default_factory=dict)
    dataset_cls: type = None
    in_mem_data: bool = True
    splits: tuple = tuple()
    loader_kwargs: dict = dtc.field(default_factory=dict)

    def __init__(self):
        super(LightningModule, self).__init__()
        self.datamodule = DataModule(self,
                                     self.db,
                                     self.keep_open,
                                     self.sources,
                                     self.schema,
                                     self.dataset_cls,
                                     self.in_mem_data,
                                     self.splits,
                                     self.loader_kwargs
                                     )
