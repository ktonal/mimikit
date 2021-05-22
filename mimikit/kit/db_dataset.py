from __future__ import annotations

from inspect import getfullargspec

import torch
from pytorch_lightning import LightningDataModule
from torch.utils.data import Dataset, random_split, DataLoader
import pytorch_lightning as pl

from ..h5data import Database


class DBDataset(Database, Dataset):
    """
    extends ``Database`` so that it can also be used in place of a ``Dataset``
    """

    def __init__(self, h5_file: str, keep_open: bool = False):
        """
        "first" init instantiates a ``Database``

        Parameters
        ----------
        h5_file : str
            path to the .h5 file containing the data
        keep_open : bool, optional
            whether to keep the h5 file open or close it after each query.
            Default is ``False``.
        """
        Database.__init__(self, h5_file, keep_open)


class DBDataModule(LightningDataModule):
    """
    boilerplate subclass of ``pytorch_lightning.LightningDataModule`` to handle standard "data-tasks" :
        - give a Database a chance to prepare itself for serving data once the model has been instantiated
        - move small datasets to the RAM of the gpu if desired
            (TODO: Otherwise, more workers are used in the DataLoaders for better performance)
        - split data into train, val and test sets and serve corresponding DataLoaders
    """
    def __init__(self,
                 model=None,
                 db: DBDataset = None,
                 in_mem_data=True,
                 splits=None,
                 **loader_kwargs,
                 ):
        super(DBDataModule, self).__init__()
        self.model = model
        self.db = db
        self.in_mem_data = in_mem_data
        self.splits = splits
        self.loader_kwargs = self._filter_loader_kwargs(loader_kwargs)
        self.train_ds, self.val_ds, self.test_ds = None, None, None

    def prepare_data(self, *args, **kwargs):
        self.db.prepare_dataset(model=self.model, datamodule=self)

    def setup(self, stage=None):
        if stage == "fit":
            if self.in_mem_data and torch.cuda.is_available():
                self.db.to_tensor()
                self.db.to("cuda")
            if not self.splits:
                sets = (self.db, )
            else:
                sets = self.db.split(self.splits)
            for ds, attr in zip(sets, ["train_ds", "val_ds", "test_ds"]):
                setattr(self, attr, ds)

    def train_dataloader(self):
        if not self.has_prepared_data:
            self.prepare_data()
        if not self.has_setup_fit:
            self.setup("fit")
        return DataLoader(self.train_ds, **self.loader_kwargs)

    def val_dataloader(self, shuffle=False):
        has_val = self.splits is not None and len(self.splits) >= 2 and self.splits[1] is not None
        if not has_val:
            return None
        if not self.has_prepared_data:
            self.prepare_data()
        if not self.has_setup_fit:
            self.setup("fit")
        kwargs = self.loader_kwargs.copy()
        kwargs["shuffle"] = shuffle
        return DataLoader(self.val_ds, **kwargs)

    def test_dataloader(self, shuffle=False):
        has_test = self.splits is not None and len(self.splits) >= 3 and self.splits[2] is not None
        if not has_test:
            return None
        if not self.has_prepared_data:
            self.prepare_data()
        if not self.has_setup_test:
            self.setup("test")
        kwargs = self.loader_kwargs.copy()
        kwargs["shuffle"] = shuffle
        return DataLoader(self.test_ds, **kwargs)

    @staticmethod
    def _filter_loader_kwargs(kwargs):
        valids = getfullargspec(DataLoader.__init__).annotations.keys()
        return {k: v for k, v in kwargs.items() if k in valids}