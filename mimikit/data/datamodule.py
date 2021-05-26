import dataclasses as dtc
import os
import pytorch_lightning as pl
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Iterable

from . import Database, Input, Target

__all__ = [
    'DefaultDataset',
    'DataModule'
]


class DefaultDataset(Dataset):

    def prepare_dataset(self, model=None):
        super(DefaultDataset, self).__init__()
        self.batch = model.batch_signature()
        # pass the lengths to the getters
        for feat in self.batch:
            if feat.getter.n_examples is None:
                feat.getter.n_examples = len(getattr(self, feat.db_key))
        self.N = min(len(feature) for feature in self.batch)

    def __getitem__(self, item):
        inputs = tuple(feat.transform(feat.getter(getattr(self, feat.db_key), item))
                       for feat in self.batch if isinstance(feat, Input))
        targets = tuple(feat.transform(feat.getter(getattr(self, feat.db_key), item))
                        for feat in self.batch if isinstance(feat, Target))
        return inputs, targets

    def __len__(self):
        return self.N


def implements_mapstyle(obj):
    if isinstance(obj, type):
        getitem, flen = getattr(obj, '__getitem__', False), getattr(obj, '__len__', False)
    else:
        getitem, flen = getattr(obj.__class__, '__getitem__', False), getattr(obj.__class__, '__len__', False)

    return getitem not in (Database.__getitem__, Dataset.__getitem__) and flen is not Database.__len__


@dtc.dataclass
class DataModule(pl.LightningDataModule):

    model: pl.LightningModule = None
    db: [Database, Dataset, str] = None
    keep_open: bool = False
    sources: [str, Iterable[str]] = ''
    schema: dict = dtc.field(default_factory=dict)
    dataset_cls: type = None
    in_mem_data: bool = True
    splits: tuple = None
    loader_kwargs: dict = dtc.field(default_factory=dict)

    def __post_init__(self):
        self.full_ds, self.train_ds, self.val_ds, self.test_ds = None, None, None, None

    def prepare_data(self, *args, **kwargs):
        # get a db object
        if isinstance(self.db, (str, os.PathLike)) and not os.path.exists(self.db):
            # user must provide sources and schema to build db
            if not self.sources and self.schema:
                raise ValueError("You need to provide sources and a schema in order to create a db")
            db = Database.build(self.db, self.sources, self.schema)
            if self.keep_open:
                db = Database(self.db, keep_open=True)
        elif isinstance(self.db, (Database, Dataset)):
            # user provided built db
            db = self.db
        else:
            raise TypeError("Expected `db` to be of type str, Database or Dataset. Got " + str(type(self.db)))

        # upgrade db to a Dataset if it isn't one already

        # db implements Dataset interface
        if implements_mapstyle(db):
            pass
        # user provided Dataset class
        elif self.dataset_cls is not None:
            db.bind(self.dataset_cls)
        # model has a batch signature
        elif hasattr(self.model, 'batch_signature'):
            db.bind(DefaultDataset)
        else:
            raise RuntimeError("couldn't instantiate a Dataset with the provided arguments")

        # now we have it!
        self.db = db
        self.db.prepare_dataset(self.model)

        # move data to device, split
        if self.in_mem_data and torch.cuda.is_available():
            self.db.to_tensor()
            self.db.to("cuda")
        if not self.splits:
            sets = (self.db,)
        else:
            sets = self.db.split(self.splits)
        # store the sets as attr
        for ds, attr in zip(sets, ["train_ds", "val_ds", "test_ds"]):
            setattr(self, attr, ds)
        setattr(self, 'full_ds', self.db)

        return self

    def setup(self, stage=None):
        if stage == "fit":
            pass

    def length(self, split: str):
        ds = dict(full=self.db, train=self.train_ds, val=self.val_ds, test=self.test_ds)[split]
        return len(ds) if ds is not None else None

    def full_dataloader(self):
        if not self.has_prepared_data:
            self.prepare_data()
        return DataLoader(self.db, **self.loader_kwargs)

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
