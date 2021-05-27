import dataclasses as dtc
import os
import pytorch_lightning as pl
import torch
from numpy.lib.stride_tricks import as_strided as np_as_strided
from torch.utils.data import Dataset, DataLoader
from typing import Iterable, Optional, Callable
import re
from torch._six import container_abcs, string_classes

from . import Database

__all__ = [
    'AsSlice',
    'AsFramedSlice',
    'Input',
    'Target',
    'DefaultDataset',
    'DataModule'
]


@dtc.dataclass
class Getter:
    n: Optional[int] = dtc.field(default=None, init=False)

    def __call__(self, feat_data, item):
        return feat_data[item]

    def __len__(self):
        return self.n


@dtc.dataclass
class AsSlice(Getter):
    shift: int = 0
    length: int = 1
    stride: int = 1

    def __call__(self, feat_data, item):
        i = item * self.stride
        return feat_data[slice(i + self.shift, i + self.shift + self.length)]

    def __len__(self):
        return (self.n - self.shift + self.length + 1) // self.stride


@dtc.dataclass
class AsFramedSlice(AsSlice):

    frame_size: int = 1
    as_strided: bool = False

    def __call__(self, feat_data, item):
        sliced = super(AsFramedSlice, self).__call__(feat_data, item)
        if self.as_strided:
            if type(feat_data) is not torch.Tensor:
                itemsize = sliced.dtype.itemsize
                as_strided = lambda arr: np_as_strided(arr,
                                                       shape=(self.length, self.frame_size),
                                                       strides=(itemsize, itemsize))
            else:
                as_strided = lambda tensor: torch.as_strided(tensor,
                                                             size=(self.length, self.frame_size),
                                                             stride=(1, 1))

            with torch.no_grad():
                return as_strided(sliced)
        else:
            return sliced.reshape(-1, self.frame_size)


@dtc.dataclass
class Input:
    db_key: str = ''
    getter: Getter = Getter()
    transform: Callable = lambda x: x

    def __len__(self):
        return len(self.getter)


class Target(Input):
    """exactly equivalent to Input, just makes code simpler to read."""
    pass


np_str_obj_array_pattern = re.compile(r'[SaUO]')


def _apply_recursively(data, test=lambda x: False, func=lambda x: x):
    """
    recursively apply func to the elements of data if test(element) is True.
    This is used in DefaultDataset to process elements (Input or Target) packed in tuples, list, dict etc...
    """
    elem_type = type(data)
    if test(data):
        return func(data)
    elif isinstance(data, container_abcs.Mapping):
        return {key: _apply_recursively(data[key], test, func) for key in data}
    elif isinstance(data, tuple) and hasattr(data, '_fields'):  # namedtuple
        return elem_type(*(_apply_recursively(d, test, func) for d in data))
    elif isinstance(data, container_abcs.Sequence) and not isinstance(data, string_classes):
        return [_apply_recursively(d, test, func) for d in data]
    else:
        return data


def _is_batchitem(obj):
    return isinstance(obj, (Input, Target))


class DefaultDataset(Dataset):

    def prepare_dataset(self, model=None):
        super(Dataset, self).__init__()
        batch = model.batch_signature()

        # pass the lengths of the db features to the getters
        def cache_lengths(feat):
            if feat.getter.n is None:
                setattr(feat.getter, 'n', len(getattr(self, feat.db_key)))
            return feat
        self.batch = _apply_recursively(batch, _is_batchitem, cache_lengths)

        # get the minimum length of all batchitems
        self.N = float('inf')

        def set_n_to_min(feat):
            self.N = min(len(feat), self.N)
            return feat
        _apply_recursively(self.batch, _is_batchitem, set_n_to_min)

    def __getitem__(self, item):
        def get_data(feat):
            return feat.transform(feat.getter(getattr(self, feat.db_key), item))
        return _apply_recursively(self.batch, _is_batchitem, get_data)

    def __len__(self):
        return self.N


def _implements_mapstyle(obj):
    if isinstance(obj, type):
        getitem, flen = getattr(obj, '__getitem__', False), getattr(obj, '__len__', False)
    else:
        getitem, flen = getattr(obj.__class__, '__getitem__', False), getattr(obj.__class__, '__len__', False)

    return getitem not in (Database.__getitem__, Dataset.__getitem__) and flen is not Database.__len__


@dtc.dataclass
class DataModule(pl.LightningDataModule):
    """
    encapsulate the logic for creating, initializing and serving databases & datasets
    """

    model: pl.LightningModule = None
    db: [Database, Dataset, str] = None
    keep_open: bool = True
    sources: [str, Iterable[str]] = ''
    schema: dict = dtc.field(default_factory=dict)
    dataset_cls: type = None
    in_mem_data: bool = True
    splits: tuple = tuple()
    loader_kwargs: dict = dtc.field(default_factory=dict)

    def __post_init__(self):
        super(DataModule, self).__init__()
        self.full_ds, self.train_ds, self.val_ds, self.test_ds = None, None, None, None

    def prepare_data(self, *args, **kwargs):
        # get a db object (create it if necessary)
        if isinstance(self.db, (str, os.PathLike)):
            if not os.path.exists(self.db):
                # user must provide sources and schema to build db
                if not self.sources and self.schema:
                    raise ValueError("You need to provide sources and a schema in order to create a db")
                db = Database.build(self.db, self.sources, self.schema)
            else:
                db = Database(self.db, keep_open=self.keep_open)
        elif isinstance(self.db, (Database, Dataset)):
            # user provided built db
            db = self.db
        else:
            raise TypeError("Expected `db` to be of type str, Database or Dataset. Got " + str(type(self.db)))

        # upgrade db to a Dataset if it isn't one already
        if _implements_mapstyle(db):
            pass
        # user provided Dataset class
        elif self.dataset_cls is not None:
            db.bind(self.dataset_cls)
        # model has a batch signature
        elif getattr(self.model, 'batch_signature', False):
            db.bind(DefaultDataset)
        else:
            raise RuntimeError("couldn't instantiate a Dataset with the provided arguments")

        # now we have it!
        self.db = db
        # maybe the db is a standard Dataset...
        if hasattr(self.db, 'prepare_dataset'):
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

        # update loader kwargs
        if hasattr(self.model, 'loader_kwargs'):
            self.loader_kwargs.update(self.model.loader_kwargs(self))

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


