import dataclasses as dtc
import os
import pytorch_lightning as pl
import torch
import numpy as np
from numpy.lib.stride_tricks import as_strided as np_as_strided
from torch.utils.data import Dataset, DataLoader
from typing import Iterable, Optional, Callable
import re
from random import randint
from torch._six import container_abcs, string_classes

from . import Database

__all__ = [
    'AsSlice',
    'AsFramedSlice',
    'Input',
    'Target',
    'process_batch',
    'DefaultDataset',
    'DataModule'
]


@dtc.dataclass
class Getter:
    """
    base class for implementing data getter

    Parameters
    ----------

    Attributes
    ----------
    n : int or None
        the length of the underlying data
    """
    n: Optional[int] = dtc.field(default=None, init=False)

    def __call__(self, feat_data, item):
        """
        apply this instance's logic to get data from ``feat_data`` for a given ``item``

        Parameters
        ----------
        feat_data: [np.ndarray, torch.Tensor, mimikit.FeatureProxy]

        item: int
            the index emitted from a sampler

        Returns
        -------
        data: Any
            the examples corresponding to this item
        """
        return feat_data[item]

    def __len__(self):
        return self.n


@dtc.dataclass()
class AsSlice(Getter):
    """
    maps an ``item`` to a slice of data

    Parameters
    ----------
    shift : int
        the slice will start at the index `item + shift`
    length : int
        the length of the slice
    stride : int
        sub-sampling factor. Every `stride` datapoints `item` increases of `1`

    Examples
    --------

    .. testcode::

       import mimikit as mmk

       slicer = mmk.AsSlice(shift=2, length=3)
       data, item = list(range(10)), 2

       # now use it like a function :
       sliced = slicer(data, item)

       print(sliced)

    will output:

    .. testoutput::

       [4, 5, 6]
    """
    shift: int = 0
    length: int = 1
    stride: int = 1

    def __call__(self, feat_data, item):
        i = item * self.stride
        return feat_data[slice(i + self.shift, i + self.shift + self.length)]

    def __len__(self):
        return (self.n - (self.shift + self.length) + 1) // self.stride


@dtc.dataclass
class AsFramedSlice(AsSlice):
    frame_size: int = 1
    as_strided: bool = False

    def __call__(self, feat_data, item):
        sliced = super(AsFramedSlice, self).__call__(feat_data, item)
        if self.as_strided:
            if isinstance(sliced, np.ndarray):
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


def process_batch(batch, test=lambda x: False, func=lambda x: x):
    """
    recursively apply func to the elements of data if test(element) is True.
    This is used in DefaultDataset to process elements (Input or Target) packed in tuples, list, dict etc...
    """
    elem_type = type(batch)
    if test(batch):
        return func(batch)
    elif isinstance(batch, container_abcs.Mapping):
        return {key: process_batch(batch[key], test, func) for key in batch}
    elif isinstance(batch, tuple) and hasattr(batch, '_fields'):  # namedtuple
        return elem_type(*(process_batch(d, test, func) for d in batch))
    elif isinstance(batch, container_abcs.Sequence) and not isinstance(batch, string_classes):
        return [process_batch(d, test, func) for d in batch]
    else:
        return batch


def _is_batchitem(obj):
    return isinstance(obj, (Input, Target))


class DefaultDataset(Dataset):

    def prepare_dataset(self, batch=tuple()):
        super(Dataset, self).__init__()

        # pass the lengths of the db features to the getters
        def cache_lengths(feat):
            if feat.getter.n is None:
                setattr(feat.getter, 'n', len(getattr(self, feat.db_key)))
            return feat

        self.batch = process_batch(batch, _is_batchitem, cache_lengths)

        # get the minimum length of all batchitems
        self.N = float('inf')

        def set_n_to_min(feat):
            self.N = min(len(feat), self.N)
            return feat

        process_batch(self.batch, _is_batchitem, set_n_to_min)

        return self

    def __getitem__(self, item):
        def get_data(feat):
            return feat.transform(feat.getter(getattr(self, feat.db_key), item))

        return process_batch(self.batch, _is_batchitem, get_data)

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
        self.has_split = False
        self.full_ds = None
        self.datasets = {}

    def prepare_data(self, *args, **kwargs):
        """
        creates a db if necessary and possible
        """
        # get a db object (create it if necessary)
        if isinstance(self.db, (str, os.PathLike)):
            if not os.path.exists(self.db):
                # user must provide sources and schema to build db
                if not self.sources and self.schema:
                    raise ValueError("You need to provide sources and a schema in order to create a db")
                Database.create(self.db, self.sources, self.schema)

    def _instantiate_db(self):
        if isinstance(self.db, (str, os.PathLike)):
            self.db = Database(self.db, keep_open=self.keep_open)
        elif isinstance(self.db, (Database, Dataset)):
            pass
        else:
            raise TypeError("Expected `db` to be of type str, Database or Dataset. Got " + str(type(self.db)))

    def _init_ds(self, ds, stage):
        # upgrade db to a Dataset if it isn't one already
        if _implements_mapstyle(ds):
            pass
        # user provided Dataset class
        elif self.dataset_cls is not None:
            ds = ds.bind(self.dataset_cls)
        # model has a batch signature
        elif getattr(self.model, 'batch_signature', False):
            ds = ds.bind(DefaultDataset)
        else:
            raise RuntimeError("couldn't instantiate a Dataset with the provided arguments")
        if hasattr(ds, 'prepare_dataset') and hasattr(self.model, 'batch_signature'):
            ds = ds.prepare_dataset(self.model.batch_signature(stage))
        return ds

    def _split(self):
        if not self.splits:
            self.datasets['full'] = self.full_ds
            self.datasets['fit'] = self.full_ds
        else:
            sets = self.full_ds.split(self.splits)
            # store the sets as attr
            for ds, stage in zip(sets, ["fit", "val", "test"]):
                if ds is not None:
                    self.datasets[stage] = ds
        setattr(self, 'has_split', True)
        return self

    def _move_to_mem(self, ds):
        if isinstance(ds, Database):
            if self.in_mem_data and torch.cuda.is_available():
                ds.to_tensor()
                ds.to("cuda")
        return ds

    def setup(self, stage=None):
        if not self.has_prepared_data:
            self.prepare_data()
        if not isinstance(self.db, (Database, Dataset)):
            self._instantiate_db()
        if self.full_ds is None:
            self.full_ds = self._init_ds(self.db, 'fit')
            self.full_ds = self._move_to_mem(self.full_ds)
        if not self.has_split:
            self._split()
        # update loader kwargs
        if hasattr(self.model, 'loader_kwargs'):
            self.loader_kwargs = self.model.loader_kwargs(stage, self)

    def full_dataloader(self):
        self.setup()
        return DataLoader(self.full_ds, **self.loader_kwargs)

    def train_dataloader(self):
        if not self.has_setup_fit:
            self.setup("fit")
        return DataLoader(self.datasets['fit'], **self.loader_kwargs)

    def val_dataloader(self):
        if 'val' not in self.datasets:
            return None
        if not self.has_setup_fit:
            self.setup("val")
        kwargs = self.loader_kwargs.copy()
        return DataLoader(self.datasets['val'], **kwargs)

    def test_dataloader(self):
        if 'test' not in self.datasets:
            return None
        if not self.has_setup_test:
            self.setup("test")
        kwargs = self.loader_kwargs.copy()
        return DataLoader(self.datasets['test'], **kwargs)

    def get_prompts(self, indices=tuple()):
        ds = self.full_ds
        if hasattr(ds, 'prepare_dataset') and hasattr(self.model, 'batch_signature'):
            ds = ds.prepare_dataset(self.model.batch_signature('test'))
        N = len(ds)
        data = [ds[randint(0, N) if ix is None else ix] for ix in indices]
        return torch.utils.data.dataloader.default_collate(data)
