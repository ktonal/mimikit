from functools import partial
import h5py
import torch
from torch.utils.data._utils.collate import default_convert
from torch.utils.data import random_split, Dataset
import numpy as np
import pandas as pd
import os
from argparse import Namespace
import pytorch_lightning as pl

from .create import make_root_db, write_feature
from .feature import Feature
from .regions import Regions
from ..file_walker import FileWalker, EXTENSIONS

__all__ = [
    "Database"
]


class FeatureProxy(object):
    """
    Interface to the numerical data (array) stored in a .h5 file created with ``mimikit.data.make_root_db``

    ``FeatureProxy`` objects are like ``numpy`` arrays (on disk) and can be used as map-style ``Dataset``
    because they implement ``__len__`` and ``__getitem__``.

    Parameters
    ----------
    h5_file : str
        name of the file
    ds_name : str
        name of the ``h5py.Dataset`` containing the data to be served

    Attributes
    ----------
    N : int
        the length of the first dimension of the underlying array
    shape : tuple of int
        the shape of the underlying array
    attrs : dict
        dictionary of additional information about the data as returned
        by the ``extract_func`` passed to ``make_root_db``.
        Typically, this is where you want to store the parameters of your extracting function, e.g.
        sample rate, hop length etc.
    files : Regions
        pandas DataFrame where each row contains name, start, stop & duration of one stored file.
        see ``Regions`` for more information
    regions : Regions or None
        pandas DataFrame containing index, start, stop & duration for segments ('sub-regions' of the files)
        This attribute isn't None only if the ``extract()`` method of the parent ``Database`` class returned such
        an object for this feature.
    """

    def __init__(self, h5_file: str, ds_name: str, keep_open=False):

        self.h5_file = h5_file
        self.name = ds_name
        with h5py.File(h5_file, "r") as f:
            ds = f[self.name]
            self.N = ds.shape[0]
            self.shape = ds.shape
            self.dtype = ds.dtype
            self.attrs = {k: v for k, v in ds.attrs.items()}
            has_files = self.name + "_files" in f
            has_regions = self.name + "_regions" in f
        self.files = Regions(pd.read_hdf(h5_file, self.name + "_files", mode="r")) if has_files else None
        self.regions = Regions(pd.read_hdf(h5_file, self.name + "_regions", mode="r")) if has_regions else None
        # handle to the file when keeping open. To support torch's Dataloader, we have to open the file by the
        # first getitem request
        self._f = None
        self.keep_open = keep_open

    def __len__(self):
        return self.N

    def __getitem__(self, item):
        if self.keep_open:
            if self._f is None:
                self._f = h5py.File(self.h5_file, "r+")
            return self._f[self.name][item]
        with h5py.File(self.h5_file, "r") as f:
            rv = f[self.name][item]
        return rv

    def __setitem__(self, item, value):
        if self.keep_open:
            if self._f is None:
                self._f = h5py.File(self.h5_file, "r+")
            self._f[self.name][item] = value
        with h5py.File(self.h5_file, "r+") as f:
            f[self.name][item] = value
        return

    def get_regions(self, regions):
        """
        get the data (numpy array) corresponding to the rows of `regions`

        Parameters
        ----------
        regions : Regions
            the files you want to get as array

        Returns
        -------
        data : np.ndarray
            the concatenated data for all the requested files

        Examples
        --------

            .. code-block::

                from mimikit import Database
                X = Database("my-db.h5").fft
                # only get files 0, 3 & 7
                X.get_regions(X.regions.iloc[[0, 3, 7]])
        """
        slices = regions.slices(0)
        return np.concatenate(tuple(self[slice_i] for slice_i in slices), axis=0)

    def close(self):
        if self._f is not None:
            self._f.close()
        return self

    def __repr__(self):
        return "<FeatureProxy: '%s/%s'>" % (self.h5_file, self.name)


class Database(Dataset):
    """
    interface to .h5 databases created by mimikit

    Parameters
    ----------
    h5_file : str
        path to the .h5 file containing the data
    keep_open : bool, optional
        whether to keep the h5 file open or close it after each query.
        Default is ``False``.

    Attributes
    ----------
    <feature_proxy> : FeatureProxy
        each feature created by the extracting function passed to ``make_root_db``
        is automatically added as attribute. If the extracting function returned a feature
        by the name ``"fft"``, the attribute ``fft`` of type ``FeatureProxy`` will be automatically
        added when the file is loaded and you will be able to access it through ``db.fft``.
    """

    schema = {}

    def __init__(self, h5_file: str, keep_open=False):
        self.h5_file = h5_file
        self.keep_open = keep_open
        with h5py.File(h5_file, "r") as f:
            self.attrs = {k: v for k, v in f.attrs.items()}
            self.features = f.attrs.get("features", ["fft"])
        # add found features as self.feature_name = FeatureProxy(self.h5_file, feature_name)
        self.params = Namespace()
        self._register_features(self.features, keep_open)

    def copy(self):
        return type(self)(self.h5_file, self.keep_open)

    @staticmethod
    def extract(path, **kwargs):
        raise NotImplementedError

    @classmethod
    def make(cls, db_name, files_ext='audio', items=tuple(), **kwargs):
        walker = FileWalker(files_ext, items)
        make_root_db(db_name, walker, partial(cls.extract, **kwargs))
        return cls(db_name)

    @classmethod
    def _load(cls, path, schema={}):
        """
        default extract_func for Database.build. Roughly equivalent to :
            ``{feat_name: feat.load(path) for feat_name, feat in features_dict.items()}``
        """
        out = {}
        for f_name, f in schema.items():
            # check that f has `load` and that `path` is of the right type
            if getattr(type(f), 'load', Feature.load) != Feature.load and \
                    os.path.splitext(path)[-1].strip('.') in EXTENSIONS[f.__ext__]:
                obj = f.load(path)
            # feature is derived from the output of an other
            elif hasattr(f, 'input_key') and f.input_key in out:
                # we us __call__ instead of load, the former being eq. to a transform
                obj = f(out[f.input_key][1])
            else:
                obj = None
            # obj is (data, regions)
            if isinstance(obj, tuple) and tuple(map(type, obj)) == (np.ndarray, Regions):
                out[f_name] = getattr(f, 'params', {}), *obj
            elif isinstance(obj, Regions):
                out[f_name] = getattr(f, 'params', {}), np.array([]), obj
            elif isinstance(obj, np.ndarray):
                out[f_name] = getattr(f, 'params', {}), obj, None
        return out

    @classmethod
    def create(cls, db_name, sources=tuple(), schema={}):
        """
        creates a db from the schema provided in `features_dict` and the files or root directories found in `items`

        Parameters
        ----------
        db_name : str
            the name of the file to be created
        sources : str or iterable of str
            the sources to be passed to `FileWalker`
        schema : dict
            keys (`str`) are the names of the features, values are `Feature` objects

        Returns
        -------
        db : Database
            an instance of the created db

        """
        # get the set of file extensions from the features and instantiate a walker
        exts = {f.__ext__ for f in schema.values() if getattr(f, '__ext__', False)}
        walker = FileWalker(exts, sources)
        # run the extraction job
        make_root_db(db_name, walker, partial(cls._load, schema=schema))
        # add post-build features
        db = cls(db_name)
        for f_name, f in schema.items():
            # let features the chance to update them selves confronted to their whole set
            if getattr(type(f), "post_create", Feature.post_create) != Feature.post_create:
                rv = f.after_create(db, f_name)
                if isinstance(rv, np.ndarray):
                    rv = (rv, getattr(db, f_name).regions)
                elif isinstance(rv, Regions):
                    rv = (getattr(db, f_name)[:], rv)
                elif tuple(*map(type, rv)) == (np.ndarray, Regions):
                    pass
                write_feature(db_name, f_name, getattr(f, 'params', {}),
                              *rv, files=getattr(db, f_name).files)
        db = cls(db_name)
        db.schema = schema
        return db

    @classmethod
    def make_temp(cls, roots=None, files=None, **kwargs):
        return cls.make("/tmp/%s-tmp.h5" % cls.__name__, roots, files, **kwargs)

    def _visit(self, func=print):
        with h5py.File(self.h5_file, "r") as f:
            f.visititems(func)

    def _register_features(self, names, keep_open):
        for name in names:
            setattr(self, name, FeatureProxy(self.h5_file, name, keep_open))
            setattr(self.params, name, getattr(self, name).attrs)
        return self

    def _register_dataframes(self, names):
        for name in names:
            try:
                df = pd.read_hdf(self.h5_file, key=name)
            except KeyError:
                df = pd.DataFrame()
            setattr(self, name, df)
        return self

    def __repr__(self):
        return "<Database: '%s'>" % os.path.split(self.h5_file)[-1]

    def bind(self, dataset_cls):
        """
        extends and if necessary overwrites the class of this instance with `dataset_cls`

        if `dataset_cls` contains a `__init__` method, it must take 2 arguments (self, model=None)
        and will be renamed to `prepare_dataset`.

        for example usage of this method, see `mimikit.data.DataModule.prepare_data()`

        Parameters
        ----------
        dataset_cls: type
            the interface to be added to the class of self

        Returns
        -------
        self
            the extended instance
        """
        def _bind(func, as_name=None):
            """
            Bind the function *func* to the class of *self*
            """
            if as_name is None:
                as_name = func.__name__
            setattr(type(self), as_name, func)

        for k, v in dataset_cls.__dict__.items():
            if k in ('__init__', 'prepare_dataset'):
                assert v.__code__.co_argcount == 2, \
                    "__init__ method of dataset_cls should take 2 arguments: self, model=None"
                # rename init
                _bind(v, 'prepare_dataset')
            elif k in ('__len__', '__getitem__') or k not in type(self).__dict__:
                if '__call__' in dir(v):
                    _bind(v, k)
                else:
                    # class attributes are likewise attached to type(self)
                    setattr(type(self), k, v)

        return self

    def prepare_dataset(self, model: pl.LightningModule):
        """
        placeholder for implementing what need to be done before serving data

        Parameters
        ----------
        model : pl.LightningModule
            The model that will consume this dataset.

        Returns
        -------
        None
        """
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError

    def __getitem__(self, item):
        raise NotImplementedError

    def split(self, splits):
        """
        performs random splits on self

        Parameters
        ----------
        splits: Sequence of floats or ints possibly containing None.
            The sequence of elements corresponds to the proportion (floats), the number of examples (ints) or the absence of
            train-set, validation-set, test-set, other sets... in that order.

        Returns
        -------
        splits: tuple
            the *-sets
        """
        nones = []
        if any(x is None for x in splits):
            if splits[0] is None:
                raise ValueError("the train-set's split cannot be None")
            nones = [i for i, x in zip(range(len(splits)), splits) if x is None]
            splits = [x for x in splits if x is not None]
        if all(type(x) is float for x in splits):
            splits = [x / sum(splits) for x in splits]
            N = len(self)
            # leave the last one out for now because of rounding
            as_ints = [int(N * x) for x in splits[:-1]]
            # check that the last is not zero
            if N - sum(as_ints) == 0:
                raise ValueError("the last split rounded to zero element. Please provide a greater float or consider "
                                 "passing ints.")
            as_ints += [N - sum(as_ints)]
            splits = as_ints
        sets = list(random_split(self, splits))
        if any(nones):
            sets = [None if i in nones else sets.pop(0) for i in range(len(sets + nones))]
        return tuple(sets)

    def to_tensor(self):
        for feat in self.features:
            as_tensor = self._to_tensor(getattr(self, feat))
            setattr(self, feat, as_tensor)
        return self

    def to(self, device):
        for feat in self.features:
            self._to(getattr(self, feat), device)
        return self

    @staticmethod
    def _to_tensor(obj):
        if type(obj) is torch.Tensor:
            return obj
        # converting obj[:] makes sure we get the data out of any db.feature object
        maybe_tensor = default_convert(obj[:])
        if type(maybe_tensor) is torch.Tensor:
            return maybe_tensor
        try:
            obj = torch.tensor(obj)
        except Exception as e:
            raise e
        return obj

    @staticmethod
    def _to(obj, device):
        """move any underlying tensor to some device"""
        if getattr(obj, "to", False):
            return obj.to(device)
        raise TypeError("object %s has no `to()` attribute" % str(obj))


def join(target_name, feature_names, databases):

    with h5py.File(target_name, "w") as f:
        for feat_name in feature_names:
            lengths = [getattr(db, feat_name).shape[0] for db in databases]
            dim = set([getattr(db, feat_name).shape[1] for db in databases]).pop()
            layout = h5py.VirtualLayout(shape=(sum(lengths), dim))
            offset = 0
            for i, n in enumerate(lengths):
                vsource = h5py.VirtualSource(databases[i].h5_file, feat_name, shape=(n, dim))
                layout[offset:offset + n] = vsource
                offset += n
            ds = f.create_virtual_dataset(feat_name, layout)
            ds.attrs = getattr(databases[0], feat_name).attrs
