from functools import partial
import h5py
import numpy as np
import pandas as pd
import os

from .write import make_root_db
from .regions import Regions


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
        self.files = pd.read_hdf(h5_file, self.name + "_files", mode="r") if has_files else None
        self.regions = pd.read_hdf(h5_file, self.name + "_regions", mode="r") if has_regions else None
        self._f = h5py.File(h5_file, "r") if keep_open else None
        # sequence for restricting the data to specific indices
        self._idx_map = []

    def __len__(self):
        return self.N

    def __getitem__(self, item):
        if any(self._idx_map):
            item = self._idx_map[item]
        if self._f is not None:
            return self._f[self.name][item]
        with h5py.File(self.h5_file, "r") as f:
            rv = f[self.name][item]
        return rv

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
        >>>from mimikit import Database
        >>>X = Database("my-db.h5").fft
        # only get files 0, 3 & 7
        >>>X.get_regions(X.regions.iloc[[0, 3, 7]])

        """
        slices = regions.slices(0)
        return np.concatenate(tuple(self[slice_i] for slice_i in slices), axis=0)

    def close(self):
        if self._f is not None:
            self._f.close()
        return self

    def __repr__(self):
        return "<FeatureProxy: '%s/%s'>" % (self.h5_file, self.name)


class Database(object):
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
    def __init__(self, h5_file: str, keep_open=False):
        self.h5_file = h5_file
        with h5py.File(h5_file, "r") as f:
            self.attrs = {k: v for k, v in f.attrs.items()}
            self.features = f.attrs.get("features", ["fft"])
        # add found features as self.feature_name = FeatureProxy(self.h5_file, feature_name)
        self._register_features(self.features, keep_open)

    @staticmethod
    def extract(path, **kwargs):
        raise NotImplementedError

    @classmethod
    def make(cls, db_name, roots=None, files=None, **kwargs):
        make_root_db(db_name, roots, files, partial(cls.extract, **kwargs))
        return cls(db_name)

    @classmethod
    def make_temp(cls, roots=None, files=None, **kwargs):
        return cls.make("/tmp/%s-tmp.h5" % cls.__name__, roots, files, **kwargs)

    def _visit(self, func=print):
        with h5py.File(self.h5_file, "r") as f:
            f.visititems(func)

    def _register_features(self, names, keep_open):
        for name in names:
            setattr(self, name, FeatureProxy(self.h5_file, name, keep_open))
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


