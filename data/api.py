import h5py
import numpy as np
import pandas as pd
from .metadata import Metadata


class FeatureProxy(object):

    def __init__(self, h5_file, ds_name):
        self.h5_file = h5_file
        self.name = ds_name
        with h5py.File(h5_file, "r") as f:
            ds = f[self.name]
            self.N = ds.shape[0]
            self.dim = ds.shape[1]
            self.attrs = {k: v for k, v in ds.attrs.items()}

    def __len__(self):
        return self.N

    def __getitem__(self, item):
        with h5py.File(self.h5_file, "r") as f:
            rv = f[self.name][item]
        return rv

    def get(self, metadata):
        t_axis = self.attrs.get("time_axis", 0)
        slices = metadata.slices(t_axis)
        return np.concatenate(tuple(self[slice_i] for slice_i in slices), axis=t_axis)


class Database(object):
    def __init__(self, h5_file):
        self.h5_file = h5_file
        self.info = self._get_dataframe("/info")
        self.metadata = Metadata(self._get_dataframe("/metadata"))
        with h5py.File(h5_file, "r") as f:
            # add found features as self.feature_name = FeatureProxy(self.h5_file, feature_name)
            self._register_features(self.features)
            self._register_dataframes(self.dataframes)

    @property
    def features(self):
        names = self.info.iloc[:, 2:].T.index.get_level_values(0)
        return list(set(names))

    @property
    def dataframes(self):
        keys = set()

        def func(k, v):
            if "pandas_type" in v.attrs.keys() and k.split("/")[0] not in ("layouts", "info", "metadata"):
                keys.add(k)
            return None

        self.visit(func)
        return list(keys)

    def visit(self, func=print):
        with h5py.File(self.h5_file, "r") as f:
            f.visititems(func)

    def _get_dataframe(self, key):
        try:
            return pd.read_hdf(self.h5_file, key=key)
        except KeyError:
            return pd.DataFrame()

    def save_dataframe(self, key, df):
        with h5py.File(self.h5_file, "r+") as f:
            if key in f:
                f.pop(key)
        df.to_hdf(self.h5_file, key=key, mode="r+")
        return self._get_dataframe(key)

    def layout_for(self, feature):
        with h5py.File(self.h5_file, "r") as f:
            if "layouts" not in f.keys():
                return pd.DataFrame()
        return self._get_dataframe("layouts/" + feature)

    def _register_features(self, names):
        for name in names:
            setattr(self, name, FeatureProxy(self.h5_file, name))
        return None

    def _register_dataframes(self, names):
        for name in names:
            setattr(self, name, self._get_dataframe(name))
        return None


