import torch
import numpy as np
from torch.utils.data import Dataset as TorchDataset, Subset
from torch.utils.data._utils.collate import default_convert

from functools import update_wrapper


def map_if_multi(attr):
    def decorator(fn):
        def wrapper(self, *arg, **kwargs):
            if isinstance(getattr(self, attr), tuple):
                return tuple(getattr(x, fn.__name__)(*arg, **kwargs) for x in getattr(self, attr))
            else:
                return fn(self, *arg, **kwargs)

        return update_wrapper(wrapper, fn,
                              # for some reason, __mod__ causes errors when using functools.wraps
                              ("__name__", '__qualname__', '__doc__', '__annotations__'),
                              ("__dict__",))

    return decorator


class Dataset(TorchDataset):

    @property
    def n_features(self):
        return self.n_features

    @property
    def dims(self):
        return self._shape

    @property
    def shape(self):
        return self._shape

    @property
    def dtype(self):
        return self._dtype

    @property
    def device(self):
        return self._device

    @property
    def style(self):
        return self._style

    @property
    def attrs(self):
        """
        @return: `attrs` : dict of user-specified infos he wants to keep close (e.g. {"sr": 22050, "time-axis": 0})
        """
        return self._attrs

    def __init__(self,
                 data_object=None,
                 wrappers=None,
                 augmentations=None,
                 attrs=None):
        """
        Here we initialize the instance.
        """
        self._object = data_object
        self._wrapper = wrappers
        self._augmentation = augmentations
        self._attrs = attrs
        self._n_features = 1 if not isinstance(data_object, tuple) else len(data_object)
        self._is_multi = self._n_features > 1

        if self._is_multi:
            # we cast to Dataset recursively
            self._object = tuple(x if isinstance(x, Dataset) else Dataset(x) for x in self._object)

        self._style = self.get_style()
        if self._style is None:
            raise ValueError("Expected data_object to either implement __getitem__ and __len__, or __iter__."
                             " object of class %s implements none of those." % str(type(data_object)))
        if self._is_multi:
            if not all(self._style[0] == style for style in self._style[1:]):
                raise TypeError("Expected all data_objects to be of the same style. Got %s" % str(self._style))
            lengths = tuple(len(obj) for obj in self._object[1:])
            if self._style[0] is "map" and not all(lengths[0] == n for n in lengths[1:]):
                raise ValueError("Expected all 'map-style' data_objects to be of same lengths. Got %s" % \
                                 str(lengths))
        self._shape = self.get_shape()
        self._device = self.get_device()
        self._dtype = self.get_dtype()

    def __call__(self, *args, **kwargs):
        """
        Method used to wrap around an other Dataset. Only used in the wrapper classes.
        """
        pass

    def __len__(self):
        return len(self._object[0]) if self._is_multi else len(self._object)

    @map_if_multi("_object")
    def __getitem__(self, item):
        if isinstance(item, tuple):
            return tuple(self._object[i] for i in item)
        return self._object[item]

    def __iter__(self):
        return iter(self._object)

    @staticmethod
    def has_len(data_object):
        return getattr(data_object, "__len__", False)

    @staticmethod
    def _get_style(data_object):
        if getattr(data_object, "__getitem__", False) and getattr(data_object, "__len__", False):
            return "map"
        elif getattr(data_object, "__iter__", False):
            return "iter"
        return None

    @map_if_multi("_object")
    def get_style(self):
        return Dataset._get_style(self._object)

    @staticmethod
    def _get_elem(data_object):
        if Dataset._get_style(data_object) == "map":
            return data_object[0]
        elif Dataset._get_style(data_object) == "iter":
            return next(iter(data_object))
        return None

    @map_if_multi("_object")
    def get_elem(self):
        return Dataset._get_elem(self._object)

    @staticmethod
    def _get_shape(data_object):
        shape = getattr(data_object, "shape", None)
        if shape is None:
            # we try to fet lengths recursively and add `None` when we get elements with no length
            # thus, in the case of a hierarchy of generators, returning at least the number of 'axes'
            N = len(data_object) if Dataset.has_len(data_object) else None
            shape = (N, )
            elem = Dataset._get_elem(data_object)
            while elem is not None:
                shape = (*shape, len(elem) if Dataset.has_len(elem) else None)
                elem = Dataset._get_elem(elem)
            # last None in the shape is bottom-level element
            shape = shape[:-1]
        return shape

    @map_if_multi("_object")
    def get_shape(self):
        return Dataset._get_shape(self._object)

    @staticmethod
    def _get_dtype(data_object):
        elem = Dataset._get_elem(data_object)
        dtype = getattr(data_object, "dtype", None)
        if dtype is None:
            dtype = getattr(elem, "dtype", None) or (type(elem) if type(elem) in (int, float, complex) else None)
            while dtype is None and elem is not None:
                elem = Dataset._get_elem(elem)
                dtype = getattr(elem, "dtype", None) or (type(elem) if type(elem) in (int, float, complex) else None)
            if dtype is None:
                raise ValueError("Couldn't figure out the dtype of data_object...")
            elif dtype in (float, int, complex):
                dtype = getattr(torch, np.dtype(dtype).name, None)
            elif getattr(dtype, "name", False):  # it's a numpy array!
                dtype = getattr(torch, dtype.name, dtype)
        return dtype

    @map_if_multi("_object")
    def get_dtype(self):
        return Dataset._get_dtype(self._object)

    @staticmethod
    def _get_device(data_object):
        elem = Dataset._get_elem(data_object)
        device = getattr(data_object, "device", None)
        if device is None:
            device = getattr(elem, "device", "cpu")
        return device

    @map_if_multi("_object")
    def get_device(self):
        return Dataset._get_device(self._object)

    @map_if_multi("_object")
    def to_tensor(self):
        if issubclass(type(self._object), torch.Tensor):
            return self
        elif self._style == "iter":
            raise TypeError("Cannot convert 'iter' style objects to tensor.")
        maybe_tensor = default_convert(self._object[:])
        if issubclass(type(maybe_tensor), torch.Tensor):
            self._object = maybe_tensor
            return self
        else:
            raise RuntimeWarning("torch couldn't convert data_object to tensor. data_object is still of "
                                 "class %s" % str(type(self._object)))

    @map_if_multi("_object")
    def select(self, indices, inplace=False):
        if not self.has_len(self._object):
            raise TypeError("Cannot select indices of objects that don't implement `__len__`")
        if inplace:
            self._object = self._object[np.sort(indices)]
        else:  # TODO : Should be a proper wrapper:
            self._object = Subset(self._object, indices)
        return self

    @map_if_multi("_object")
    def to(self, device):
        """move any underlying tensor to some device"""
        pass

    def download_source(self, *args, **kwargs):
        """method to download a single source.
        @returns: Dataset object"""
        pass

    def load_file(self, path):
        """method to load a single file.
        Here's the place to define how an audio file should be transformed (e.g. stft) and to extract some features from
        it (e.g. segments' or filenames' labels).
        @returns: Dataset object"""
        pass

    def wrap_in(self, *wrappers):
        """add batch definition, augmentations (and transforms?) with wrapper(s)"""
        pass

    def split(self, splits):
        pass

    def load(self, **kwargs):
        """pack self into a batch producer (Dataloader)"""
        pass

    def transform(self, function):
        """transform any underlying tensor"""
        pass

    def augment(self, *augmentations):
        """`augmentations` are tuples whose first elements are probabilities (0. < float <= 1) for the augmentations
        to occur during training and the second elements, functions that takes a batch as single argument
         and return a transformed batch - the augmentation"""
        pass

    def random_example(self, n):
        """get n random examples"""
        pass
