import torch
from torch.utils.data import Dataset as TorchDataset, IterableDataset, TensorDataset, Subset, DataLoader, \
    random_split
import pytorch_lightning as pl
import numpy as np
from typing import Sequence, Tuple, Generator, NewType, Union
from multiprocessing import cpu_count

from mmk.data import FeatureProxy
from mmk.kit.ds_wrappers import DatasetWrapper, IterableDatasetWrapper, is_map_style_dataset
from mmk.data.metadata import Metadata


def is_wrapper_ds(dataset):
    return issubclass(type(dataset), DatasetWrapper) or issubclass(type(dataset), IterableDatasetWrapper)


def zip_stack(batch_lists):
    """
    returns (stack(feat_1), stack(feat_2)...)
    """
    rv = tuple(torch.stack(x) for x in zip(*batch_lists))
    return rv


def maybe_not_tensor(x):
    x_class = type(x)
    if issubclass(x_class, torch.Tensor):
        pass
    elif issubclass(x_class, np.ndarray):
        x = torch.from_numpy(x)
    else:
        # First, try to instantiate a tensor from feat, then try to select the subset
        try:
            x = torch.tensor(x, requires_grad=False)
        except Exception as e:
            raise ValueError("Couldn't instantiate a `torch.Tensor` with the provided `feature` argument."
                             "Call to torch.tensor raised following Exception : \n" + str(e))
    return x


class SingleTensorDataset(TensorDataset):
    """
    torch's class has a getitem that returns tuples of tuples...
    """

    def __getitem__(self, item):
        rv = super(SingleTensorDataset, self).__getitem__(item)
        return rv[0]


class H5Dataset(TorchDataset):
    """
    just wraps a FeatureProxy in a Dataset class
    """

    def __init__(self, feature_proxy):
        self.ds = feature_proxy

    def __getitem__(self, item):
        return torch.from_numpy(self.ds[item])

    def __len__(self):
        return len(self.ds)


class GeneratorDataset(IterableDataset):
    """
    just wrap a Generator in an IterableDataset
    """

    def __getitem__(self, index):
        pass

    def __init__(self, generator):
        self.generator = generator

    def __iter__(self):
        return iter(self.generator)


Feature = NewType("Feature",
                  Union[TorchDataset, IterableDataset, FeatureProxy,
                        np.ndarray, torch.Tensor, Sequence, Generator])


class Dataset(TorchDataset):

    @property
    def n_features(self):
        return len(self._shape)

    @property
    def dims(self):
        return self._shape

    @property
    def shape(self):
        return self._shape

    def size(self):
        return self._shape

    @property
    def dtype(self):
        return self._dtype

    @property
    def device(self):
        return self._device

    @property
    def indices(self):
        return self._indices

    @property
    def style(self):
        return is_map_style_dataset(self)

    @property
    def attrs(self):
        """
        @return: `attrs` : dict of user-specified infos he wants to keep close (e.g. {"sr": 22050, "time-axis": 0})
        """
        return self._attrs

    @staticmethod
    def __new__(cls,
                sources=None,  # remote locations of data
                files=None,   # local paths to data
                objects=None,   # 'datasetable' objects like arrays, tensors or generators
                datasets=None,  # Datasets we want to wrap or augment
                wrappers=None,  # wrappers to specify batch creation
                augmentations=None,   # see the docstring of the `augment` method
                ):
        """
        create a new Dataset either from sources, from files, from python objects or from Datasets.
        Having instantiation here allows to define subclasses that explicitly deal only with sources or only with files
        and to instantiate the class most appropriate to the user's input. It's also handy for declaring Datasets
        on the fly at any step of the workflow described below in the "WORKFLOWS" note.
        """
        pass

    def __init__(self):
        """
        Here we initialize the instance.
        """
        self._source = None
        self._file = None
        self._object = None
        self._wrapper = None
        self._augmentation = None
        self._shape = tuple()
        self._dtype = tuple()
        self._device = tuple()
        self._indices = None
        self._attrs = dict()

    def __call__(self, *args, **kwargs):
        """
        Method used to wrap around an other Dataset. Only used in the wrapper classes.
        """
        pass

    def __len__(self):
        pass

    def __getitem__(self, item):
        pass

    def __iter__(self):
        pass

    """
    WORKFLOWS :
    The Idea behind the four following methods is two-fold :
    
    1. we want to enable 2 different workflows. One where we fetch data prior to training and this could mean download 
    data to disks, or load files as tensors into RAM. The other where we only specify a promise that will be executed 
    during training ; for instance, we collect file-names and file-lengths prior to training but we load them as tensors
    into the RAM only when they're requested by the Trainer during training.
    
    2. In either workflow, we need to execute tasks that should run in parallel workers/threads and return objects that
    allow us to keep track of properties of the underlying data, e.g. shapes and lengths. Both requirements being already
    full-filled by torch's synergy between Dataloaders and Datasets, we only require that the methods executed
    in parallel (`download_source` and `load_file`) return Dataset objects!
    Then, we can flexibly define any data-fetching pipeline by simply specifying which steps/methods should be run 
    prior to training and which steps/methods should be kept as promises. Simple, efficient, and without any 
    "yet-an-other-API" overhead : at any step, we use the same slight extension of torch's Dataset! 
    """

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

    def prepare(self):
        """perform here the steps that create files prior to training. For instance, download from neptune/youtube or
        aggregate transformed files into a .h5 DB.
        Base class implementation of this method creates a simple multiprocessing Pool that calls
        `download_source` on all the sources if some were passed to the constructor."""
        return None

    def setup(self):
        """perform here the steps that create python objects (e.g. Datasets of Tensors, generators...)
         prior to training.
         Base class implementation of this method creates a simple multiprocessing Pool that calls
        `load_file` on all the files created by `prepare()` or passed to the constructor, it then packs the results into
        an appropriate Dataset class and concatenate all of them into a single Dataset object"""
        return None

    def wraps_in(self, *wrappers):
        """add batch definition, augmentations (and transforms?) with wrapper(s)"""
        pass

    def transform_item(self, item):
        """callback to do something with the items before they are passed to __getitem__"""
        pass

    def load(self, **kwargs):
        """pack self into a batch producer (Dataloader)"""
        pass

    def to(self, device):
        """move any underlying tensor to some device"""
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


class MMKDataModule(pl.LightningDataModule):

    def __init__(self,
                 feature: [Feature, Tuple[Feature]],
                 subset_idx: [np.ndarray, torch.Tensor, Metadata, Sequence] = None,
                 malloc_device: [torch.device, str, Sequence] = None,
                 ds_wrapper: [DatasetWrapper, IterableDatasetWrapper] = None,
                 train_val_split=1.,
                 **loader_kwargs,
                 ):
        super(MMKDataModule, self).__init__()
        self.feature = feature
        self.subset_idx = subset_idx
        self.malloc_device = malloc_device
        self.ds_wrapper = ds_wrapper
        self.train_val_split = train_val_split
        self.loader_kwargs = loader_kwargs
        self.train_ds, self.val_ds, self.test_ds = None, None, None
        self.train_loader, self.val_loader, self.test_loader = None, None, None

    def prepare_data(self, *args, **kwargs):
        # Reminder : do not assign state in this method (e.g. self.x = y)
        pass

    def setup(self, stage=None):
        dataset = self._prepare_feature()
        # this is the 'official' attributes for this
        self.dims = self.infer_dims(dataset)
        if is_map_style_dataset(dataset):
            tr_length = int(self.train_val_split * len(dataset))
            lengths = (tr_length, len(dataset) - tr_length)
            self.train_ds, self.val_ds = random_split(dataset, lengths)
        else:  # it's an iterable, we can't split...
            self.train_ds, self.val_ds = dataset, None
        self.loader_kwargs = self._set_defaults_loader_kwargs(dataset)

    def train_dataloader(self, *args, **kwargs):
        return DataLoader(self.train_ds, **self.loader_kwargs)

    def val_dataloader(self, *args, **kwargs):
        if self.val_ds is not None:
            return DataLoader(self.val_ds, **self.loader_kwargs)
        return None

    def test_dataloader(self, *args, **kwargs):
        if self.test_ds is not None:
            return DataLoader(self.test_ds, **self.loader_kwargs)
        return None

    def transfer_batch_to_device(self, batch, device):
        return super(MMKDataModule, self).transfer_batch_to_device(batch, device)

    ###############################
    # Machinery to prepare features :
    ###############################

    def _prepare_feature(self):
        """
        this is the high-level method (called in setup()) that "converts" whatever the user passed as `feature` argument
        into a valid Dataset for torch's Dataloader.
        if `feature` is a tuple of Feature, it will first convert each element into a Dataset and the resulting Datasets
        will be wrapped into a DatasetWrapper.
        For details about the conversion process, please refer to the next method `_prepare_single_feature`

        @return: `self.feature` packed into a subclass of Dataset
        """
        feature, subset_idx = self.feature, self.subset_idx
        malloc_device, ds_wrapper = self.malloc_device, self.ds_wrapper

        if isinstance(feature, tuple):
            dataset = tuple(self._prepare_single_feature(feat, subset_idx, malloc_device) for feat in feature)
        else:
            dataset = self._prepare_single_feature(feature, subset_idx, malloc_device)

        if ds_wrapper is not None:
            if not isinstance(dataset, tuple):
                dataset = (dataset,)
            if self._is_valid_ds_tuple(dataset):
                dataset = ds_wrapper(*dataset)
        elif ds_wrapper is None and isinstance(dataset, tuple) and self._is_valid_ds_tuple(dataset):
            # by default, pack multiple features in a default wrapper
            if is_map_style_dataset(dataset[0]):
                dataset = DatasetWrapper()(*dataset)
            else:
                dataset = IterableDatasetWrapper()(*dataset)

        return dataset

    def _prepare_single_feature(self, feature, subset_idx, malloc_device):
        """
        in pseudo-code:
        if malloc:
            feature = tensor(feature)[subset].to(device)
        dataset = Dataset(feature)
        if subset and not malloc:
            dataset = Subset(dataset, subset)
        return dataset
        """
        if malloc_device is not None:
            if issubclass(type(feature), TorchDataset):
                raise ValueError("Cannot allocate a %s to a device's memory." % type(feature) +
                                 " Please set `malloc_device` to None "
                                 "or pass a `feature` object of one of the following type: "
                                 "mmk.data.FeatureProxy, np.ndarray, torch.Tensor, Sequence")
            feature = self._feature_to_tensor(feature, subset_idx)
            feature = self._alloc_feature(feature, malloc_device)

        dataset = self._feature_to_dataset(feature)

        if malloc_device is None and subset_idx is not None:
            if issubclass(type(dataset), IterableDataset):
                raise ValueError("Cannot take a subset of an IterableDataset. Please set `subset_idx` to None or "
                                 "pass a feature of another class.")
            subset_idx = MMKDataModule._subset_idx_to_indexing_object(subset_idx)
            dataset = Subset(dataset, subset_idx)

        return dataset

    @staticmethod
    def _feature_to_tensor(feature, subset_idx):
        feat_class = type(feature)
        subset_idx = MMKDataModule._subset_idx_to_indexing_object(subset_idx)

        # because h5 feature can be huge and need sorted indices,
        # we single this case out and return the "subseted" feature right away
        if issubclass(feat_class, FeatureProxy):
            if isinstance(subset_idx, Sequence):
                subset_idx = np.sort(subset_idx)
            return torch.from_numpy(feature[subset_idx])

        # otherwise, we first attempt to build a tensor before taking the subset
        feature = maybe_not_tensor(feature)

        return feature[subset_idx]

    @staticmethod
    def _alloc_feature(feature, device):
        if device is None:
            return
        if not isinstance(device, torch.device):
            if not isinstance(device, str) and isinstance(device, Sequence):
                device = torch.device(*device)
            else:
                try:
                    device = torch.device(device)
                except Exception as e:
                    raise ValueError("Couldn't instantiate a `torch.device` with the provided `malloc_device` argument."
                                     "Call to torch.device raised following Exception : \n" + str(e))
        return feature.to(device)

    @staticmethod
    def _feature_to_dataset(feature):
        feat_class = type(feature)
        if issubclass(feat_class, TorchDataset):
            dataset = feature
        elif issubclass(feat_class, torch.Tensor):
            dataset = SingleTensorDataset(feature)
        elif issubclass(feat_class, np.ndarray):
            dataset = SingleTensorDataset(torch.from_numpy(feature))
        elif issubclass(feat_class, FeatureProxy):
            dataset = H5Dataset(feature)
        elif issubclass(feat_class, Sequence):
            dataset = SingleTensorDataset(torch.tensor(feature, requires_grad=False))
        elif issubclass(feat_class, Generator):
            dataset = GeneratorDataset(feature)
        else:
            raise TypeError("Cannot instantiate a Dataset with a `feature` object"
                            " of type `" + str(feat_class) + "`")
        return dataset

    @staticmethod
    def _wrap_dataset(feat, wrapper):
        if issubclass(wrapper, DatasetWrapper):
            if not isinstance(feat, tuple):
                feat = (feat,)
            feat = wrapper(*feat)
        else:
            raise TypeError("Expected `ds_wrapper` to be a subclass of `DatasetWrapper` or `IterableDatasetWrapper`,"
                            " got : `%s`" % str(type(wrapper)))
        return feat

    @staticmethod
    def _subset_idx_to_indexing_object(subset_idx):
        if subset_idx is None:
            subset_idx = slice(None, None, None)
        elif isinstance(subset_idx, Metadata):
            subset_idx = subset_idx.all_indices
        # hopefully, those cover all valid cases...
        elif type(subset_idx) in (np.ndarray, list, torch.Tensor, slice, tuple):
            pass
        else:
            raise TypeError("Cannot construct a valid indexing object"
                            " with the provided `subset_idx` argument of type `" +
                            str(type(subset_idx)) + "`")
        return subset_idx

    @staticmethod
    def _is_valid_ds_tuple(datasets):
        types = [type(ds) for ds in datasets]
        if any(is_map_style_dataset(ds) for ds in datasets) \
                and any(issubclass(t, IterableDataset) for t in types):
            raise ValueError("argument `feature` cannot mix map-style and iterable datasets.")
        return True

    def _set_defaults_loader_kwargs(self, dataset):
        """
        if the user didn't explicitly set a parameter that needs to (or should) be set, we do it for him.
        Specifically:
            - if the ds will return Sequences of Features, we need an appropriate collate_fn
            - if the ds isn't mallocated, we should use several workers and pin_memory
        """
        kwargs = self.loader_kwargs
        # if the batch_size is explicitly None, we assume that the user doesn't want us to fiddle with his batch
        # regardless whether or not he specified a collate_fn.
        if is_wrapper_ds(dataset) and kwargs.get("batch_size", 1) is not None:
            kwargs.setdefault("collate_fn", zip_stack)
        if self.malloc_device is None:
            kwargs.setdefault("pin_memory", True)
            kwargs.setdefault("num_workers", cpu_count())
        return kwargs

    @staticmethod
    def infer_dims(dataset):
        """returns all the shapes found in the first example of `dataset` as a list"""
        if is_map_style_dataset(dataset):
            example = dataset[0]
        else:
            example = next(iter(dataset))
        if isinstance(example, tuple):
            return [x.shape for x in example]
        else:
            return [example.shape]
