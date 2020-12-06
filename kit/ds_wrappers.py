from torch.utils.data import Dataset, IterableDataset


# Dataset (map-style) is a parent of IterableDataset, so we need a strong test
def is_map_style_dataset(dataset):
    tp = type(dataset)
    return issubclass(tp, Dataset) and not issubclass(tp, IterableDataset)


class DatasetWrapper(Dataset):
    """
    Base wrapper for map-style Datasets.
    the gist of this kind of wrapper is to be able to configure the wrapping in __init__
    and then to bind the datasets int the wrapper through the __call__ method.
    E.g. :
    features = (Dataset(inputs), Dataset(targets))
    wrapper = MaybeScrambledTargetsWrapper(pr_scramble=0.5)
    dataset = wrapper(*features)

    This base class could be named 'ZipDataset' (in the spirit of torch's 'ConcatDataset') since
    __getitem__ packs data-points from several datasets, but for the same item, in a tuple (ds_1[i], ds_2[i], ...)

    IMPORTANT NOTE: Please make sure to call the super's __call__ method when subclassing
    in order to validate the datasets passed as arguments or you'll likely end up with exceptions all over the place!
    """

    def __init__(self, *args, **kwargs):
        self.datasets = None

    def __call__(self, *datasets: Dataset):
        if not all(is_map_style_dataset(ds) for ds in datasets):
            raise ValueError("Expected all datasets to be map-style datasets, got following types: %s"
                             % str([type(ds) for ds in datasets]))
        assert all(len(datasets[0]) == len(ds) for ds in datasets), "All Datasets must have the same length."
        self.datasets = datasets
        return self

    def __getitem__(self, item):
        return tuple(ds[item] for ds in self.datasets)

    def __len__(self):
        return len(self.datasets[0])


class IterableDatasetWrapper(IterableDataset):
    """
    Base wrapper for IterableDatasets.

    IMPORTANT NOTE: Please make sure to call the super's __call__ method when subclassing
    in order to validate the datasets passed as arguments or you'll likely end up with exceptions all over the place!
    """

    def __init__(self, *args, **kwargs):
        self.datasets = None

    def __call__(self, *datasets: IterableDataset):
        if not all(issubclass(type(ds), IterableDataset) for ds in datasets):
            raise ValueError("Expected all datasets to be subclasses of `IterableDataset`, got following types: %s"
                             % str([type(ds) for ds in datasets]))
        self.datasets = datasets
        return self

    def __iter__(self):
        return iter(zip(*tuple(iter(ds) for ds in self.datasets)))


class InputEqualTargetWrapper(DatasetWrapper):
    """
    Duplicates the pointer to a Dataset, to serve batches where inputs == targets efficiently (in terms of memory)
    """
    def __call__(self, *datasets: Dataset):
        super(InputEqualTargetWrapper, self).__call__(*datasets)
        if len(self.datasets) > 1:
            raise ValueError("Expected only one Dataset, got %i." % len(datasets))
        self.datasets = tuple([self.datasets[0]] * 2)
        return self


class ShiftedSeqsPairWrapper(InputEqualTargetWrapper):
    """
    First wraps a Feature in an AutoEncodingWrapper, then maps `item` in __getitem__ to shifted slices.
    for a standard language-modelling task, one would just need to do :
    wrapper = SequenceModelWrapper(sequence_length=128, shift=1)
    dataset = wrapper(Dataset(my_texts))
    """
    def __init__(self, sequence_length, shift, stride=1):
        super(ShiftedSeqsPairWrapper, self).__init__()
        self.shift = shift
        self.sequence_length = sequence_length
        self.stride = stride
        self.N = None

    def __call__(self, *datasets: Dataset):
        super(ShiftedSeqsPairWrapper, self).__call__(*datasets)
        self.N = len(self.datasets[0])

    def __len__(self):
        return (self.N - self.sequence_length - self.shift + 1) // self.stride

    def _item_to_slices(self, item):
        i = item * self.stride
        input_slice = slice(i, i + self.sequence_length)
        target_slice = slice(i + self.shift, i + self.shift + self.sequence_length)
        return input_slice, target_slice

    def __getitem__(self, item):
        return tuple(ds[idx] for ds, idx in
                     zip(self.datasets, self._item_to_slices(item)))
