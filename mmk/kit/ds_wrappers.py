from .dataset import Dataset
from copy import copy


class DSWrapper(Dataset):

    def __init__(self):
        pass

    def upgrade(self, dataset):
        """
        dynamically extends an object with own methods and attributes.
        @param dataset: object to be extended
        @return: extended object will be an instance of object.__class__ and of self.__class__
        """
        new = copy(dataset)
        bases = (self.__class__, new.__class__)
        name = self.__class__.__name__ + "Dataset"
        new.__dict__.update(self.__dict__)
        new.__class__ = type(name, bases, new.__dict__)
        return new

    def __call__(self, dataset: Dataset):
        return self.upgrade(dataset)


class InputEqualTarget(DSWrapper):
    def __getitem__(self, item):
        return tuple(self.data[i] for i in [item, item])

    def __len__(self):
        return len(self.data)


class ShiftedSeqsPair(DSWrapper):
    def __init__(self, sequence_length, shift, stride=1):
        self.shift = shift
        self.sequence_length = sequence_length
        self.stride = stride
        self.N = None

    def __call__(self, dataset: Dataset):
        # grab the number of time-steps BEFORE we upgrade
        self.N = len(dataset.data)
        return super(ShiftedSeqsPair, self).__call__(dataset)

    def __len__(self):
        return (self.N - self.sequence_length - self.shift + 1) // self.stride

    def __getitem__(self, item):
        i = item * self.stride
        input_slice = slice(i, i + self.sequence_length)
        target_slice = slice(i + self.shift, i + self.shift + self.sequence_length)
        return tuple(self.data[idx] for idx in [input_slice, target_slice])


