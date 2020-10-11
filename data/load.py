import torch
from torch.utils.data import Sampler, BatchSampler, Dataset, DataLoader
import numpy as np
import h5py
from .api import FeatureProxy


class FlatSampler(Sampler):
    """
    returns INT indices
    """
    def __init__(self, metadata, shuffle=True):
        super(FlatSampler, self).__init__([0])
        self.indices = metadata.flat
        self.N = self.indices.size
        self.shuffle = shuffle

    def __iter__(self):
        if self.shuffle:
            np.random.shuffle(self.indices)
        return iter(self.indices)

    def __len__(self):
        return self.N


class EventSampler(Sampler):
    """
    returns EVENT indices
    """
    def __init__(self, metadata, shuffle=True):
        super(EventSampler, self).__init__([0])
        self.metadata = metadata
        self.N = len(metadata)
        self.shuffle = shuffle

    def __iter__(self):
        if self.shuffle:
            self.metadata = self.metadata.sample(frac=1.)
        return self.metadata.events

    def __len__(self):
        return self.N


class FrameSampler(Sampler):
    """
    returns SLICE indices
    """
    def __init__(self, N, k=1, stride=1, shifts=tuple(), shuffle=True):
        super(FrameSampler, self).__init__([0])
        self.base_idx = np.arange(N - k - sum(shifts) + 1, step=stride)
        self.k = k
        self.N = len(self.base_idx)
        self.shuffle = shuffle

    def __iter__(self):
        if self.shuffle:
            np.random.shuffle(self.base_idx)
        return iter(slice(i, i+self.k) for i in self.base_idx)

    def __len__(self):
        return self.N


def get_event(x, event, shift=0):
    if type(event) is slice or "Event" in str(type(event)):
        slc = slice(event.start+shift, event.stop+shift)
        return x[slc]
    return x[event+shift]


def zip_stack(batch_lists):
    """
    returns (stack(feat_1), stack(feat_2)...)
    """
    rv = tuple(torch.stack(x) for x in zip(*batch_lists))
    return rv


def zip_list(batch_lists):
    return tuple(x for x in zip(*batch_lists))


class DSBase(Dataset):
    def __init__(self):
        self.N = None

    def __len__(self):
        return self.N

    def set_length(self, N):
        self.N = N
        return self


class OpenedDS(DSBase):
    def __init__(self, file, ds_name, device="cpu", shift=0):
        self.file = h5py.File(file, "r")
        self.ds = self.file[ds_name]
        self.shape = self.ds.shape
        self.shift = shift
        self.device = device
        super(OpenedDS, self).__init__()

    def __getitem__(self, event):
        array = get_event(self.ds, event, self.shift)
        return torch.from_numpy(array)


class TensorSource(DSBase):
    def __init__(self, src, shift=0):
        self.src = src
        self.shape = self.src.shape
        self.shift = shift
        super(TensorSource, self).__init__()

    def __getitem__(self, event):
        return get_event(self.src, event, self.shift)


class MultiSource(DSBase):
    def __init__(self, *sources):
        self.sources = sources
        super(MultiSource, self).__init__()

    def __getitem__(self, event):
        return tuple(src[event] for src in self.sources)


def prepare_data(source, metadata, pre_cat=False, device="cpu", shift=0):
    if isinstance(source, FeatureProxy):
        if pre_cat:
            source = torch.from_numpy(source.get(metadata)).to(device)
            source = TensorSource(source, shift)
        else:
            source = OpenedDS(source.h5_file, source.name, device, shift)
    else:
        if not pre_cat:
            # TODO : log a warning (?)
            pass
        if type(source) is np.ndarray:
            source = torch.from_numpy(source).to(device)
        if type(source) is torch.Tensor:
            source = source.to(device)
        if pre_cat:
            source = torch.cat([source[i] for i in metadata.slices(time_axis=0)])
        source = TensorSource(source, shift)
    return source


def prepare_sampler(mode, metadata, shifts, shuffle, **kwargs):
    if mode == "flat":
        sampler = FlatSampler(metadata, shuffle)
    elif mode == "event":
        sampler = EventSampler(metadata, shuffle)
    elif mode == "frame":
        if not metadata.is_contiguous():
            print("WARNING: metadata is not contiguous which "
                  " leads to errors and/or undesired results when using the mode='frame'")
        sampler = FrameSampler(metadata.span, shifts=shifts, shuffle=shuffle, **kwargs)
    else:
        raise ValueError("`mode` value '%s' not recognized. Must be one of 'flat', 'event'" % mode)
    return sampler


def load(features, metadata, mode,
         pre_cat=False, device="cpu", shifts=None,
         shuffle=True, batch_size=None, drop_last=False,
         num_workers=0,
         **frame_sampler_kwargs):

    # prepare
    features = tuple(prepare_data(src, metadata, pre_cat, device, shft)
                     for src, shft in zip(features, shifts if shifts is not None else [0] * len(features)))
    if pre_cat:
        metadata.make_contiguous()
    sampler = prepare_sampler(mode, metadata, shifts, shuffle, **frame_sampler_kwargs)

    # batches are ALWAYS tuples of tensor
    features = MultiSource(*features).set_length(len(sampler))

    # source + sampler -> DataLoader
    if batch_size is not None:
        sampler = BatchSampler(sampler, batch_size, drop_last)
        return DataLoader(features, batch_sampler=sampler, collate_fn=zip_stack, num_workers=num_workers)
    return DataLoader(features, sampler=sampler, collate_fn=zip_stack, num_workers=num_workers)
