import tempfile

import h5mapper as h5m
from torch.utils.data import ConcatDataset
import numpy as np


# from .extractor import Extractor
# from .functionals import *
# from ..utils import SOUND_FILE_REGEX


class LazyLoader:

    def __init__(self, file_path, n_samples, sr, item_length):
        self.file_path = file_path
        self.n_samples = n_samples
        self.sr = sr
        self.item_length = item_length

    def __getitem__(self, item):
        if isinstance(item, int):
            func = mmk.FileToSignal(sr=self.sr, offset=item / self.sr, duration=self.item_length / self.sr)
        elif isinstance(item, slice):
            func = mmk.FileToSignal(sr=self.sr, offset=item.start / self.sr, duration=(item.stop-item.start) / self.sr)
        elif isinstance(item, tuple):
            assert len(item) == 1
            return self.__getitem__(item[0])
        else:
            raise TypeError()
        return func(self.file_path)

    def __len__(self):
        return self.n_samples


class LazyFile:
    def __init__(self, loader):
        self.loader = loader


if __name__ == '__main__':
    import mimikit as mmk

    sr = 22050
    sources = list(h5m.FileWalker(".mp3", "../../../example-files"))


    class FileLength(h5m.Feature):
        def load(self, src):
            return np.array([mmk.FileToSignal(sr)(src).shape[0]])


    with tempfile.NamedTemporaryFile() as f:
        tf = h5m.TypedFile.create(f.name, sources, mode='w',
                                  schema=dict(n_samples=FileLength()), parallelism='mp',
                                  keep_open=True)
        meta = dict(zip(tf.index.keys(), tf.n_samples[:]))

    datasets = []
    for path, n_samples in meta.items():
        loader = LazyLoader(path, n_samples, sr, 1000)
        datasets += [h5m.ProgrammableDataset(
            LazyFile(loader),
            (h5m.Input(data='loader',
                       getter=h5m.AsSlice(length=5000),
                       transform=mmk.Envelop()),
             )
        )]
    ds = ConcatDataset(datasets)
    print(ds[20102][0].shape)
    print(len(ds) / sr)
