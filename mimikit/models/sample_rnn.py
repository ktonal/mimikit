import numpy as np
from numpy.lib.stride_tricks import as_strided as np_as_strided
from pytorch_lightning import LightningModule
import torch
import torch.nn.functional as F
import torch.nn as nn

from ..kit import DBDataset
from ..audios import transforms as A

from ..kit import SuperAdam, SequenceModel, DataSubModule

from ..kit.networks.sample_rnn import SampleRNNNetwork

from torch.utils.data import Sampler, RandomSampler, BatchSampler
import math


class ThreadedSampler(Sampler):
    def __init__(self, n_samples, batch_size=64, thread_length=8 * 16000, seq_len=512):
        super().__init__(None)
        self.n_samples = n_samples
        self.batch_size = batch_size
        self.thread_length = thread_length
        self.seq_len = seq_len
        self.n_chunks = self.n_samples // self.thread_length
        self.n_per_chunk = self.thread_length // self.seq_len

    def __iter__(self):

        smp = RandomSampler(torch.arange(self.n_chunks))
        for top in BatchSampler(smp, self.batch_size, True):
            for start in range(self.n_per_chunk):
                idx = tuple((t * self.thread_length) + (start * self.seq_len) for t in top)
                yield idx

    def __len__(self):
        return int(max(1, math.floor(self.n_chunks / self.batch_size)) * self.n_per_chunk)


class FramedSequences:

    def __init__(self, seq_len, frame_sizes):
        frame_sizes = list(frame_sizes[:-1])
        top_size = frame_sizes[0]
        bottom_size = frame_sizes[-1]

        self.seq_len = seq_len
        self.seq_shifts = [top_size - size for size in frame_sizes + [bottom_size]]
        self.seq_strides = frame_sizes + [1]
        frame_sizes += [bottom_size]
        self.frame_sizes = frame_sizes
        self.top_size = top_size

    def __call__(self, item):
        shapes = [dict(slice=slice(item + shift, item + shift + self.seq_len),
                       shape=(int(self.seq_len / stride), size),
                       stride_0=stride)
                  for shift, stride, size in zip(self.seq_shifts, self.seq_strides, self.frame_sizes)]
        shapes[-1]["slice"] = slice(shapes[-1]["slice"].start, item + self.top_size + self.seq_len - 1)
        return shapes, slice(item + self.top_size, item + self.top_size + self.seq_len)


class FramesDB(DBDataset):
    qx = None

    @staticmethod
    def extract(path, sr=22050, mu=255):
        signal = A.FileTo.mu_law_compress(path, sr=sr, mu=mu)
        return dict(qx=(dict(sr=sr, mu=mu), signal.reshape(-1, 1), None))

    def prepare_dataset(self, model, datamodule):
        batch_seq_len, frame_sizes = model.batch_info()
        self.frame_prm = FramedSequences(batch_seq_len, frame_sizes)
        # round the size of the dataset to a multiple of the chunk size :
        batch_sampler = ThreadedSampler(8 * 16000 * (len(self.qx) // (8 * 16000)),
                                        model.hparams.batch_size,
                                        8 * 16000,
                                        model.hparams.batch_seq_len)
        self.length = len(batch_sampler)
        datamodule.loader_kwargs.update(dict(batch_sampler=batch_sampler))
        for k in ["batch_size", "shuffle", "drop_last"]:
            if k in datamodule.loader_kwargs:
                datamodule.loader_kwargs.pop(k)
        datamodule.loader_kwargs["sampler"] = None
        print("Qx", len(self.qx), "rounded", 8 * 16000 * (len(self.qx) // (8 * 16000)))
        print("sampler", len(batch_sampler))
        print("self", len(self))

    def __getitem__(self, item):
        if type(self.qx) is not torch.Tensor:
            itemsize = self.qx.dtype.itemsize
            as_strided = lambda prm: np_as_strided(self.qx[prm["slice"]],
                                                   shape=prm["shape"],
                                                   strides=(itemsize * prm["stride_0"], itemsize))
        else:
            as_strided = lambda prm: torch.as_strided(self.qx[prm["slice"]],
                                                      size=prm["shape"],
                                                      stride=(prm["stride_0"], 1))

        shapes, tgrt_slice = self.frame_prm(item)
        with torch.no_grad():
            inputs = [self.qx[prm["slice"]].reshape(-1, prm["shape"][1])
                      for prm in shapes[:-1]] + [as_strided(shapes[-1])]
        target = self.qx[tgrt_slice]

        return tuple(inputs), target

    def __len__(self):
        return len(self.qx)


def wavenet_loss_fn(output, target):
    criterion = nn.CrossEntropyLoss(reduction="none")
    # diff = (output.max(dim=-1).indices.view(-1) - target.view(-1)).abs()
    return (criterion(output.view(-1, output.size(-1)), target.view(-1)) * (1)).mean()


class SampleRNN(SequenceModel,
                DataSubModule,
                SuperAdam,
                SampleRNNNetwork,
                LightningModule):
    @property
    def loss_fn(self):
        return wavenet_loss_fn

    db_class = FramesDB

    def __init__(self,
                 frame_sizes=(4, 4),
                 net_dim=128,
                 emb_dim=128,
                 mlp_dim=256,
                 n_rnn=1,
                 q_levels=256,
                 max_lr=1e-3,
                 betas=(.9, .9),
                 div_factor=3.,
                 final_div_factor=1.,
                 pct_start=.25,
                 cycle_momentum=False,
                 db=None,
                 files=None,
                 batch_size=64,
                 batch_seq_len=512,
                 in_mem_data=True,
                 splits=[],
                 **loaders_kwargs
                 ):
        super(LightningModule, self).__init__()
        SequenceModel.__init__(self)
        DataSubModule.__init__(self, db, files, in_mem_data, splits, batch_size=batch_size, **loaders_kwargs)
        SuperAdam.__init__(self, max_lr, betas, div_factor, final_div_factor, pct_start, cycle_momentum)
        SampleRNNNetwork.__init__(self, frame_sizes, net_dim, n_rnn, q_levels, emb_dim, mlp_dim)
        self.save_hyperparameters()

    def batch_info(self, *args, **kwargs):
        return self.hparams.batch_seq_len, self.frame_sizes

    def setup(self, stage: str):
        SuperAdam.setup(self, stage)

    def backward(self, loss, optimizer, optimizer_idx: int, *args, **kwargs) -> None:
        loss.backward(retain_graph=True)

    def on_train_batch_start(self, batch, batch_idx, dataloader_idx):
        if (batch_idx * self.hparams.batch_seq_len) % (8 * 16000) == 0:
            self.reset_h0()
