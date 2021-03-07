import numpy as np
from numpy.lib.stride_tricks import as_strided as np_as_strided
from pytorch_lightning import LightningModule
import torch
import torch.nn as nn

from ..kit import DBDataset
from ..audios import transforms as A
from ..kit.ds_utils import ShiftedSequences
from ..kit import SuperAdam, SequenceModel, DataSubModule
from ..kit.networks.sample_rnn import SampleRNNNetwork

from torch.utils.data import Sampler, RandomSampler, BatchSampler
import math


class TBPTTSampler(Sampler):
    """
    yields batches of indices for performing Truncated Back Propagation Through Time
    """
    def __init__(self,
                 n_samples,
                 batch_size=64,  # nbr of "tracks" per batch
                 chunk_length=8*16000,  # total length of a track
                 seq_len=512  # nbr of samples per backward pass
                 ):
        super().__init__(None)
        self.n_samples = n_samples
        self.batch_size = batch_size
        self.chunk_length = chunk_length
        self.seq_len = seq_len
        self.n_chunks = self.n_samples // self.chunk_length
        self.n_per_chunk = self.chunk_length // self.seq_len

    def __iter__(self):
        smp = RandomSampler(torch.arange(self.n_chunks))
        for top in BatchSampler(smp, self.batch_size, True):  # drop last!
            for start in range(self.n_per_chunk):
                # start indices of the batch
                yield tuple((t * self.chunk_length) + (start * self.seq_len) for t in top)

    def __len__(self):
        return int(max(1, math.floor(self.n_chunks / self.batch_size)) * self.n_per_chunk)


class FramesDB(DBDataset):
    qx = None

    @staticmethod
    def extract(path, sr=16000, mu=255):
        signal = A.FileTo.mu_law_compress(path, sr=sr, mu=mu)
        return dict(qx=(dict(sr=sr, mu=mu), signal.reshape(-1, 1), None))

    def prepare_dataset(self, model, datamodule):
        batch_size, chunk_len, batch_seq_len, frame_sizes = model.batch_info()
        shifts = [frame_sizes[0] - size for size in frame_sizes + (0,)]  # (0,) for the target
        lengths = [batch_seq_len for _ in frame_sizes[:-1]]
        lengths += [frame_sizes[0] + batch_seq_len]
        lengths += [batch_seq_len]

        self.slicer = ShiftedSequences(len(self.qx), list(zip(shifts, lengths)))
        self.frame_sizes = frame_sizes
        self.seq_len = batch_seq_len

        # round the size of the dataset to a multiple of the chunk size :
        batch_sampler = TBPTTSampler(chunk_len * (len(self.qx) // chunk_len),
                                     batch_size,
                                     chunk_len,
                                     batch_seq_len)
        datamodule.loader_kwargs.update(dict(batch_sampler=batch_sampler))
        for k in ["batch_size", "shuffle", "drop_last"]:
            if k in datamodule.loader_kwargs:
                datamodule.loader_kwargs.pop(k)
        datamodule.loader_kwargs["sampler"] = None

    def __getitem__(self, item):
        if type(self.qx) is not torch.Tensor:
            itemsize = self.qx.dtype.itemsize
            as_strided = lambda slc, fs: np_as_strided(self.qx[slc],
                                                       shape=(self.seq_len, fs),
                                                       strides=(itemsize, itemsize))
        else:
            as_strided = lambda slc, fs: torch.as_strided(self.qx[slc],
                                                          size=(self.seq_len, fs),
                                                          stride=(1, 1))

        slices = self.slicer(item)
        tiers_slc, bottom_slc, target_slc = slices[:-2], slices[-2], slices[-1]
        inputs = [self.qx[slc].reshape(-1, fs) for slc, fs in zip(tiers_slc, self.frame_sizes[:-1])]
        # ugly but necessary if self.qx became a tensor...
        with torch.no_grad():
            inputs += [as_strided(bottom_slc, self.frame_sizes[-1])]

        target = self.qx[target_slc]

        return tuple(inputs), target

    def __len__(self):
        return len(self.qx)


class SampleRNN(SequenceModel,
                DataSubModule,
                SuperAdam,
                SampleRNNNetwork,
                LightningModule):

    @staticmethod
    def loss_fn(output, target):
        criterion = nn.CrossEntropyLoss(reduction="mean")
        return criterion(output.view(-1, output.size(-1)), target.view(-1))

    db_class = FramesDB

    def __init__(self,
                 frame_sizes=(4, 4),
                 net_dim=128,
                 emb_dim=128,
                 mlp_dim=256,
                 n_rnn=1,
                 q_levels=256,  # == mu + 1
                 max_lr=1e-3,
                 betas=(.9, .9),
                 div_factor=3.,
                 final_div_factor=1.,
                 pct_start=.25,
                 cycle_momentum=True,
                 db=None,
                 files=None,
                 batch_size=64,
                 batch_seq_len=512,
                 chunk_len=8*16000,
                 in_mem_data=True,
                 splits=None,  # tbptt should implement the splits...
                 **loaders_kwargs
                 ):
        super(LightningModule, self).__init__()
        SequenceModel.__init__(self)
        DataSubModule.__init__(self, db, files, in_mem_data, splits, batch_size=batch_size, **loaders_kwargs)
        SuperAdam.__init__(self, max_lr, betas, div_factor, final_div_factor, pct_start, cycle_momentum)
        SampleRNNNetwork.__init__(self, frame_sizes, net_dim, n_rnn, q_levels, emb_dim, mlp_dim)
        self.save_hyperparameters()
        self.stored_grad_norms = []

    def batch_info(self, *args, **kwargs):
        return tuple(self.hparams[key] for key in ["batch_size", "chunk_len", "batch_seq_len", "frame_sizes"])

    def setup(self, stage: str):
        SuperAdam.setup(self, stage)

    def on_train_batch_start(self, batch, batch_idx, dataloader_idx):
        if (batch_idx * self.hparams.batch_seq_len) % self.hparams.chunk_len == 0:
            self.reset_h0()

    def on_after_backward(self):
        if self.global_step % 5 == 0:
            out, norms = {}, []
            prefix = f'grad_1_norm_'
            for name, p in self.named_parameters():
                if p.grad is None:
                    continue

                # `np.linalg.norm` implementation likely uses fp64 intermediates
                flat = p.grad.data.cpu().numpy().ravel()
                norm = np.linalg.norm(flat, 1)
                norms.append(norm)

                out[name] = round(norm, 4)

            # handle total norm
            norm = np.linalg.norm(norms, 1)
            out[prefix + 'total'] = round(norm, 4)
            self.stored_grad_norms.append(out)


