import math

import torch
from torch.utils.data import Sampler, RandomSampler, BatchSampler


__all__ = [
    'TBPTTSampler'
]


class TBPTTSampler(Sampler):
    """
    yields batches of indices for performing Truncated Back Propagation Through Time
    """

    def __init__(self,
                 n_samples,
                 batch_size=64,  # nbr of "tracks" per batch
                 chunk_length=8 * 16000,  # total length of a track
                 seq_len=512  # nbr of samples per backward pass
                 ):
        super().__init__(None)
        self.n_samples = n_samples
        self.chunk_length = chunk_length
        self.seq_len = seq_len
        self.n_chunks = self.n_samples // self.chunk_length
        self.remainder = self.n_samples % self.chunk_length
        self.n_per_chunk = self.chunk_length // self.seq_len
        self.batch_size = min(batch_size, self.n_chunks)

    def __iter__(self):
        smp = RandomSampler(torch.arange(self.n_chunks))
        for top in BatchSampler(smp, self.batch_size, False):  # don't drop last!
            offsets = torch.randint(0, self.remainder, (self.batch_size,))
            top = tuple(o + (t * self.chunk_length) for t, o in zip(top, offsets))
            for start in range(self.n_per_chunk):
                # start indices of the batch
                yield tuple(t + (start * self.seq_len) for t in top)

    def __len__(self):
        return int(max(1, math.floor(self.n_chunks / self.batch_size)) * self.n_per_chunk)