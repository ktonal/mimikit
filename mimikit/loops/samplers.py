import math

import torch
from torch.utils.data import Sampler, RandomSampler, BatchSampler

__all__ = [
    'TBPTTSampler',
    'IndicesSampler'
]


class TBPTTSampler(Sampler):
    """
    yields batches of indices for performing Truncated Back Propagation Through Time
    """

    def __init__(self,
                 n_samples,
                 batch_size=64,  # nbr of "tracks" per batch
                 chunk_length=8 * 16000,  # total length of a track
                 seq_len=512,  # nbr of samples per backward pass
                 oversampling=1,
                 ):
        super().__init__(None)
        self.n_samples = n_samples
        self.chunk_length = min(chunk_length, n_samples)
        self.seq_len = seq_len
        self.n_chunks = max(1, self.n_samples // self.chunk_length - int(oversampling > 1))
        self.remainder = self.n_samples % self.chunk_length
        self.n_per_chunk = self.chunk_length // self.seq_len
        self.batch_size = batch_size
        self.oversampling = oversampling

    def __iter__(self):
        indices = torch.arange(self.n_chunks * self.oversampling)
        smp = RandomSampler(indices)
        for top in BatchSampler(smp, self.batch_size, False):
            # for _ in range(self.oversampling):
            offsets = torch.randint(0, self.remainder, (self.batch_size,))
            top_idx = tuple(o + ((t % self.n_chunks) * self.chunk_length)
                            for t, o in zip(top, offsets))
            for start in range(self.n_per_chunk):
                # start indices of the batch
                yield tuple(t + (start * self.seq_len) for t in top_idx)

    def __len__(self):
        return (self.oversampling * self.n_chunks // self.batch_size) * self.n_per_chunk


class IndicesSampler(Sampler):
    def __init__(self,
                 N=0,
                 indices=(),
                 min_i=0,
                 max_i=None,
                 redraw=True,
                 ):
        super().__init__(None)
        self.N = N
        self._indices = indices
        self.min_i = min_i
        self.max_i = max_i
        self.redraw = redraw
        self.indices = self.draw_indices(N, indices)

    def __iter__(self):
        for i in self.indices:
            yield i
        if self.redraw:
            self.indices = self.draw_indices(self.N, self._indices)

    def draw_indices(self, N, indices):
        n_idx = len(indices)
        if N == 0 and n_idx == 0:
            raise ValueError("`indices` can not be empty if `N` == 0")
        elif N > 0 and n_idx == 0:
            # only random
            return torch.randint(self.min_i, self.max_i, (N,))
        elif N == 0 and n_idx > 0:
            # only static
            return indices
        elif N > 0 and 0 < n_idx < N:
            # complete statics with randoms
            return [*indices, *torch.randint(self.min_i, self.max_i, (N,))]
        else:
            # N <= n_idx -> only statics
            return indices
