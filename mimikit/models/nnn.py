import numpy as np
import torch
import torch.nn as nn
from librosa.sequence import dtw
from sklearn.metrics import pairwise_distances


__all__ = [
    "optimal_path",
    "NearestNextNeighbor"
]


def optimal_path(x, y):
    return dtw(C=pairwise_distances(abs(x), abs(y), metric='cosine'), subseq=True)[1][::-1]


class NearestNextNeighbor(nn.Module):

    device = property(lambda self: next(self.parameters()).device)

    def __init__(self, feature, snd):
        super(NearestNextNeighbor, self).__init__()
        self.feature = feature
        self.snd = feature.transform(snd[:])
        self._t = -100
        self._starts = None
        self._param = nn.Parameter(torch.ones(1))
        self.shift = self.rf = feature.n_fft
        self.output_length = lambda x: 1

    def predict_start_frame(self, X):
        path = optimal_path(X, self.snd)
        return path[-1, -1]+1

    def generate_step(self, t, inputs, ctx):
        """predict start frame if inputs is new else return next frame"""
        if t != self._t + 1:
            self._starts = [self.predict_start_frame(x.detach().cpu().numpy())
                            for x in inputs[0]]
            self._t = t - 1
        output = np.stack([self.snd[i:i+1] for i in self._starts])
        self._starts = [x+1 for x in self._starts]
        self._t += 1
        return torch.from_numpy(output).to(inputs[0].device)
