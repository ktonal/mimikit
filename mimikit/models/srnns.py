from ..networks import SampleRNN as SampleRNNNet

__all__ = [
    "SampleRNN"
]


class SampleRNN(SampleRNNNet):

    def __init__(self, feature, chunk_length=16000 * 8, **net_hp):
        super(SampleRNN, self).__init__(**net_hp)
        self.feature = self.hp.feature = feature
        self.hp.chunk_length = chunk_length
