import h5mapper as h5m

from . import audio_fmodules as T
from .ifeature import DiscreteFeature, RealFeature, SequenceSpec

__all__ = [
    'AudioSignal',
    'MuLawSignal',
    'ALawSignal',
    'Spectrogram',
]


# @dtc.dataclass(unsafe_hash=True)
class AudioSignal(RealFeature):
    sr: int = 22050
    emphasis: float = 0.

    h5m_type = property(lambda self: h5m.Sound(sr=self.sr, normalize=True))
    out_dim = 1
    support = (-1., 1.)

    # noinspection PyMethodOverriding
    def __post_init__(self):
        # we need this for instantiating fresh SequenceSpecs
        super().__post_init__(self.sr)

    def t(self, inputs):
        if self.emphasis:
            inputs = T.Emphasis(self.emphasis)(inputs)
        return inputs

    def inv(self, inputs):
        if self.emphasis:
            inputs = T.Deemphasis(self.emphasis)(inputs)
        return inputs


# @dtc.dataclass(unsafe_hash=True)
class MuLawSignal(DiscreteFeature):
    sr: int = 16000
    emphasis: float = 0.
    q_levels: int = 256
    compression_factor: float = 1.

    class_size = property(lambda self: self.q_levels)
    vector_dim = 1

    h5m_type = property(lambda self: h5m.Sound(sr=self.sr, normalize=True))

    # noinspection PyMethodOverriding
    def __post_init__(self):
        super().__post_init__(self.sr)
        self.base = AudioSignal(self.sr, self.emphasis)
        self.t_ = T.MuLawCompress(q_levels=self.q_levels)
        self.inv_ = T.MuLawExpand(self.q_levels)

    def t(self, inputs):
        return self.t_(self.base.t(inputs))

    def inv(self, inputs):
        return self.inv_(self.base.inv(inputs))


# @dtc.dataclass(unsafe_hash=True)
class ALawSignal(DiscreteFeature):
    sr: int = 16000
    emphasis: float = 0.
    A: float = 87.7
    q_levels: int = 256

    class_size = property(lambda self: self.q_levels)
    vector_dim = 1

    h5m_type = property(lambda self: h5m.Sound(sr=self.sr, normalize=True))

    # noinspection PyMethodOverriding
    def __post_init__(self):
        super().__post_init__(self.sr)
        self.base = AudioSignal(self.sr, self.emphasis)
        self.t_ = T.ALawCompress(A=self.A, q_levels=self.q_levels)
        self.inv_ = T.ALawExpand(A=self.A, q_levels=self.q_levels)

    def t(self, inputs):
        return self.t_(self.base.t(inputs))

    def inv(self, inputs):
        return self.inv_(self.base.inv(inputs))


# @dtc.dataclass(unsafe_hash=True)
class Spectrogram(RealFeature):
    sr: int = 22050
    emphasis: float = 0.
    n_fft: int = 2048
    hop_length: int = 512
    coordinate: str = 'pol'
    center: bool = True

    out_dim = property(lambda self: 1 + self.n_fft // 2)
    support = (0., float("inf"))

    h5m_type = property(lambda self: h5m.Sound(sr=self.sr, normalize=True))

    # noinspection PyMethodOverriding
    def __post_init__(self):
        self.seq_spec = SequenceSpec(self.sr, shift=-int(self.center)*self.n_fft//2,
                                     frame_size=self.n_fft,
                                     hop_length=self.hop_length)
        self.base = AudioSignal(self.sr, self.emphasis)
        if self.coordinate == 'mag':
            self.t_ = T.MagSpec(self.n_fft, self.hop_length, center=self.center)
            self.inv_ = T.GLA(self.n_fft, self.hop_length, center=self.center, n_iter=32)
        else:
            self.t_ = T.STFT(self.n_fft, self.hop_length, self.coordinate, center=self.center)
            self.inv_ = T.ISTFT(self.n_fft, self.hop_length, self.coordinate, center=self.center)

    def t(self, inputs):
        return self.t_(self.base.t(inputs))

    def inv(self, inputs):
        return self.inv_(self.base.inv(inputs))
