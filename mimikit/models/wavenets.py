
from ..networks import WNBlock
from ..features import MuLawSignal, Spectrogram
from .io_modules import mag_spec_io, qx_io, pol_spec_io


__all__ = [
    "WaveNetQx",
    "WaveNetFFT"
]


class WaveNetQx(WNBlock):
    feature = MuLawSignal(sr=16000, q_levels=256, normalize=True)

    def __init__(self, feature=None, mlp_dim=128, **block_hp):
        super(WaveNetQx, self).__init__(**block_hp)
        if feature is not None:
            self.hp.feature = self.feature = feature
        self.hp.mlp_dim = mlp_dim
        inpt_mod, outpt_mod = qx_io(feature.q_levels, self.hp.dims_dilated[0], mlp_dim)
        self.with_io([inpt_mod], [outpt_mod])


class WaveNetFFT(WNBlock):
    feature = Spectrogram(sr=22050, n_fft=2048, hop_length=512, coordinate='mag')

    def __init__(self,
                 feature: Spectrogram,
                 input_heads=2,
                 output_heads=4,
                 scaled_activation=True,
                 phs="a",
                 **block_hp
                 ):
        super(WaveNetFFT, self).__init__(**block_hp)
        if feature is not None:
            self.hp.feature = self.feature = feature
        self.hp.input_heads = input_heads
        self.hp.output_heads = output_heads
        self.hp.scaled_activation = scaled_activation
        self.hp.phs = phs
        fft_dim = self.feature.n_fft // 2 + 1
        net_dim = self.hp.dims_dilated[0]
        if feature.coordinate == "mag":
            inpt_mod, outpt_mod = mag_spec_io(fft_dim, net_dim, input_heads, output_heads, scaled_activation)
        elif feature.coordinate == "pol":
            inpt_mod, outpt_mod = pol_spec_io(fft_dim, net_dim, input_heads, output_heads, scaled_activation, phs)
        else:
            raise ValueError(f"WaveNetFFT doesn't support coordinate of type {feature.coordinate}")
        self.with_io([inpt_mod], [outpt_mod])
