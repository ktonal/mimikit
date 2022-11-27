import dataclasses as dtc
from typing import Optional, Tuple
import torch.nn as nn

from ..networks import WNBlock
from ..features import MuLawSignal, Spectrogram
from .io_modules import mag_spec_io, qx_io, pol_spec_io, wn_qx_io

__all__ = [
    "WaveNetQx",
    "WaveNetFFT",
    "WaveNetBlockHP"
]


@dtc.dataclass
class WaveNetBlockHP:
    input_dim: Optional[int] = None
    kernel_sizes: Tuple[int] = (2,)
    blocks: Tuple[int] = ()
    dims_dilated: Tuple[int] = (128,)
    dims_1x1: Tuple[int] = ()
    residuals_dim: Optional[int] = None
    apply_residuals: bool = False
    skips_dim: Optional[int] = None
    groups: int = 1
    act_f: str = "tanh"
    act_g: Optional[str] = "sigmoid"
    pad_side: int = 1
    stride: int = 1
    bias: bool = True


class WaveNetQx(WNBlock):
    feature = MuLawSignal(sr=16000, q_levels=256, normalize=True)

    def __init__(self, feature=None, mlp_dim=128, embedding_dim=64, **block_hp):
        block_hp["input_dim"] = embedding_dim
        super(WaveNetQx, self).__init__(**block_hp)
        if feature is not None:
            self.hp.feature = self.feature = feature
        self.hp.mlp_dim = mlp_dim
        inpt_mod, outpt_mod = wn_qx_io(
            feature.q_levels,
            embedding_dim,
            self.hp.skips_dim if self.hp.skips_dim else self.hp.dims_dilated[0],
            mlp_dim
        )
        self.with_io([inpt_mod], [outpt_mod])


class WaveNetFFT(WNBlock):
    feature = Spectrogram(sr=22050, n_fft=2048, hop_length=512, coordinate='mag')

    @dtc.dataclass
    class HP:
        core: WaveNetBlockHP
        input_heads: int = 2
        output_heads: int = 4
        scaled_activation: bool = True
        phs: str = "a"
        with_sampler: bool = False

    # TODO: FIND SOLUTION FOR LOADING CHECKPOINTS!!......
    def __init__(self,
                 feature: Spectrogram,
                 hp: HP,
                 ):
        core_hp = dtc.asdict(hp.core)
        acts = dict(tanh=nn.Tanh(), sigmoid=nn.Sigmoid)
        core_hp["act_f"] = acts[core_hp["act_f"]]
        core_hp["act_g"] = acts.get(core_hp["act_g"], None)
        super(WaveNetFFT, self).__init__(**core_hp)
        if feature is not None:
            self.hp.feature = self.feature = feature
        self.hp.input_heads = hp.input_heads
        self.hp.output_heads = hp.output_heads
        self.hp.scaled_activation = hp.scaled_activation
        self.hp.phs = hp.phs
        self.hp.with_sampler = hp.with_sampler
        fft_dim = self.feature.n_fft // 2 + 1
        net_dim = self.hp.dims_dilated[0]
        if feature.coordinate == "mag":
            inpt_mod, outpt_mod = mag_spec_io(fft_dim, net_dim, hp.input_heads, hp.output_heads, hp.scaled_activation, hp.with_sampler)
        elif feature.coordinate == "pol":
            inpt_mod, outpt_mod = pol_spec_io(fft_dim, net_dim, hp.input_heads, hp.output_heads, hp.scaled_activation, hp.phs, hp.with_sampler)
        else:
            raise ValueError(f"WaveNetFFT doesn't support coordinate of type {feature.coordinate}")
        self.with_io([inpt_mod], [outpt_mod])
