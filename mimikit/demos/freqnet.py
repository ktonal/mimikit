import dataclasses as dtc
from typing import List
import h5mapper as h5m
import os

from ..features import Spectrogram
from ..models.wavenets import WaveNetBlockHP, WaveNetFFT
from ..train import train
from ..loops.train_loops import TrainARMConfig

__all__ = [
    "FreqNetData",
    "TrainFreqnetConfig",
    "main"
]


@dtc.dataclass
class FreqNetData:
    sources_dir: str = ''
    sources: List[str] = dtc.field(default_factory=list)
    db_path: str = "train.h5"
    sr: int = 22050
    n_fft: int = 2048
    hop_length: int = 512
    coordinate: str = 'mag'
    center: bool = False

    def walk_sources_dir(self):
        if len(self.sources_dir) > 0:
            self.sources.extend(
                list(h5m.FileWalker(h5m.Sound.__re__, self.sources_dir))
            )

    def validate(self):
        if len(self.sources) == 0:
            raise ValueError("Empty dataset. No audio file were found")


# noinspection PyCallByClass
@dtc.dataclass
class TrainFreqnetConfig:

    data: FreqNetData = FreqNetData()

    network: WaveNetFFT.HP = WaveNetFFT.HP(
        core=WaveNetBlockHP(
            # number of layers (per block)
            blocks=(4,),
            # dimension of the layers
            dims_dilated=(1024,),
            groups=2,
            pad_side=0,
        ),
        input_heads=1,
        output_heads=1,
        scaled_activation=False,
    )

    training: TrainARMConfig = TrainARMConfig(
        root_dir="./",
        # BATCH
        batch_size=4,
        batch_length=64,
        downsampling=512,
        shift_error=0,
        # OPTIM
        max_epochs=100,
        limit_train_batches=None,
        max_lr=1e-3,
        betas=(0.9, 0.9),
        div_factor=3.,
        final_div_factor=1.,
        pct_start=0.,
        cycle_momentum=False,
        # MONITORING / OUTPUTS
        CHECKPOINT_TRAINING=True,
        MONITOR_TRAINING=True,
        OUTPUT_TRAINING="",
        every_n_epochs=10,
        n_examples=4,
        prompt_length=64,
        n_steps=int(12 * (22050 // 512)),
    )


def main(cfg: TrainFreqnetConfig = TrainFreqnetConfig()):
    """### Load Data"""

    if os.path.exists(cfg.data.db_path):
        os.remove(cfg.data.db_path)

    class SoundBank(h5m.TypedFile):
        snd = h5m.Sound(sr=cfg.data.sr, mono=True, normalize=True)

    SoundBank.create(cfg.data.db_path, cfg.data.sources)
    soundbank = SoundBank(cfg.data.db_path, mode='r', keep_open=True)

    """### Configure and run training"""

    # INPUT / TARGET

    feature = Spectrogram(
        sr=cfg.data.sr,
        n_fft=cfg.data.n_fft,
        hop_length=cfg.data.hop_length,
        coordinate=cfg.data.coordinate,
        center=cfg.data.center
    )

    # NETWORK

    net = WaveNetFFT(
        feature=feature,
        hp=cfg.network
    )
    net.use_fast_generate = False

    # OPTIMIZATION LOOP

    train(
        cfg.training,
        soundbank,
        net,
        input_feature=feature,
        target_feature=feature,
    )

    """----------------------------"""
