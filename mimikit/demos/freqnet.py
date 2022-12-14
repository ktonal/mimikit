import dataclasses as dtc
from typing import List
import h5mapper as h5m
import os

from ..features import Spectrogram
from ..models.wavenets import WaveNetBlockHP, WaveNetFFT, WaveNetFFTHP
from ..train import train
from ..loops.train_loops import TrainARMConfig


__all__ = [
    "FreqNetData",
    "TrainFreqnetConfig",
    "main"
]


@dtc.dataclass
class FreqNetData:
    sources: List[str] = dtc.field(default_factory=list)
    db_path: str = "train.h5"

    def validate(self):
        if len(self.sources) == 0:
            raise ValueError("Empty dataset. No audio file were found")


@dtc.dataclass
class FreqNetConfig:
    inputs = [Spectrogram(
        sr=22050, n_fft=2048, hop_length=512, coordinate='mag', center=False
    )]
    outputs = [Spectrogram(
        sr=22050, n_fft=2048, hop_length=512, coordinate='mag', center=False
    )]
    network = WaveNetFFTHP(
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


@dtc.dataclass
class TrainFreqnetConfig:
    dataset: FreqNetData = FreqNetData()

    model = FreqNetConfig()

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

    if os.path.exists(cfg.dataset.db_path):
        os.remove(cfg.dataset.db_path)

    fft = cfg.model.inputs[0]

    class SoundBank(h5m.TypedFile):
        snd = h5m.Sound(sr=fft.sr, mono=True, normalize=True)

    SoundBank.create(cfg.dataset.db_path, cfg.dataset.sources)
    soundbank = SoundBank(cfg.dataset.db_path, mode='r', keep_open=True)

    """### Configure and run training"""

    # INPUT / TARGET

    feature = Spectrogram(
        sr=fft.sr,
        n_fft=fft.n_fft,
        hop_length=fft.hop_length,
        coordinate=fft.coordinate,
        center=fft.center
    )

    # NETWORK

    net = WaveNetFFT(
        feature=feature,
        hp=cfg.model.network
    )
    net.use_fast_generate = False

    # OPTIMIZATION LOOP

    train(
        cfg.training,
        cfg.model,
        soundbank,
        net,
        input_feature=feature,
        target_feature=feature,
    )

    """----------------------------"""
