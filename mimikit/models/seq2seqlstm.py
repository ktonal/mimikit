import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import dataclasses as dtc

from ..audios import Spectrogram
from ..data import Feature, AsSlice, Input, Target
from .parts import SuperAdam, SequenceModel, IData
from ..networks import Seq2SeqLSTM
from .parts.loss_functions import mean_L1_prop

__all__ = [
    'Seq2SeqLSTMModel'
]


@dtc.dataclass
class Seq2SeqData(IData):
    feature: Feature = None
    batch_size: int = 16

    @classmethod
    def schema(cls, sr=22050, emphasis=0., n_fft=2048, hop_length=512):
        schema = {"fft": Spectrogram(sr=sr, emphasis=emphasis,
                                     n_fft=n_fft, hop_length=hop_length,
                                     magspec=True)}
        return schema

    @classmethod
    def dependant_hp(cls, db):
        return dict(
            feature=Spectrogram(**db.fft.attrs), input_dim=db.fft.shape[-1]
        )

    def batch_signature(self, stage='fit'):
        inpt = Input('fft', AsSlice(shift=0, length=self.shift))
        trgt = Target('fft', AsSlice(shift=self.shift,
                                     length=self.shift))
        if stage in ('full', 'fit', 'train', 'val'):
            return inpt, trgt
        # test, predict, generate...
        return inpt

    def loader_kwargs(self, stage, datamodule):
        return dict(
            batch_size=self.batch_size,
            drop_last=False,
            shuffle=True
        )


class Seq2SeqLSTMModel(
    Seq2SeqData,
    SequenceModel,
    SuperAdam,
    Seq2SeqLSTM,
):

    @staticmethod
    def loss_fn(output, target):
        return {"loss": mean_L1_prop(output, target)}

    def encode_inputs(self, inputs: torch.Tensor):
        return self.feature.encode(inputs)

    def decode_outputs(self, outputs: torch.Tensor):
        return self.feature.decode(outputs)
