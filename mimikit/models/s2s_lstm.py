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
from .model import model

__all__ = [
    'Seq2SeqLSTMModel'
]


@dtc.dataclass(init=True, repr=False, eq=False, frozen=False, unsafe_hash=True)
class Seq2SeqData(IData):
    feature: Feature = None
    batch_size: int = 16
    shift: int = 8

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


@model
class Seq2SeqLSTMModel(
    Seq2SeqData,
    SuperAdam,
    SequenceModel,
    Seq2SeqLSTM,
):

    @staticmethod
    def loss_fn(output, target):
        return {"loss": mean_L1_prop(output, target)}

    def encode_inputs(self, inputs: torch.Tensor):
        return self.feature.encode(inputs)

    def decode_outputs(self, outputs: torch.Tensor):
        return self.feature.decode(outputs)


def demo():
    """### import and arguments"""
    import mimikit as mmk

    # DATA

    # list of files or directories to use as data ("./" is the cwd of the notebook)
    sources = ['./data']
    # audio sample rate
    sr = 22050
    # the size of the stft
    n_fft = 2048
    # hop_length of the
    hop_length = n_fft // 4

    # NETWORK

    # this network takes `shift` fft frames as input and outputs `shift` future frames
    shift = 8
    # the net contains at least 3 LSTM modules (1 in the Encoder, 2 in the Decoder)
    # you can add modules to the Encoder by increasing the next argument
    n_lstm = 1
    # all LSTM modules have internally the same number of layers :
    num_layers = 1
    # the dimensionality of the model
    model_dim = 1024

    # OPTIMIZATION

    # how many epochs should we train for
    max_epochs = 50
    # how many examples are used pro training steps
    batch_size = 16
    # the learning rate
    max_lr = 1e-3
    # betas control how fast the network changes its 'learning course'.
    # generally, betas should be close but smaller than 1. and be balanced with the batch_size :
    # the smaller the batch, the higher the betas 'could be'.
    betas = (0.9, 0.93)

    # MONITORING

    # how often should the network generate during training
    every_n_epochs = 2
    # how many examples from random prompts should be generated
    n_examples = 3
    # how many steps (1 step = `shift` fft frames!) should be generated
    n_steps = 1000 // shift

    print("arguments are ok!")

    """### create the data"""
    schema = mmk.Seq2SeqLSTMModel.schema(sr, n_fft=n_fft, hop_length=hop_length)

    db_path = 's2s-demo.h5'
    print("collecting data...")
    db = mmk.Database.create(db_path, sources, schema)
    if not len(db.fft.files):
        raise ValueError("Empty db. No audio files were found")
    print("successfully created the db.")

    """### create network and train"""
    net = mmk.Seq2SeqLSTMModel(
        **mmk.Seq2SeqLSTMModel.dependant_hp(db),
        shift=shift,
        n_lstm=n_lstm,
        num_layers=num_layers,
        model_dim=model_dim,
        batch_size=batch_size,
        max_lr=max_lr,
        div_factor=5,
        betas=betas,

    )
    print(net.hparams)

    dm = mmk.DataModule(net, db, splits=tuple())

    cb = mmk.GenerateCallback(every_n_epochs,
                              indices=[None] * n_examples,
                              n_steps=n_steps,
                              play_audios=True,
                              plot_audios=True)

    trainer = mmk.get_trainer(root_dir=None,
                              max_epochs=max_epochs,
                              callbacks=[cb],
                              checkpoint_callback=False)
    print("here we go!")
    trainer.fit(net, datamodule=dm)

    """----------------------------"""
