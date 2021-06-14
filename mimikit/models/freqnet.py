import torch
import dataclasses as dtc

from ..abstract.features import Feature, SegmentLabels, FilesLabels
from ..audios import Spectrogram
from ..data import Input, AsSlice, Target
from .parts import SuperAdam, SequenceModel, mean_L1_prop, IData
from .model import model
from ..networks import FreqNetNetwork, WNNetwork

__all__ = [
    'FreqNetData',
    'FreqNet',
    'demo'
]


@dtc.dataclass(init=True, repr=False, eq=False, frozen=False, unsafe_hash=True)
class FreqNetData(IData):

    feature: Feature = None
    batch_size: int = 64
    batch_seq_length: int = 64

    @classmethod
    def schema(cls, sr=22050, emphasis=0., n_fft=2048, hop_length=512,
               segment_labels=False, files_labels=False):

        schema = {"fft": Spectrogram(sr=sr, emphasis=emphasis,
                                     n_fft=n_fft, hop_length=hop_length,
                                     magspec=True)}
        if segment_labels:
            schema.update({
                'loc': SegmentLabels(input_key='fft')
            })
        if files_labels:
            schema.update({
                'glob': FilesLabels(input_key='fft')
            })
        return schema

    @classmethod
    def dependant_hp(cls, db):
        hp = {}
        if 'loc' in db.features:
            hp.update(dict(n_cin_classes=len(db.loc.regions)))
        if 'glob' in db.features:
            hp.update(dict(n_gin_classes=len(db.glob.files)))
        return dict(
            feature=Spectrogram(**db.fft.attrs), input_dim=db.fft.shape[-1], **hp
        )

    def batch_signature(self, stage='fit'):
        inpt = [Input('fft', AsSlice(shift=0, length=self.batch_seq_length))]
        trgt = Target('fft', AsSlice(shift=self.shift,
                                     length=self.output_shape((-1, self.batch_seq_length, -1))[1]))
        # where are we conditioned?
        loc, glob = self.n_cin_classes is not None, self.n_gin_classes is not None
        if loc:
            inpt += [Input('loc', AsSlice(shift=0, length=self.batch_seq_length))]
        if glob:
            inpt += [Input('glob', AsSlice(shift=0, length=self.batch_seq_length))]
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
class FreqNet(
    # data configuration :
    FreqNetData,
    # optimizer :
    SuperAdam,
    # training step, generate routine & interface :
    SequenceModel,
    # overrides wavenet's inputs/outputs modules :
    FreqNetNetwork,
    # we inherit the networks __init__ and all generate_* from wavenet
    WNNetwork
):

    @staticmethod
    def loss_fn(output, target):
        return {"loss": mean_L1_prop(output, target)}

    def setup(self, stage: str):
        SequenceModel.setup(self, stage)
        SuperAdam.setup(self, stage)

    def encode_inputs(self, inputs: torch.Tensor):
        return self.feature.encode(inputs)

    def decode_outputs(self, outputs: torch.Tensor):
        return self.feature.decode(outputs)


# this script is a base template for programmatically generating
# demo notebooks
def demo():
    """### import and arguments"""
    import mimikit as mmk

    # DATA

    # list of files or directories to use as data
    sources = ['./data']
    # audio sample rate
    sr = 22050
    # the size of the stft
    n_fft = 2048
    # hop_length of the
    hop_length = n_fft // 4

    # NETWORK

    # the number of layers determines 'how much past' is used to predict the next future step
    # here you can make blocks of layers by specifying a tuple of integers, e.g. (2, 3, 2)
    n_layers = (3, )
    # how many parameters pro layer (must be divisible by 2)
    gate_dim = 1024
    # this multiplies the number of parameters used in the input & output layers.
    n_heads = 4

    # OPTIMIZATION

    # how many epochs should we train for
    max_epochs = 50
    # how many examples are used pro training steps
    batch_size = 16
    # how long are the examples in each training step
    # must be bigger than the network's receptive field
    batch_seq_length = sum(2**n for n in n_layers) * 2
    # the learning rate
    max_lr = 5e-4
    # betas control how fast the network changes its 'learning course'.
    # generally, betas should be close but smaller than 1. and be balanced with the batch_size :
    # the smaller the batch, the higher the betas 'could be'.
    betas = (0.9, 0.93)

    # MONITORING

    # how often should the network generate during training
    every_n_epochs = 2
    # how many examples from random prompts should be generated
    n_examples = 3
    # how many steps (1 step = 1 fft frame) should be generated
    n_steps = 1000

    # make sure batches are long enough
    rf = sum(2**n for n in n_layers)
    assert batch_seq_length > rf, f"batch_seq_length ({batch_seq_length}) needs to be greater than the receptive field ({rf})"
    print("arguments are ok!")

    """### create the data"""
    schema = mmk.FreqNet.schema(sr, n_fft=n_fft, hop_length=hop_length,
                                segment_labels=False, files_labels=False)

    db_path = 'freqnet-demo.h5'
    print("collecting data...")
    db = mmk.Database.create(db_path, sources, schema)
    print("successfully created the db.")

    """### create network and train"""
    net = mmk.FreqNet(
        **mmk.FreqNet.dependant_hp(db),
        n_layers=n_layers,
        gate_dim=gate_dim,
        groups=2,
        n_input_heads=n_heads,
        n_output_heads=n_heads,
        batch_size=batch_size,
        batch_seq_length=batch_seq_length,
        max_lr=max_lr,
        div_factor=5,
        betas=betas,

    )
    print(net.hparams)

    dm = mmk.DataModule(net, db, splits=tuple())

    cb = mmk.GenerateCallBack(every_n_epochs,
                              indices=[None]*n_examples,
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


if __name__ == '__main__':
    demo()
