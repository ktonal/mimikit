import dataclasses as dtc
import torch.nn as nn
from itertools import accumulate
import operator
import torch

from ..abstract.features import SegmentLabels, FilesLabels
from ..audios import MuLawSignal
from ..data import Feature, Input, AsSlice, Target
from . import model
from ..networks import WNNetwork
from .parts import IData, SuperAdam, SequenceModel

__all__ = [
    'WaveNetData',
    'WaveNet'
]


@dtc.dataclass
class WaveNetData(IData):
    feature: Feature = None
    batch_size: int = 32
    batch_seq_length: int = 64

    @classmethod
    def schema(cls, sr=22050, emphasis=0., q_levels=256,
               segment_labels=False, files_labels=False):

        schema = {"qx": MuLawSignal(sr=sr, emphasis=emphasis, q_levels=q_levels)}
        if segment_labels:
            # not yet supported...
            pass
            # schema.update({
            #     'loc': SegmentLabels(input_key='qx')
            # })
        if files_labels:
            schema.update({
                'glob': FilesLabels(input_key='qx')
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
            feature=db.schema['qx'], q_levels=db.schema['qx'].q_levels, **hp
        )

    def batch_signature(self, stage='fit'):
        inpt = [Input('qx', AsSlice(shift=0, length=self.batch_seq_length))]
        trgt = Target('qx', AsSlice(shift=self.shift,
                                    length=self.output_shape((-1, self.batch_seq_length, -1))[1]))
        # where are we conditioned?
        loc, glob = self.n_cin_classes is not None, self.n_gin_classes is not None
        if loc:
            pass
        if glob:
            pass
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
class WaveNet(
    WaveNetData,
    SequenceModel,
    SuperAdam,
    WNNetwork,
):

    @staticmethod
    def loss_fn(output, target):
        criterion = nn.CrossEntropyLoss(reduction="mean")
        return {"loss": criterion(output.view(-1, output.size(-1)), target.view(-1))}

    def setup(self, stage: str):
        SequenceModel.setup(self, stage)
        SuperAdam.setup(self, stage)

    def encode_inputs(self, inputs: torch.Tensor):
        return self.feature.encode(inputs)

    def decode_outputs(self, outputs: torch.Tensor):
        return self.feature.decode(outputs)

    @staticmethod
    def rf(n_layers, kernel_size):
        if isinstance(kernel_size, tuple):
            assert sum(n_layers) == len(kernel_size), "total number of layers and of kernel sizes must match"
            k_iter = kernel_size
            dilations = list(accumulate([1, *kernel_size], operator.mul))
        else:
            # reverse_dilation_order leads to the connectivity of the FFTNet
            k_iter = [kernel_size] * sum(n_layers)
            dilations = [kernel_size ** (i)
                         for block in n_layers for i in range(block)]
        seq = list(dilations[i-1] * k_iter[i-1] for i in accumulate(n_layers))
        return sum(seq) - len(seq) + 1


def demo():
    """### import and arguments"""
    import mimikit as mmk
    import torch

    # DATA

    # list of files or directories to use as data ("./" is the cwd of the notebook)
    sources = ['./data']
    # audio sample rate
    sr = 16000
    # number of quantization levels (256 -> 8-bit)
    q_levels = 256

    # NETWORK

    # the number of layers determines 'how much past' is used to predict the next future step
    # here you can make blocks of layers by specifying a tuple of integers, e.g. (2, 3, 2)
    n_layers = (3,)
    # kernel_size is the size of the convolution. You can specify a single int for the whole
    # network or one size per layer
    kernel_size = (16, 8, 2)
    # how many parameters pro convolution layer
    gate_dim = 256
    # next arg can take 3 values : -1 -> input & output are summed at the end of the input,
    # 1 -> at the beginning, 0 -> they are not summed
    accum_outputs = 0
    # the next 2 args can take integers or None. Integers add skips and/or residuals layers of this size.
    # None adds no layers
    skip_dim = None
    residuals_dim = None

    # OPTIMIZATION

    # how many epochs should we train for
    max_epochs = 50
    # how many examples are used pro training steps
    batch_size = 16
    # the learning rate
    max_lr = 5e-4
    # betas control how fast the network changes its 'learning course'.
    # generally, betas should be close but smaller than 1. and be balanced with the batch_size :
    # the smaller the batch, the higher the betas 'could be'.
    betas = (0.9, 0.93)
    # one wavenet epoch can be very long, so as to monitor the net's progress,
    # we limit the number of batches pro epoch
    limit_train_batches = 1000

    # MONITORING

    # how often should the network generate during training
    every_n_epochs = 4
    # how many examples from random prompts should be generated
    n_examples = 3
    # how many steps (1 step = 1 sample) should be generated
    n_steps = 5 * sr
    # the sampling temperature changes outputs a lot!
    # roughly : prefer values close to 1. & hot -> noisy ; cold -> silence
    temperature = torch.tensor([.9, .999, 1.25]).unsqueeze(1)

    assert temperature.size(0) == n_examples, "number of values in temperature must be equal to n_examples"
    rf = mmk.WaveNet.rf(n_layers, kernel_size)
    print("arguments are ok! The network will have a receptive field of size :", rf, "samples")

    """### create the data"""
    schema = mmk.WaveNet.schema(sr, 0., q_levels)

    db_path = 'wavenet-demo.h5'
    print("collecting data...")
    db = mmk.Database.create(db_path, sources, schema)
    if not len(db.qx.files):
        raise ValueError("Empty db. No audio files were found...")
    print("successfully created the db.")

    """### create network and train"""
    net = mmk.WaveNet(
        **mmk.WaveNet.dependant_hp(db),
        kernel_size=kernel_size,
        gate_dim=gate_dim,
        accum_outputs=accum_outputs,
        residuals_dim=residuals_dim,
        skip_dim=skip_dim,
        n_layers=n_layers,
        batch_size=batch_size,
        batch_seq_length=rf * 2 if rf <= 128 else rf + rf // 4,
        max_lr=max_lr,
        betas=betas,
        div_factor=5,
    )
    print(net.hparams)

    dm = mmk.DataModule(net, db,
                        splits=tuple(),
                        in_mem_data=True)

    cb = mmk.GenerateCallback(every_n_epochs, indices=[None] * n_examples,
                              n_steps=n_steps,
                              play_audios=True,
                              plot_audios=True,
                              temperature=temperature.to('cuda') if torch.cuda.is_available() else temperature)

    trainer = mmk.get_trainer(root_dir=None,
                              max_epochs=max_epochs,
                              callbacks=[cb],
                              limit_train_batches=limit_train_batches,
                              checkpoint_callback=False)
    print("here we go!")
    trainer.fit(net, datamodule=dm)

    """----------------------------"""


if __name__ == '__main__':
    demo()
