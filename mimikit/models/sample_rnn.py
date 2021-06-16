import inspect
import click
import torch
import torch.nn as nn
import dataclasses as dtc

from ..audios.features import MuLawSignal
from ..data import TBPTTSampler, Feature, AsSlice, AsFramedSlice, Input, Target
from .parts import SuperAdam, SequenceModel
from ..networks.sample_rnn import SampleRNNNetwork

from . import model, IData

__all__ = [
    'SampleRNNData',
    'SampleRNN',
    'demo'
]


@dtc.dataclass
class SampleRNNData(IData):
    feature: Feature = None
    batch_size: int = 16
    chunk_len: int = 16000 * 8
    batch_seq_len: int = 512

    @classmethod
    def schema(cls, sr=22050, emphasis=0., q_levels=256):
        return {'qx': MuLawSignal(sr=sr, emphasis=emphasis, q_levels=q_levels)}

    def batch_signature(self, stage='fit'):
        batch_seq_len, frame_sizes = tuple(getattr(self, key) for key in ["batch_seq_len", "frame_sizes"])
        shifts = [frame_sizes[0] - size for size in frame_sizes]
        inputs = []
        for fs, shift in zip(frame_sizes[:-1], shifts[:-1]):
            inputs.append(
                Input('qx', AsFramedSlice(shift, batch_seq_len, frame_size=fs,
                                          as_strided=False)))
        inputs.append(
            Input('qx', AsFramedSlice(shifts[-1], batch_seq_len+frame_sizes[-1]-1, frame_size=frame_sizes[-1],
                                      as_strided=True)))
        targets = Target('qx', AsSlice(shift=frame_sizes[0], length=batch_seq_len))
        if stage in ('fit', 'train', 'val'):
            return inputs, targets
        # test, predict, generate
        return Input('qx', AsSlice(0, batch_seq_len))

    @classmethod
    def dependant_hp(cls, db):
        return dict(feature=db.schema['qx'], q_levels=db.schema['qx'].q_levels)

    def loader_kwargs(self, stage, datamodule):
        ds = datamodule.datasets[stage]
        N = len(ds)
        batch_sampler = TBPTTSampler(N,
                                     self.batch_size,
                                     self.chunk_len,
                                     self.batch_seq_len)
        return dict(batch_sampler=batch_sampler)


@model
class SampleRNN(
    # db schema, batch sig
    SampleRNNData,
    # training step, generate routine & interface
    SequenceModel,
    # optimizer & scheduler
    SuperAdam,
    # nn.Module, generate_ function
    SampleRNNNetwork
):

    def setup(self, stage: str):
        SequenceModel.setup(self, stage)
        SuperAdam.setup(self, stage)

    @staticmethod
    def loss_fn(output, target):
        criterion = nn.CrossEntropyLoss(reduction="mean")
        return {"loss": criterion(output.view(-1, output.size(-1)), target.view(-1))}

    def on_train_batch_start(self, batch, batch_idx, dataloader_idx):
        if (batch_idx * self.batch_seq_len) % self.chunk_len == 0:
            self.reset_h0()

    def encode_inputs(self, inputs: torch.Tensor):
        return self.feature.encode(inputs)

    def decode_outputs(self, outputs: torch.Tensor):
        return self.feature.decode(outputs)


def make_click_command(f):
    aspec = inspect.getfullargspec(f)
    if aspec.defaults is not None:
        for arg, default in zip(reversed(aspec.args), reversed(aspec.defaults)):
            opt = click.option('--' + arg.replace("_", "-"), default=default)
            f = opt(f)
    return click.command()(f)


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

    # how many samples each tier receives as input
    # stick to decreasing sequences, size_i must be divisible by size_i+1 and
    # last 2 numbers must be equal. You can have as many tiers as you want.
    frame_sizes = (16, 4, 4)
    # number of lstm network pro tier
    n_rnn = 2
    # dimensionality of the lstms
    dim = 512

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

    # MONITORING

    # how often should the network generate during training
    every_n_epochs = 4
    # how many examples from random prompts should be generated
    n_examples = 3
    # how many steps (1 step = 1 sample) should be generated
    n_steps = 15 * sr
    # the sampling temperature changes outputs a lot!
    # roughly : prefer values close to 1. & hot -> noisy ; cold -> silence
    temperature = torch.tensor([.9, .999, 1.25]).unsqueeze(1)

    assert temperature.size(0) == n_examples, "number of values in temperature must be equal to n_examples"
    print("arguments are ok!")

    """### create the data"""
    schema = mmk.SampleRNN.schema(sr, 0., q_levels)

    db_path = 'sample-rnn-demo.h5'
    print("collecting data...")
    db = mmk.Database.create(db_path, sources, schema)
    if not len(db.qx.files):
        raise ValueError("Empty db. No audio files were found...")
    print("successfully created the db.")

    """### create network and train"""
    net = mmk.SampleRNN(
        feature=schema['qx'],
        q_levels=q_levels,
        frame_sizes=frame_sizes,
        n_rnn=n_rnn,
        dim=dim,
        mlp_dim=dim,
        batch_size=batch_size,
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
                              checkpoint_callback=False)
    print("here we go!")
    trainer.fit(net, datamodule=dm)

    """----------------------------"""


if __name__ == '__main__':
    demo()
