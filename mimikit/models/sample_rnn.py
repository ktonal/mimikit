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
    'main'
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
            Input('qx', AsFramedSlice(shifts[-1], batch_seq_len, frame_size=frame_sizes[-1],
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


@make_click_command
def main(sources=['./data'], sr=16000, q_levels=256):
    import mimikit as mmk
    import os

    schema = mmk.SampleRNN.schema(sr, 0., q_levels)

    net = mmk.SampleRNN(
        feature=schema['qx'],
        q_levels=q_levels,
        frame_sizes=(16, 4, 4),
        n_rnn=2,

        max_lr=7e-4,
        betas=(.9, .91),
        div_factor=5,
    )

    print(net.hparams)

    dm = mmk.DataModule(net, "/tmp/srnn.h5",
                        sources=sources, schema=schema,
                        splits=tuple(),
                        in_mem_data=True)

    cb = mmk.GenerateCallBack(5, indices=[None] * 4,
                              n_steps=16000*10,
                              play_audios=False,
                              plot_audios=False,
                              log_audios=True,
                              log_dir=os.path.abspath('outputs/sample-rnn'),
                              temperature=torch.tensor([[.9], [.999], [1.1], [1.25]]).to('cuda'))

    trainer = mmk.get_trainer(root_dir=None,
                              max_epochs=50,
                              callbacks=[cb],
                              checkpoint_callback=False)

    trainer.fit(net, datamodule=dm)


if __name__ == '__main__':
    main()
