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
    'main'
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


def main(
        sources='./data/RANDOLF_SHORT_LOW.mp3',
        sr=22050, n_fft=2048, hop_length=512,
        segments_labels=False, files_labels=False):
    import mimikit as mmk
    import os

    schema = mmk.FreqNet.schema(sr, emphasis=0., n_fft=n_fft, hop_length=hop_length,
                                segment_labels=True, files_labels=True)

    db_path = '/tmp/freqnet_db.h5'
    db = mmk.Database.create(db_path, sources, schema)

    net = mmk.FreqNet(
        **mmk.FreqNet.dependant_hp(db),
        cin_dim=128,
        gin_dim=128,
        n_layers=(6,),
        gate_dim=2048,
        groups=4,
        batch_size=16,
        batch_seq_length=128,
        max_lr=1e-3,
        div_factor=10,
        betas=(.9, .94),

    )
    print(net.hparams)

    dm = mmk.DataModule(net, db,
                        splits=tuple())

    cb = mmk.GenerateCallBack(1, indices=[None]*4,
                              n_steps=1000,
                              play_audios=False,
                              plot_audios=False,
                              log_audios=True,
                              log_dir=os.path.abspath('outputs/freqnet3'))

    trainer = mmk.get_trainer(root_dir=None,
                              max_epochs=100,
                              limit_train_batches=20,
                              callbacks=[cb],
                              checkpoint_callback=False)

    trainer.fit(net, datamodule=dm)

    prp = dm.get_prompts([None] * 4)
    print(prp.size())

    out = net.generate(prp, 32)
    print('Done!')


if __name__ == '__main__':
    main()
