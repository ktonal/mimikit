import torch

from ..abstract.features import SegmentLabels, FilesLabels
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


class FreqNetData(IData):

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
            feature=db.schema['fft'], input_dim=db.schema['fft'].dim, **hp
        )

    def batch_signature(self, stage='fit'):
        inpt = Input('fft', AsSlice(shift=0, length=self.batch_seq_length))
        trgt = Target('fft', AsSlice(shift=self.shift,
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
        sources='./gould',
        sr=22050, n_fft=2048, hop_length=512,
        segments_labels=False, files_labels=False):
    import mimikit as mmk

    schema = mmk.FreqNet.schema(sr, emphasis=0., n_fft=n_fft, hop_length=hop_length,
                                segment_labels=True, files_labels=True)

    db_path = '/tmp/freqnet_db.h5'
    # if not os.path.exists(db_path):
    db = mmk.Database.create(db_path, sources, schema)
    # else:
    #     db = mmk.Database(db_path)
    print(db._visit())
    print(db.loc[:])
    print(db.glob[:])

    net = mmk.FreqNet(
        **mmk.FreqNet.dependant_hp(db),
        cin_dim=256,
        gin_dim=256
    )
    print(net.hparams)

    dm = mmk.DataModule(net, db,
                        splits=(0.8, 0.2))
    trainer = mmk.get_trainer(root_dir=None,
                              max_epochs=1,
                              limit_train_batches=4)

    trainer.fit(net, datamodule=dm)

    prp = dm.get_prompts([None] * 4)
    print(prp.size())

    out = net.generate(prp, 32)
    print('Done!')


if __name__ == '__main__':
    main()
