import dataclasses as dtc
import torch.nn as nn
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
            schema.update({
                'loc': SegmentLabels(input_key='qx')
            })
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
        inpt = Input('qx', AsSlice(shift=0, length=self.batch_seq_length))
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
