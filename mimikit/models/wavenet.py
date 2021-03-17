import torch.nn as nn
import torch
from torchaudio.transforms import MuLawDecoding
import pytorch_lightning as pl

from ..kit import DBDataset, ShiftedSequences
from ..audios import transforms as A

from ..kit import SuperAdam, SequenceModel, DataSubModule

from ..kit.networks.wavenet import WNNetwork


class WaveNetDB(DBDataset):
    qx = None

    @staticmethod
    def extract(path, mu=256, sr=22050):
        qx = A.FileTo.mu_law_compress(path, sr, mu - 1)
        return dict(qx=(dict(mu=mu, sr=sr), qx.reshape(-1, 1), None))

    def prepare_dataset(self, model, datamodule):
        prm = model.batch_info()
        self.slicer = ShiftedSequences(len(self.qx), list(zip(prm["shifts"], prm["lengths"])))
        datamodule.loader_kwargs.setdefault("drop_last", False)
        datamodule.loader_kwargs.setdefault("shuffle", True)

    def __getitem__(self, item):
        slices = self.slicer(item)
        return tuple(self.qx[sl].reshape(-1) for sl in slices)

    def __len__(self):
        return len(self.slicer)


class WaveNet(WNNetwork,
              DataSubModule,
              SuperAdam,
              SequenceModel,
              pl.LightningModule):

    @staticmethod
    def loss_fn(output, target):
        criterion = nn.CrossEntropyLoss(reduction="mean")
        return criterion(output.view(-1, output.size(-1)), target.view(-1))

    db_class = WaveNetDB

    def __init__(self,
                 n_layers=(4,),
                 q_levels=256,
                 n_cin_classes=None,
                 cin_dim=None,
                 n_gin_classes=None,
                 gin_dim=None,
                 layers_dim=128,
                 kernel_size=2,
                 groups=1,
                 accum_outputs=0,
                 pad_input=0,
                 skip_dim=None,
                 residuals_dim=None,
                 max_lr=1e-3,
                 betas=(.9, .9),
                 div_factor=3.,
                 final_div_factor=1.,
                 pct_start=.25,
                 cycle_momentum=False,
                 batch_seq_length=64,
                 db=None,
                 batch_size=64,
                 in_mem_data=True,
                 splits=[.8, .2],
                 **loaders_kwargs
                 ):
        super(pl.LightningModule, self).__init__()
        SequenceModel.__init__(self)
        DataSubModule.__init__(self, db, in_mem_data, splits, batch_size=batch_size, **loaders_kwargs)
        SuperAdam.__init__(self, max_lr, betas, div_factor, final_div_factor, pct_start, cycle_momentum)
        # noinspection PyArgumentList
        WNNetwork.__init__(self, n_layers=n_layers, q_levels=q_levels, n_cin_classes=n_cin_classes, cin_dim=cin_dim,
                           n_gin_classes=n_gin_classes, gin_dim=gin_dim,
                           layers_dim=layers_dim, kernel_size=kernel_size, groups=groups, accum_outputs=accum_outputs,
                           pad_input=pad_input,
                           skip_dim=skip_dim, residuals_dim=residuals_dim)
        self.save_hyperparameters()

    def setup(self, stage: str):
        super().setup(stage)

    def batch_info(self, *args, **kwargs):
        lengths = (self.hparams.batch_seq_length, self.output_shape((-1, self.hparams.batch_seq_length, -1))[1])
        shifts = (0, self.shift)
        return dict(shifts=shifts, lengths=lengths)

    def generation_slices(self):
        # input is always the last receptive field
        input_slice = slice(-self.receptive_field(), None)
        if self.pad_input == 1:
            # then there's only one future time-step : the last output
            output_slice = slice(-1, None)
        else:
            # for all other cases, the first output has the shift of the whole network
            output_slice = slice(None, 1)
        return input_slice, output_slice

    def decode_outputs(self, outputs):
        decoder = MuLawDecoding(self.hparams.q_levels)
        return decoder(outputs)
