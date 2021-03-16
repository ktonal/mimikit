import torch.nn.functional as F
import torch.nn as nn
import torch
import pytorch_lightning as pl

from ..kit import DBDataset, ShiftedSequences
from ..audios import transforms as A

from ..kit import SuperAdam, SequenceModel, DataSubModule

from ..kit.networks.wavenet import WNNetwork


class WaveNetDB(DBDataset):
    qx = None

    @staticmethod
    def extract(path, mu=256, sr=22050):
        qx = A.FileTo.mu_law_compress(path, sr, mu-1)
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


def wavenet_loss_fn(output, target):
    output = nn.LogSoftmax(dim=-1)(output)
    return F.nll_loss(output.view(-1, output.size(-1)), target.view(-1))


class WaveNet(WNNetwork,
              DataSubModule,
              SuperAdam,
              SequenceModel,
              pl.LightningModule):

    loss_fn = property(lambda self: wavenet_loss_fn)
    db_class = WaveNetDB

    def __init__(self,
                 n_layers=(4,),
                 mu=256,
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
                 files=None,
                 batch_size=64,
                 in_mem_data=True,
                 splits=[.8, .2],
                 **loaders_kwargs
                 ):
        super(pl.LightningModule, self).__init__()
        SequenceModel.__init__(self)
        DataSubModule.__init__(self, db, files, in_mem_data, splits, batch_size=batch_size, **loaders_kwargs)
        SuperAdam.__init__(self, max_lr, betas, div_factor, final_div_factor, pct_start, cycle_momentum)

        WNNetwork.__init__(self, n_layers, mu, n_cin_classes, cin_dim, n_gin_classes, gin_dim,
                           layers_dim, kernel_size, groups, accum_outputs, pad_input,
                           skip_dim, residuals_dim)
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
        if not self.strict and self.pad_input == 1:
            # then there's only one future time-step : the last output
            output_slice = slice(-1, None)
        elif self.strict and self.pad_input == 1:
            # then there are as many future-steps as they are layers and they all are
            # at the end of the outputs
            output_slice = slice(-len(self.layers), None)
        else:
            # for all other cases, the first output has the shift of the whole network
            output_slice = slice(None, 1)
        return input_slice, output_slice

    def generate_step(self, so_far_generated, step_idx):
        input_slice, output_slice = self.generation_slices()
        with torch.no_grad():
            out = self.forward(so_far_generated[:, input_slice])
            out = nn.Softmax(dim=-1)(out)
        return out[:, output_slice]
