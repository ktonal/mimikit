import torch.nn as nn
import torch.nn.functional as F

from . import FreqNet
from .modules.io import PermuteTF
from ..kit.data import DBDataset
from ..data import transforms as T
from ..kit.ds_utils import ShiftedSequences


class WaveNetDB(DBDataset):

    qx = None

    @staticmethod
    def extract(path, mu=255, sr=22050):
        qx = T.FileTo.mu_law_compress(path, sr, mu)
        return dict(qx=(dict(mu=mu, sr=sr), qx.reshape(-1, 1), None))

    def prepare_dataset(self, model):
        args = model.targets_shifts_and_lengths(model.hparams["input_seq_length"])
        self.slicer = ShiftedSequences(len(self.qx), [(0, model.hparams["input_seq_length"])] + args)

    def __getitem__(self, item):
        slices = self.slicer(item)
        return tuple(self.qx[sl] for sl in slices)

    def __len__(self):
        return len(self.slicer)


def wavenet_loss_fn(output, target):
    return F.nll_loss(output.view(-1, output.size(-1)), target.view(-1))


class WaveNet(FreqNet):

    def __init__(self,
                 input_dim=256,
                 model_dim=256,
                 groups=1,
                 n_layers=(1,),
                 with_skip_conv=True,
                 with_residual_conv=True,
                 **data_optim_kwargs
                 ):
        super(WaveNet, self).__init__(
            loss_fn=wavenet_loss_fn,
            model_dim=model_dim,
            groups=groups,
            n_layers=n_layers,
            strict=False,
            accum_outputs="right",
            concat_outputs=None,
            pad_input="left",
            learn_padding=False,
            with_skip_conv=with_skip_conv,
            with_residual_conv=with_residual_conv,
            **data_optim_kwargs)

        self.input_dim = input_dim

        self.inpt = nn.Sequential(
            nn.Embedding(self.input_dim, self.model_dim), PermuteTF())

        self.outpt = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Conv1d(self.model_dim, self.model_dim, 1),
            nn.ReLU(inplace=True),
            nn.Conv1d(self.model_dim, self.input_dim, 1),
            PermuteTF(),
            nn.LogSoftmax(dim=1)
        )

        self.save_hyperparameters()

    def forward(self, x):
        x = self.inpt(x.squeeze())
        skips = None
        for layer in self.layers:
            x, skips = layer(x, skips)
        x = self.outpt(skips)
        return x
