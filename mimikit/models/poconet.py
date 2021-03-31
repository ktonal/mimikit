import torch
import torch.nn as nn
from torchaudio.transforms import GriffinLim
import pytorch_lightning as pl
import numpy as np

from ..audios import transforms as A
from ..kit.db_dataset import DBDataset
from ..kit.ds_utils import ShiftedSequences

from ..kit import SuperAdam, SequenceModel, DataSubModule

from ..kit.networks.poconet import PocoNetNetwork


class PocoNetDB(DBDataset):
    features = ["fft"]
    fft = None

    @staticmethod
    def extract(path, n_fft=2048, hop_length=512, sr=22050):
        y = A.FileTo.signal(path, sr)
        fft = SignalTo.polar_spec(y, n_fft, hop_length)
        params = dict(n_fft=n_fft, hop_length=hop_length, sr=sr)
        return dict(fft=(params, fft.transpose((0, 2, 1)), None))

    def prepare_dataset(self, model, datamodule):
        prm = model.batch_info()
        self.slicer = ShiftedSequences(self.fft.shape[1], list(zip(prm["shifts"], prm["lengths"])))
        datamodule.loader_kwargs.setdefault("drop_last", False)
        datamodule.loader_kwargs.setdefault("shuffle", True)

    def __getitem__(self, item):
        slices = self.slicer(item)
        return tuple(self.fft[:, sl] for sl in slices)

    def __len__(self):
        return len(self.slicer)


def l1_loss_with_phs(output, target):
    y = target
    x = output
    norm = target[:, 0].abs().sum(dim=(0, -1), keepdim=True)
    cd = (torch.cos(y[:, 1]) - torch.cos(x[:, 1]))
    sd = (torch.sin(y[:, 1]) - torch.sin(x[:, 1]))
    phserr = torch.mean(torch.norm(torch.stack((sd, cd)) * torch.sqrt((y[:, 0] / norm + 0.01)), dim=0))
    L = nn.L1Loss(reduction="none")(output[:, 0], target[:, 0]).sum(dim=(0, -1), keepdim=True)
    return 100 * (L / norm).mean(), 100 * phserr


class PocoNet(PocoNetNetwork,
              DataSubModule,
              SuperAdam,
              SequenceModel,
              pl.LightningModule):

    @staticmethod
    def loss_fn(output, target):
        return l1_loss_with_phs(output, target)

    db_class = PocoNetDB

    def __init__(self,
                 n_layers=(4,),
                 cin_dim=None,
                 gin_dim=None,
                 gate_dim=128,
                 kernel_size=2,
                 groups=1,
                 accum_outputs=0,
                 pad_input=0,
                 skip_dim=None,
                 residuals_dim=None,
                 n_1x1layers=3,
                 n_2x3layers=2,
                 dim2x3=128,
                 dim1x1=128,
                 phs_groups=1,
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
        self.hparams.n_fft = self.db.params.fft["n_fft"]
        self.hparams.hop_length = self.db.params.fft["hop_length"]
        self.hparams.input_dim = self.hparams.n_fft // 2 + 1
        if hasattr(db, "labels") and cin_dim is not None:
            n_cin_classes = db.params.labels["n_classes"]
        else:
            n_cin_classes = None
        if hasattr(db, "g_labels") and gin_dim is not None:
            n_gin_classes = db.params.g_labels["n_classes"]
        else:
            n_gin_classes = None
        # noinspection PyArgumentList
        PocoNetNetwork.__init__(self,
                                n_layers=n_layers, input_dim=self.hparams.n_fft // 2 + 1,
                                n_cin_classes=n_cin_classes, cin_dim=cin_dim,
                                n_gin_classes=n_gin_classes, gin_dim=gin_dim,
                                gate_dim=gate_dim, kernel_size=kernel_size, groups=groups,
                                dim1x1=dim1x1, dim2x3=dim2x3, n_1x1layers=n_1x1layers, n_2x3layers=n_2x3layers,
                                phs_groups=phs_groups, accum_outputs=accum_outputs, pad_input=pad_input,
                                skip_dim=skip_dim, residuals_dim=residuals_dim)
        self.save_hyperparameters()

    def setup(self, stage: str):
        super().setup(stage)
        self.center_adv.to('cuda')

    def batch_info(self, *args, **kwargs):
        lengths = (self.hparams.batch_seq_length, self.output_shape((-1, self.hparams.batch_seq_length, -1))[1])
        shifts = (0, self.shift)
        return dict(shifts=shifts, lengths=lengths)

    def training_step(self, batch, batch_idx):
        batch, target = batch
        output = self.forward(batch)
        mag_loss, phs_loss = self.loss_fn(output, target)
        return {"loss": mag_loss+phs_loss, "mag_loss": mag_loss, "phs_loss": phs_loss}

    def validation_step(self, batch, batch_idx):
        batch, target = batch
        output = self.forward(batch)
        mag_loss, phs_loss = self.loss_fn(output, target)
        return {"val_loss":  mag_loss+phs_loss, "val_mag_loss": mag_loss, "val_phs_loss": phs_loss}

    def generation_slices(self):
        # input is always the last receptive field
        input_slice = slice(-self.receptive_field, None)
        if self.pad_input == 1:
            # then there's only one future time-step : the last output
            output_slice = slice(-1, None)
        else:
            # for all other cases, the first output has the shift of the whole network
            output_slice = slice(None, 1)
        return input_slice, output_slice

    # untested - don't expect it to work
    def decode_outputs(self, outputs: torch.Tensor):
        mag = outputs[:, 0]
        phs = outputs[:, 1]
        spec = torch.exp(phs * 1j) * mag
        hann_window = torch.hann_window(self.hparams.n_fft)
        signal = torch.istft(spec, self.hparams.n_fft, self.hparams.hop_length, window=hann_window)
        # gla = GriffinLim(n_fft=self.hparams.n_fft, hop_length=self.hparams.hop_length, power=1.,
        #                  wkwargs=dict(device=outputs.device))
        return signal.transpose(-1, -2).contiguous()

    def generate(self, prompt, n_steps, decode_outputs=False, **kwargs):
        self.before_generate()

        output = self.prepare_prompt(prompt, n_steps, at_least_nd=3)
        prior_t = prompt.size(1)
        rf = self.receptive_field
        _, out_slc = self.generation_slices()

        for t in self.generate_tqdm(range(prior_t, prior_t + n_steps)):
            new_data = self.forward(output[:, :, t - rf:t])[:, :, out_slc]
            new_data[:, 1] = self.principarg(new_data[:, 1])
            output.data[:, :, t:t + 1] = new_data

        if decode_outputs:
            output = self.decode_outputs(output)

        self.after_generate()

        return output

    @staticmethod
    def principarg(x):
        return x - 2.0 * np.pi * torch.round(x / (2.0 * np.pi))
