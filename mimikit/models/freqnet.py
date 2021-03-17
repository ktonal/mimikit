import torch
import torch.nn as nn
from torchaudio.transforms import GriffinLim
import pytorch_lightning as pl

from ..audios import transforms as A
from ..kit.db_dataset import DBDataset
from ..kit.ds_utils import ShiftedSequences
from ..kit.sub_models.utils import tqdm

from ..kit import SuperAdam, SequenceModel, DataSubModule

from ..kit.networks.freqnet import FreqNetNetwork


class FreqNetDB(DBDataset):
    features = ["fft"]
    fft = None

    @staticmethod
    def extract(path, n_fft=2048, hop_length=512, sr=22050):
        params = dict(n_fft=n_fft, hop_length=hop_length, sr=sr)
        fft = A.FileTo.mag_spec(path, **params)
        return dict(fft=(params, fft.T, None))

    def prepare_dataset(self, model, datamodule):
        prm = model.batch_info()
        self.slicer = ShiftedSequences(len(self.fft), list(zip(prm["shifts"], prm["lengths"])))
        datamodule.loader_kwargs.setdefault("drop_last", False)
        datamodule.loader_kwargs.setdefault("shuffle", True)

    def __getitem__(self, item):
        slices = self.slicer(item)
        return tuple(self.fft[sl] for sl in slices)

    def __len__(self):
        return len(self.slicer)


def mean_L1_prop(output, target):
    L = nn.L1Loss(reduction="none")(output, target).sum(dim=(0, -1), keepdim=True)
    return 100 * (L / target.abs().sum(dim=(0, -1), keepdim=True)).mean()


class FreqNet(FreqNetNetwork,
              DataSubModule,
              SuperAdam,
              SequenceModel,
              pl.LightningModule):

    @staticmethod
    def loss_fn(output, target):
        return mean_L1_prop(output, target)

    db_class = FreqNetDB

    def __init__(self,
                 n_layers=(4,),
                 input_dim=1025,
                 n_cin_classes=None,
                 cin_dim=None,
                 n_gin_classes=None,
                 gin_dim=None,
                 gate_dim=128,
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
        FreqNetNetwork.__init__(self,
                                n_layers=n_layers, input_dim=input_dim,
                                n_cin_classes=n_cin_classes, cin_dim=cin_dim,
                                n_gin_classes=n_gin_classes, gin_dim=gin_dim,
                                gate_dim=gate_dim, kernel_size=kernel_size, groups=groups,
                                accum_outputs=accum_outputs, pad_input=pad_input,
                                skip_dim=skip_dim, residuals_dim=residuals_dim)
        self.hparams.n_fft = self.db.fft.attrs["n_fft"]
        self.hparams.hop_length = self.db.fft.attrs["hop_length"]
        self.save_hyperparameters()

    def setup(self, stage: str):
        super().setup(stage)

    def batch_info(self, *args, **kwargs):
        lengths = (self.hparams.batch_seq_length, self.output_shape((-1, self.hparams.batch_seq_length, -1))[1])
        shifts = (0, self.shift)
        return dict(shifts=shifts, lengths=lengths)

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

    def decode_outputs(self, outputs: torch.Tensor):
        gla = GriffinLim(n_fft=self.hparams.n_fft, hop_length=self.hparams.hop_length, power=1.,
                         wkwargs=dict(device=outputs.device))
        return gla(outputs.transpose(-1, -2).contiguous())

    def generate(self, prompt, n_steps, decode_outputs=False, **kwargs):
        # prepare model
        was_training = self.training
        initial_device = self.device
        self.eval()
        self.to("cuda" if torch.cuda.is_available() else "cpu")

        # prepare prompt
        if not isinstance(prompt, torch.Tensor):
            prompt = torch.from_numpy(prompt)
        if len(prompt.shape) == 2:
            prompt = prompt.unsqueeze(0)
        new = prompt.to(self.device)

        inpt_slc, outpt_slc = self.generation_slices()

        for _ in tqdm(range(n_steps),
                      desc="Generate", dynamic_ncols=True, leave=False, unit="step"):
            with torch.no_grad():
                outpt = self.forward(new[:, inpt_slc])
                new = torch.cat((new, outpt[:, outpt_slc]), dim=1)

        # reset model
        self.to(initial_device)
        self.train() if was_training else None

        if decode_outputs:
            new = self.decode_outputs(new)

        return new
