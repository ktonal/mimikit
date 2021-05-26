import torch
import torch.nn as nn
from torchaudio.transforms import GriffinLim
import pytorch_lightning as pl

from ..audios import transforms as A
from ..kit.db_dataset import DBDataset
from ..kit.ds_utils import ShiftedSequences
from ..kit.loss_functions import mean_L1_prop
from ..kit import SuperAdam, SequenceModel, DataSubModule

from ..kit.networks.freqnet import FreqNetNetwork


class FreqNetDB(DBDataset):
    features = ["fft"]
    fft = None

    @staticmethod
    def extract(path, n_fft=2048, hop_length=512, sr=22050):
        y = A.FileTo.signal(path, sr)
        y = A.normalize(y)
        fft = A.SignalTo.mag_spec(y, n_fft, hop_length)
        params = dict(n_fft=n_fft, hop_length=hop_length, sr=sr)
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


class FreqNet(FreqNetNetwork,
              DataSubModule,
              SuperAdam,
              SequenceModel,
              pl.LightningModule):

    @staticmethod
    def loss_fn(output, target):
        return {"loss": mean_L1_prop(output, target)}

    db_class = FreqNetDB

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
        if hasattr(db, "labels") and cin_dim is not None:
            n_cin_classes = db.params.labels["n_classes"]
        else:
            n_cin_classes = None
        if hasattr(db, "g_labels") and gin_dim is not None:
            n_gin_classes = db.params.g_labels["n_classes"]
        else:
            n_gin_classes = None
        # noinspection PyArgumentList
        FreqNetNetwork.__init__(self,
                                n_layers=n_layers, input_dim=self.db.fft.shape[1],
                                n_cin_classes=n_cin_classes, cin_dim=cin_dim,
                                n_gin_classes=n_gin_classes, gin_dim=gin_dim,
                                gate_dim=gate_dim, kernel_size=kernel_size, groups=groups,
                                accum_outputs=accum_outputs, pad_input=pad_input,
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

    def get_prompts(self, n_prompts, prompt_length=None):
        return next(iter(self.datamodule.train_dataloader()))[0][:n_prompts, :prompt_length]

    def generate(self, prompt, n_steps, decode_outputs=False, **kwargs):
        self.before_generate()

        output = self.prepare_prompt(prompt, n_steps, at_least_nd=3)
        prior_t = prompt.size(1)
        rf = self.receptive_field
        _, out_slc = self.generation_slices()

        for t in self.generate_tqdm(range(prior_t, prior_t + n_steps)):
            output.data[:, t:t+1] = self.forward(output[:, t-rf:t])[:, out_slc]

        if decode_outputs:
            output = self.decode_outputs(output)

        self.after_generate()

        return output
