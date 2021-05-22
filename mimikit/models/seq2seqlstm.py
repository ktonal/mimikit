import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn

from ..audios.features import MagSpec
from ..h5data import Database
from ..model_parts import SuperAdam, SequenceModel, DataPart
from ..ds_utils import ShiftedSequences
from ..networks.seq2seq_lstms import Seq2SeqLSTM
from ..loss_functions import mean_L1_prop


class MagSpecDB(Database):
    fft = None

    @staticmethod
    def extract(path, n_fft=2048, hop_length=512, sr=22050):
        return MagSpec.extract(path, n_fft, hop_length, sr)

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


class Seq2SeqLSTMModel(Seq2SeqLSTM,
                       DataPart,
                       SuperAdam,
                       SequenceModel,
                       pl.LightningModule):

    @staticmethod
    def loss_fn(output, target):
        return {"loss": mean_L1_prop(output, target)}

    db_class = MagSpecDB

    def __init__(self,
                 shift=12,
                 model_dim=1024,
                 num_layers=1,
                 bottleneck="add",
                 n_fc=1,
                 max_lr=1e-3,
                 betas=(.9, .9),
                 div_factor=3.,
                 final_div_factor=1.,
                 pct_start=.25,
                 cycle_momentum=False,
                 db: [Database, str] = None,
                 batch_size=64,
                 in_mem_data: bool = True,
                 splits: [list, None] = [.8, .2],
                 keep_open=False,
                 **loaders_kwargs,
                 ):
        super(pl.LightningModule, self).__init__()
        SequenceModel.__init__(self)
        DataPart.__init__(self, db, in_mem_data, splits, keep_open, batch_size=batch_size, **loaders_kwargs)
        # SuperAdam.__init__(self, lr, alpha, eps, weight_decay, momentum, centered)
        SuperAdam.__init__(self, max_lr, betas, div_factor, final_div_factor, pct_start, cycle_momentum)
        input_dim = self.hparams.n_fft // 2 + 1
        Seq2SeqLSTM.__init__(self, input_dim, model_dim, num_layers, bottleneck, n_fc)
        self.save_hyperparameters()

    def batch_info(self, *args, **kwargs):
        lengths = (self.hparams.shift, self.hparams.shift,)
        shifts = (0, self.hparams.shift)
        return dict(shifts=shifts, lengths=lengths)

    def decode_outputs(self, outputs: torch.Tensor):
        return MagSpec.decode(outputs, self.hparams.n_fft, self.hparams.hop_length)

    def get_prompts(self, n_prompts, prompt_length=None):
        return next(iter(self.datamodule.train_dataloader()))[0][:n_prompts, :prompt_length]

    def generate(self, prompt, n_steps, decode_outputs=False, **kwargs):
        self.before_generate()

        shift = self.hparams.shift
        output = self.prepare_prompt(prompt, shift * n_steps, at_least_nd=3)
        prior_t = prompt.size(1)

        for t in self.generate_tqdm(range(prior_t, prior_t + (shift * n_steps), shift)):
            output.data[:, t:t+shift] = self.forward(output[:, t-shift:t])

        if decode_outputs:
            output = self.decode_outputs(output)

        self.after_generate()

        return output

