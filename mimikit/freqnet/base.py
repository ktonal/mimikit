import torch
from pytorch_lightning import LightningModule
import librosa
from abc import ABC

from ..data import Database, HOP_LENGTH
from ..kit import MMKHooks, LoggingHooks, tqdm
from ..kit.submodules.data import DBDataset, DataSubModule
from ..kit.ds_utils import ShiftedSequences


class ManyOneCycleLR(torch.optim.lr_scheduler.OneCycleLR):
    def step(self, epoch=None):
        try:
            super(ManyOneCycleLR, self).step(epoch=epoch)
        except ValueError:
            self.last_epoch = 0
            super(ManyOneCycleLR, self).step(epoch=epoch)


class FreqOptim(LightningModule):
    """
    simple class to modularize the optimization setup used in a ``FreqNetModel``.

    Here the method ``configure_optimizers()`` returns an ``Adam`` optimizer and a slightly modified
    ``OneCycleLR`` scheduler (the only modification being that it won't raise a ``ValueError`` if you use for more
    steps than what it expects and will instead restarts its cycle)
    """

    def __init__(self,
                 max_lr=1e-3,
                 betas=(.9, .9),
                 div_factor=3.,
                 final_div_factor=1.,
                 pct_start=.25,
                 cycle_momentum=False):
        super(FreqOptim, self).__init__()
        self.max_lr = max_lr
        self.betas = betas
        self.div_factor = div_factor
        self.final_div_factor = final_div_factor
        self.pct_start = pct_start
        self.cycle_momentum = cycle_momentum
        # those are set in setup when we know what the trainer and datamodule got
        self.steps_per_epoch = None
        self.max_epochs = None

    def configure_optimizers(self):
        self.opt = torch.optim.Adam(self.parameters(), lr=self.max_lr, betas=self.betas)
        self.sched = ManyOneCycleLR(self.opt,
                                    steps_per_epoch=self.steps_per_epoch,
                                    epochs=self.max_epochs,
                                    max_lr=self.max_lr, div_factor=self.div_factor,
                                    final_div_factor=self.final_div_factor,
                                    pct_start=self.pct_start,
                                    cycle_momentum=self.cycle_momentum
                                    )
        return [self.opt], [{"scheduler": self.sched, "interval": "step", "frequency": 1}]


class FreqNetDB(DBDataset):
    features = ["fft"]
    fft = None

    @staticmethod
    def extract(path, n_fft=2048, hop_length=512, sr=22050):
        import mimikit.data.transforms as T
        params = dict(n_fft=n_fft, hop_length=hop_length, sr=sr)
        fft = T.FileTo.mag_spec(path, **params)
        return dict(fft=(params, fft, None))

    def prepare_dataset(self, model):
        args = model.targets_shifts_and_lengths(model.hparams["input_seq_length"])
        self.slicer = ShiftedSequences(len(self.fft), [(0, model.hparams["input_seq_length"])] + args)

    def __getitem__(self, item):
        slices = self.slicer(item)
        return tuple(self.fft[sl] for sl in slices)

    def __len__(self):
        return len(self.slicer)


class FreqNetModel(MMKHooks,
                   LoggingHooks,
                   DataSubModule,
                   FreqOptim,
                   LightningModule,
                   ABC):
    """
    base class for all ``FreqNets`` that handles optim, data and a few handy methods like ``generate()``
    """

    db_class = FreqNetDB

    def __init__(self,
                 db=None,
                 files=None,
                 input_seq_length=64,
                 batch_size=64,
                 in_mem_data=True,
                 splits=[.8, .2],
                 max_lr=1e-3,
                 betas=(.9, .9),
                 div_factor=3.,
                 final_div_factor=1.,
                 pct_start=.25,
                 cycle_momentum=False,
                 **loaders_kwargs):
        # first : call super of LightningModule
        # Note: don't call super(FreqNetModel, self)! This way, we can explicitly specify how to init submodules
        super(LightningModule, self).__init__()

        # then : init submodules WITHOUT calling their super because
        # they are supposed to call their supers in their own init methods.
        FreqOptim.__init__(self, max_lr, betas, div_factor, final_div_factor, pct_start, cycle_momentum)
        DataSubModule.__init__(self, db, files, in_mem_data, splits, batch_size=batch_size, **loaders_kwargs)

        # finally : do what you need to!
        self.input_dim = self.db.fft.shape[-1]

        # calling this updates self.hparams from any subclass : call it when subclassing!
        self.save_hyperparameters()

    def training_step(self, batch, batch_idx):
        batch, target = batch
        output = self.forward(batch)
        recon = self.loss_fn(output, target)
        return {"loss": recon}

    def validation_step(self, batch, batch_idx):
        batch, target = batch
        output = self.forward(batch)
        recon = self.loss_fn(output, target)
        return {"val_loss": recon}

    def setup(self, stage: str):
        if stage == "fit":
            self.logger.log_hyperparams(self.hparams)
            self.max_epochs = self.trainer.max_epochs
            self.steps_per_epoch = len(self.datamodule.train_dataloader())

    # Convenience Methods for generative audio models (see also, `LoggingHooks.log_audio`):

    def generation_slices(self):
        raise NotImplementedError("subclasses of `FreqNetModel` have to implement `generation_slices`")

    def targets_shifts_and_lengths(self, input_length):
        raise NotImplementedError("subclasses of `FreqNetModel` have to implement `targets_shifts_and_lengths`")

    def generate(self, prompt, n_steps, hop_length=HOP_LENGTH, time_domain=True):
        was_training = self.training
        self.eval()
        initial_device = self.device
        self.to("cuda" if torch.cuda.is_available() else "cpu")
        if not isinstance(prompt, torch.Tensor):
            prompt = torch.from_numpy(prompt)
        if len(prompt.shape) < 3:
            prompt = prompt.unsqueeze(0)
        input_slice, output_slice = self.generation_slices()
        generated = prompt.to(self.device)
        for _ in tqdm(range(n_steps), desc="Generate", dynamic_ncols=True, leave=False, unit="step"):
            with torch.no_grad():
                out = self(generated[:, input_slice])
                generated = torch.cat((generated, out[:, output_slice]), dim=1)
        if time_domain:
            generated = generated.transpose(1, 2).squeeze()
            generated = librosa.griffinlim(generated.cpu().numpy(), hop_length=hop_length, n_iter=64)
            generated = torch.from_numpy(generated)
        else:  # for consistency with time_domain :
            generated = generated.cpu()
        self.to(initial_device)
        self.train() if was_training else None
        return generated.unsqueeze(0)
