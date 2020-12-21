import torch
from pytorch_lightning import LightningModule, LightningDataModule
from torch.utils.data import DataLoader
import librosa
from abc import ABC

from ..data import DataObject, HOP_LENGTH
from ..kit import ShiftedSeqsPair, MMKHooks, LoggingHooks


class FreqOptim:

    def __init__(self,
                 model,
                 max_lr=5e-4,
                 betas=(.9, .9),
                 div_factor=3.,
                 final_div_factor=1.,
                 pct_start=.25,
                 cycle_momentum=False):
        self.model = model
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
        self.opt = torch.optim.Adam(self.model.parameters(), lr=self.max_lr, betas=self.betas)
        self.sched = torch.optim.lr_scheduler.OneCycleLR(self.opt,
                                                         steps_per_epoch=self.steps_per_epoch,
                                                         epochs=self.max_epochs,
                                                         max_lr=self.max_lr, div_factor=self.div_factor,
                                                         final_div_factor=self.final_div_factor,
                                                         pct_start=self.pct_start,
                                                         cycle_momentum=self.cycle_momentum
                                                         )
        return [self.opt], [{"scheduler": self.sched, "interval": "step", "frequency": 1}]


class FreqData(LightningDataModule):
    def __init__(self,
                 model,
                 data_object=None,
                 input_seq_length=64,
                 batch_size=64,
                 to_gpu=True,
                 splits=None,
                 **loader_kwargs,
                 ):
        super(FreqData, self).__init__()
        self.model = model
        self.ds = DataObject(data_object)
        self.input_seq_length = input_seq_length
        self.batch_size = batch_size
        self.to_gpu = to_gpu
        self.splits = splits
        loader_kwargs.setdefault("drop_last", False)
        loader_kwargs.setdefault("shuffle", True)
        self.loader_kwargs = loader_kwargs
        self.train_ds, self.val_ds, self.test_ds = None, None, None

    def prepare_data(self, *args, **kwargs):
        if not (getattr(self.model, "targets_shifts_and_lengths", False)):
            raise TypeError("Expected `model` to implement `targets_shifts_and_lengths(input_length)`"
                            " in order to compute the right slices for the batches")
        targets_def = self.model.targets_shifts_and_lengths(self.input_seq_length)
        wrapper = ShiftedSeqsPair(self.input_seq_length, targets_def)
        self.ds = wrapper(self.ds)

    def setup(self, stage=None):
        if stage == "fit":
            if self.to_gpu and torch.cuda.is_available():
                self.ds.to_tensor()
                self.ds.to("cuda")
            if self.splits is None:
                self.splits = (1.,)
            sets = self.ds.split(self.splits)
            for ds, attr in zip(sets, ["train_ds", "val_ds", "test_ds"]):
                setattr(self, attr, ds)

    def train_dataloader(self):
        if self.train_ds is not None:
            return DataLoader(self.train_ds, batch_size=self.batch_size, **self.loader_kwargs)

    def val_dataloader(self):
        if self.val_ds is not None:
            kwargs = self.loader_kwargs.copy()
            kwargs["shuffle"] = False
            return DataLoader(self.val_ds, batch_size=self.batch_size, **kwargs)

    def test_dataloader(self):
        if self.test_ds is not None:
            kwargs = self.loader_kwargs.copy()
            kwargs["shuffle"] = False
            return DataLoader(self.test_ds, batch_size=self.batch_size, **kwargs)


class FreqNetModel(MMKHooks,
                   LoggingHooks,
                   LightningModule,
                   ABC):

    def __init__(self,
                 data_object=None,
                 input_seq_length=64,
                 batch_size=64,
                 to_gpu=True,
                 splits=[.8, .2],
                 max_lr=5e-4,
                 betas=(.9, .9),
                 div_factor=3.,
                 final_div_factor=1.,
                 pct_start=.25,
                 cycle_momentum=False,
                 **loaders_kwargs):
        super(FreqNetModel, self).__init__()
        # dimensionality of inputs is automatically available
        self.input_dim = data_object.shape[-1]
        self.datamodule = FreqData(self, data_object, input_seq_length, batch_size,
                                   to_gpu, splits, **loaders_kwargs) if data_object is not None else None
        self.optim = FreqOptim(self, max_lr, betas, div_factor, final_div_factor, pct_start,
                               cycle_momentum)
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
            self.optim.max_epochs = self.trainer.max_epochs
            self.optim.steps_per_epoch = len(self.datamodule.train_dataloader())

    def configure_optimizers(self):
        return self.optim.configure_optimizers()

    def _set_hparams(self, hp):
        # remove any inputs passed to the hparams...
        if "data_object" in hp:
            hp.pop("data_object")
        super(FreqNetModel, self)._set_hparams(hp)

    # Convenience Methods for generative audio models (see also, `LoggingHooks.log_audio`):

    def random_train_example(self):
        return next(iter(self.datamodule.train_dataloader()))[0][0:1]

    def first_val_example(self):
        return next(iter(self.datamodule.val_dataloader()))[0][0:1]

    def generation_slices(self):
        raise NotImplementedError("subclasses of `FreqNetModel` have to implement `generation_slices`")

    def targets_shifts_and_lengths(self, input_length):
        raise NotImplementedError("subclasses of `FreqNetModel` have to implement `targets_shifts_and_lengths`")

    def generate(self, input, n_steps, hop_length=HOP_LENGTH):
        if not isinstance(input, torch.Tensor):
            input = torch.from_numpy(input)
        if len(input.shape) < 3:
            input = input.unsqueeze(0)
        input_slice, output_slice = self.generation_slices()
        generated = input.to(self.device)
        for _ in range(n_steps):
            with torch.no_grad():
                out = self(generated[:, input_slice])
                generated = torch.cat((generated, out[:, output_slice]), dim=1)
        generated = generated.transpose(1, 2).squeeze()
        generated = librosa.griffinlim(generated.numpy(), hop_length=hop_length, n_iter=64)
        return torch.from_numpy(generated).unsqueeze(0)
