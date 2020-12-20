import torch
from pytorch_lightning import LightningModule, LightningDataModule
from torch.utils.data import DataLoader

from ..kit import Dataset, ShiftedSeqsPair, MMKHooks, EpochEndPrintHook


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
                 inputs=None,
                 input_seq_length=64,
                 batch_size=64,
                 to_gpu=True,
                 splits=None,
                 **loader_kwargs,
                 ):
        super(FreqData, self).__init__()
        self.model = model  # ! model must implement `targets_shifts_and_lengths` !
        self.ds = Dataset(inputs)
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
                   EpochEndPrintHook,
                   LightningModule):

    def __init__(self,
                 inputs=None,
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
        self.datamodule = FreqData(self, inputs, input_seq_length, batch_size,
                                   to_gpu, splits, **loaders_kwargs) if inputs is not None else None

        self.optim = FreqOptim(self, max_lr, betas, div_factor, final_div_factor, pct_start,
                               cycle_momentum)

    def training_step(self, batch, batch_idx):
        batch, target = batch
        output = self.forward(batch)
        recon = self.loss_fn(output, target)
        self.log("train_recon", recon, on_step=False, on_epoch=True)
        return {"loss": recon}

    def validation_step(self, batch, batch_idx):
        batch, target = batch
        output = self.forward(batch)
        recon = self.loss_fn(output, target)
        self.log("val_recon", recon, on_step=False, on_epoch=True)
        return {"val_loss": recon}

    def setup(self, stage: str):
        if stage == "fit":
            self.optim.max_epochs = self.trainer.max_epochs
            self.optim.steps_per_epoch = len(self.datamodule.train_dataloader())

    def configure_optimizers(self):
        return self.optim.configure_optimizers()
