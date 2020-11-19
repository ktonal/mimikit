import torch
from IPython import get_ipython
import numpy as np
import torch.nn as nn
import pytorch_lightning as pl
from time import time, gmtime
import os
import shutil
from ..data import load


DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("mmk initialized with device:", DEVICE)


def is_notebook():
    shell = get_ipython().__class__.__name__
    if shell == 'ZMQInteractiveShell':
        return True
    elif shell == 'TerminalInteractiveShell':
        return False
    else:
        return False


if is_notebook():
    from tqdm.notebook import tqdm
else:
    from tqdm import tqdm


def mean_L1_prop(output, target):
    L = nn.L1Loss(reduction="none")(output, target).sum(dim=(0, -1), keepdim=True)
    return 100 * (L / target.abs().sum(dim=(0, -1), keepdim=True)).mean()


def DefaultHP(**kwargs):
    defaults = dict(
        loss_fn=mean_L1_prop,
        # data
        database=None,
        train_set=None,
        input_dim=1025,
        # train stuff
        batch_size=64,
        batch_length=64,
        # optim / scheduler
        max_lr=1e-3,
        betas=(.9, .9),
        div_factor=3.,
        final_div_factor=1.,
        pct_start=.25,
        cycle_momentum=False,
        max_epochs=100,
        # Checkpoints
        root_dir="test_model/",
        name="model",
        version="v0",
        overwrite=False,
        era_duration=101,
    )
    defaults.update(kwargs)
    return defaults


class Model(pl.LightningModule):

    @property
    def path(self):
        return self.root_dir + self.name + (("/" + self.version) if self.version else "") + "/"

    def __init__(self, **kwargs):
        super(Model, self).__init__()
        self.hparams = DefaultHP(**kwargs)
        for k, v in self.hparams.items():
            setattr(self, k, v)

    @property
    def shift(self):
        raise NotImplementedError("base class `Model` doesn't implement the shift property")

    def prepare_data(self):
        # Default data routine
        db = self.database
        self.dl = load([db.fft] * 2, self.train_set.copy(), "frame",
                       k=self.batch_length, stride=1, shifts=(0, self.shift),
                       batch_size=self.batch_size, shuffle=True,
                       pre_cat=True, device=DEVICE)

    def train_dataloader(self):
        return self.dl

    def configure_optimizers(self):
        # for warm restarts
        if self.trainer is not None and self.trainer.optimizers is not None \
                and len(self.trainer.optimizers) >= 1:
            return self.trainer.optimizers
        self.opt = torch.optim.Adam(self.parameters(), lr=self.max_lr, betas=self.betas)
        self.sched = torch.optim.lr_scheduler.OneCycleLR(self.opt,
                                                         steps_per_epoch=len(self.dl),
                                                         epochs=self.max_epochs,
                                                         max_lr=self.max_lr, div_factor=self.div_factor,
                                                         final_div_factor=self.final_div_factor,
                                                         pct_start=self.pct_start,
                                                         cycle_momentum=self.cycle_momentum
                                                         )
        return [self.opt], [{"scheduler": self.sched, "interval": "step", "frequency": 1}]

    @staticmethod
    def load(clazz, version_dir, epoch=None):
        hp = torch.load(version_dir + "hparams.pt")
        hp["overwrite"] = False
        instance = clazz(**hp)
        sd = torch.load(version_dir + ("epoch=%i.ckpt" % epoch if epoch is not None else "final.ckpt"),
                        map_location=DEVICE)
        instance.load_state_dict(sd)
        return instance

    def restore_trainer(self):
        optimizers = sorted([file for file in os.listdir(self.path) if "optimizer" in file])
        optimizers = [torch.load(file) for file in optimizers]
        checkpoint = {
            "optimizer_states": [optimizers],
            "lr_schedulers": [],
            "global_step": 0,
            "epoch": 0
        }
        trainer = self.get_trainer()
        trainer.optimizers, trainer.lr_schedulers, _ = trainer.init_optimizers(self)
        trainer.restore_training_state(checkpoint)
        return trainer

    def on_train_start(self):
        self.init_directories()
        self._save_hp()
        self.ep_bar = tqdm(range(1, 1 + self.max_epochs), unit="epoch",
                           position=0, leave=False, dynamic_ncols=True)
        self.ep_losses = []
        self.losses = []
        self.start_time = time()

    def training_step(self, batch, batch_idx):
        batch, target = batch
        output = self.forward(batch)
        loss = self.loss_fn(output, target)
        self.ep_losses += [loss.item()]
        return {"loss": loss}

    def on_epoch_end(self):
        ep_loss = sum(self.ep_losses) / len(self.ep_losses)
        self.losses += [ep_loss]
        self.ep_losses = []
        self.ep_bar.update()
        if self.current_epoch % self.era_duration == 0:
            self._save_state("epoch=%i.ckpt" % self.current_epoch)
        print("Epoch: %i - Loss: %.4f" % (self.current_epoch, self.losses[-1]))

    def on_train_end(self):
        self._save_state("final.ckpt")
        self._save_loss()
        self._save_optimizers()
        total_time = gmtime(time() - self.start_time)
        print("Training finished after {0} days {1} hours {2} mins {3} seconds".format(total_time[2] - 1, total_time[3],
                                                                                       total_time[4], total_time[5]))

    def _save_loss(self):
        return np.save(self.path + "tr_losses", np.array(self.losses))

    def _save_hp(self):
        torch.save(self.hparams, self.path + "hparams.pt")

    def _save_state(self, filename="state.ckpt"):
        torch.save(self.state_dict(), self.path + filename)

    def _save_optimizers(self):
        for i, opt in enumerate(self.trainer.optimizers):
            torch.save(opt.state_dict(), self.path + "optimizer_" + str(i))

    def get_trainer(self, **kwargs):
        defaults = dict(gpus=1 if DEVICE.type == "cuda" else 0,
                        min_epochs=1,
                        max_epochs=self.hparams.get("max_epochs", 1024),
                        reload_dataloaders_every_epoch=False,
                        checkpoint_callback=False,
                        progress_bar_refresh_rate=5,
                        logger=False,
                        process_position=1,
                        )
        defaults.update(kwargs)
        return pl.Trainer(**defaults)

    def init_directories(self):
        if not os.path.isdir(self.root_dir):
            os.mkdir(self.root_dir)
        if not os.path.isdir(self.root_dir + self.name):
            os.mkdir(self.root_dir + self.name)
        if self.version != "" and self.version in os.listdir(self.root_dir + self.name) and not self.overwrite:
            while self.version in os.listdir(self.root_dir + self.name):
                self.version = "v" + str(int(self.version[1:]) + 1)
            os.mkdir(self.path)
        else:
            if os.path.isdir(self.path):
                if self.overwrite:
                    shutil.rmtree(self.path, ignore_errors=True)
                else:
                    raise ValueError("Model has no version string but overwrite=False")
            os.mkdir(self.path)
        print("initialized directory:", self.path)

