import pytorch_lightning as pl
import torch
from pytorch_lightning import LightningModule
from torch.utils.data import DataLoader

from ..h5data.api import Database


class DataModule(pl.LightningDataModule):
    """
    boilerplate subclass of ``pytorch_lightning.LightningDataModule`` to handle standard "data-tasks" :
        - give a Database a chance to prepare itself for serving data once the model has been instantiated
        - move small datasets to the RAM of the gpu if desired
            (TODO: Otherwise, more workers are used in the DataLoaders for better performance)
        - split data into train, val and test sets and serve corresponding DataLoaders
    """
    def __init__(self,
                 model=None,
                 db: Database = None,
                 in_mem_data=True,
                 splits=None,
                 loader_kwargs={},
                 ):
        super(DataModule, self).__init__()
        self.model = model
        self.db = db
        self.in_mem_data = in_mem_data
        self.splits = splits
        self.loader_kwargs = loader_kwargs
        self.train_ds, self.val_ds, self.test_ds = None, None, None

    def prepare_data(self, *args, **kwargs):
        self.db.prepare_dataset(model=self.model, loader_kwargs=self.loader_kwargs)

    def setup(self, stage=None):
        if stage == "fit":
            if self.in_mem_data and torch.cuda.is_available():
                self.db.to_tensor()
                self.db.to("cuda")
            if not self.splits:
                sets = (self.db, )
            else:
                sets = self.db.split(self.splits)
            for ds, attr in zip(sets, ["train_ds", "val_ds", "test_ds"]):
                setattr(self, attr, ds)

    def train_dataloader(self):
        if not self.has_prepared_data:
            self.prepare_data()
        if not self.has_setup_fit:
            self.setup("fit")
        return DataLoader(self.train_ds, **self.loader_kwargs)

    def val_dataloader(self, shuffle=False):
        has_val = self.splits is not None and len(self.splits) >= 2 and self.splits[1] is not None
        if not has_val:
            return None
        if not self.has_prepared_data:
            self.prepare_data()
        if not self.has_setup_fit:
            self.setup("fit")
        kwargs = self.loader_kwargs.copy()
        kwargs["shuffle"] = shuffle
        return DataLoader(self.val_ds, **kwargs)

    def test_dataloader(self, shuffle=False):
        has_test = self.splits is not None and len(self.splits) >= 3 and self.splits[2] is not None
        if not has_test:
            return None
        if not self.has_prepared_data:
            self.prepare_data()
        if not self.has_setup_test:
            self.setup("test")
        kwargs = self.loader_kwargs.copy()
        kwargs["shuffle"] = shuffle
        return DataLoader(self.test_ds, **kwargs)


class DataPart(LightningModule):

    db_class = None

    def __init__(self,
                 db: [Database, str] = None,
                 in_mem_data: bool = True,
                 splits: [list, None] = [.8, .2],
                 keep_open=False,
                 loaders_kwargs={},
                 ):
        super(LightningModule, self).__init__()
        if db is not None:
            db_ = db
            if isinstance(db, str):
                db_ = self.db_class(db_, keep_open=keep_open)
            self.datamodule = DataModule(self, db_, in_mem_data, splits, loaders_kwargs)
            # cache the db params in self.hparams
            for feat_name, params in db_.params.__dict__.items():
                for p, val in params.items():
                    setattr(self.hparams, p, val)
            self.db = db_

    def _set_hparams(self, hp):
        # break the link in the hparams so as to not include data in checkpoints :
        if "db" in hp and hasattr(hp['db'], 'h5_file'):
            hp["db"] = f"<{str(self.db_class.__name__)} : {hp['db'].h5_file}>"
        super(DataPart, self)._set_hparams(hp)
