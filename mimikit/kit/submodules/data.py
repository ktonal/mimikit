import torch
from torch.utils.data import DataLoader
from ..db_dataset import DBDataset
import pytorch_lightning as pl
from inspect import getfullargspec


class DBDataModule(pl.LightningDataModule):
    """
    boilerplate subclass of ``pytorch_lightning.LightningDataModule`` to handle standard "data-tasks" :
        - give a Database a chance to prepare itself for serving data once the model has been instantiated
        - move small datasets to the RAM of the gpu if desired
            (TODO: Otherwise, more workers are used in the DataLoaders for better performance)
        - split data into train, val and test sets and serve corresponding DataLoaders
    """
    def __init__(self,
                 model=None,
                 db: DBDataset = None,
                 in_mem_data=True,
                 splits=None,
                 **loader_kwargs,
                 ):
        super(DBDataModule, self).__init__()
        self.model = model
        self.db = db
        self.in_mem_data = in_mem_data
        self.splits = splits
        loader_kwargs.setdefault("drop_last", False)
        loader_kwargs.setdefault("shuffle", True)
        self.loader_kwargs = self._filter_loader_kwargs(loader_kwargs)
        self.train_ds, self.val_ds, self.test_ds = None, None, None

    def prepare_data(self, *args, **kwargs):
        self.db.prepare_dataset(self.model)

    def setup(self, stage=None):
        if stage == "fit":
            if self.in_mem_data and torch.cuda.is_available():
                self.db.to_tensor()
                self.db.to("cuda")
            if self.splits is None:
                self.splits = (1.,)
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
        has_val = len(self.splits) >= 2 and self.splits[1] is not None
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
        has_test = len(self.splits) >= 3 and self.splits[2] is not None
        if not has_test:
            return None
        if not self.has_prepared_data:
            self.prepare_data()
        if not self.has_setup_test:
            self.setup("test")
        kwargs = self.loader_kwargs.copy()
        kwargs["shuffle"] = shuffle
        return DataLoader(self.test_ds, **kwargs)

    @staticmethod
    def _filter_loader_kwargs(kwargs):
        valids = getfullargspec(DataLoader.__init__).annotations.keys()
        return {k: v for k, v in kwargs.items() if k in valids}


class DataSubModule(pl.LightningModule):

    db_class = None

    @property
    def db(self):
        """short-hand to quickly access the db passed to the constructor"""
        return self.datamodule.db

    def __init__(self,
                 db: [DBDataset, str] = None,
                 files: list = None,
                 in_mem_data: bool = True,
                 splits: list = [.8, .2],
                 **loaders_kwargs,
                 ):
        super(DataSubModule, self).__init__()
        if db is not None:
            if isinstance(db, str):
                db = self.db_class(db, keep_open=not in_mem_data)
            if files is not None:
                db = db.restrict_to_files(files)
            self.datamodule = DBDataModule(self, db, in_mem_data, splits, **loaders_kwargs)
            # data_params = dict(h5_file=db.h5_file, files=files, splits=splits, **loaders_kwargs, **db.hparams)
            self.save_hyperparameters()

    def random_train_example(self):
        return next(iter(self.datamodule.train_dataloader()))[0][0:1]

    def random_val_example(self):
        return next(iter(self.datamodule.val_dataloader(shuffle=True)))[0][0:1]