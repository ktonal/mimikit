import torch
from torch.utils.data import Dataset, random_split
from torch.utils.data._utils.collate import default_convert
import pytorch_lightning as pl

from ..h5data import Database


class DBDataset(Database, Dataset):
    """
    extends ``Database`` so that it can also be used in place of a ``Dataset``
    """

    def __init__(self, h5_file: str, keep_open: bool = False):
        """
        "first" init instantiates a ``Database``

        Parameters
        ----------
        h5_file : str
            path to the .h5 file containing the data
        keep_open : bool, optional
            whether to keep the h5 file open or close it after each query.
            Default is ``False``.
        """
        Database.__init__(self, h5_file, keep_open)

    def prepare_dataset(self, model: pl.LightningModule):
        """
        placeholder for implementing what need to be done before serving data.

        If this class was just a ``Dataset``, this would be its constructor.
        This method is best called in ``prepare_data()`` as shown in ``DBDataModule``.
        Passing the model as argument allows to use its hparams or any of its computed properties
        to "configure" ``self.__len__`` and ``self.__getitem__``.

        Parameters
        ----------
        model : pl.LightningModule
            the model that will consume this dataset.

        Returns
        -------
        None
        """
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError

    def __getitem__(self, item):
        raise NotImplementedError

    def split(self, splits):
        """
        Parameters
        ----------
        splits: Sequence of floats or ints possibly containing None.
            The sequence of elements corresponds to the proportion (floats), the number of examples (ints) or the absence of
            train-set, validation-set, test-set, other sets... in that order.

        Returns
        -------
        splits: tuple
            the *-sets
        """
        nones = []
        if any(x is None for x in splits):
            if splits[0] is None:
                raise ValueError("the train-set's split cannot be None")
            nones = [i for i, x in zip(range(len(splits)), splits) if x is None]
            splits = [x for x in splits if x is not None]
        if all(type(x) is float for x in splits):
            splits = [x / sum(splits) for x in splits]
            N = len(self)
            # leave the last one out for now because of rounding
            as_ints = [int(N * x) for x in splits[:-1]]
            # check that the last is not zero
            if N - sum(as_ints) == 0:
                raise ValueError("the last split rounded to zero element. Please provide a greater float or consider "
                                 "passing ints.")
            as_ints += [N - sum(as_ints)]
            splits = as_ints
        sets = list(random_split(self, splits))
        if any(nones):
            sets = [None if i in nones else sets.pop(0) for i in range(len(sets + nones))]
        return tuple(sets)

    @property
    def hparams(self):
        params = dict()
        for feat in self.features:
            this_feat = {feat + "_" + k: v for k, v in getattr(self, feat).attrs.items()}
            params.update(this_feat)
        return params

    # ***************************************************************************************************
    # ************  Convenience functions for converting features to (cuda) tensors  ********************

    def to_tensor(self):
        for feat in self.features:
            as_tensor = self._to_tensor(getattr(self, feat))
            setattr(self, feat, as_tensor)
        return self

    def to(self, device):
        for feat in self.features:
            self._to(getattr(self, feat), device)
        return self

    @staticmethod
    def _to_tensor(obj):
        if isinstance(obj, torch.Tensor):
            return obj
        # converting obj[:] makes sure we get the data out of any db.feature object
        maybe_tensor = default_convert(obj[:])
        if isinstance(maybe_tensor, torch.Tensor):
            return maybe_tensor
        try:
            obj = torch.tensor(obj)
        except Exception as e:
            raise e
        return obj

    @staticmethod
    def _to(obj, device):
        """move any underlying tensor to some device"""
        if getattr(obj, "to", False):
            return obj.to(device)
        raise TypeError("object %s has no `to()` attribute" % str(obj))
