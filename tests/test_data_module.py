import unittest

from mmk.kit.ds_wrappers import InputEqualTargetWrapper
from mmk.kit.datamodule import *
from ..data.factory import file_to_db, Database
import os
from sklearn.model_selection import ParameterGrid

# MAKE SOME DATASETS :

shape = (7, 2)
np_feat = np.random.randint(0, 8, shape)
torch_feat = torch.from_numpy(np_feat)
list_feat = list(np_feat[:, 0])


class TestDataset(Dataset):

    def __getitem__(self, item):
        return torch.from_numpy(np_feat[item])

    def __len__(self):
        return len(np_feat)


ds_feat = TestDataset()

test_db_path = "./mmk/tests/test_db.wav"


def extract_func(path):
    return {"feat": ({"t_axis": 0}, np_feat),
            "metadata": ({}, Metadata.from_duration([2] * 3 + [1]))}


test_db_path = file_to_db(test_db_path, extract_func)
db = Database(test_db_path)


def generator():
    for x in range(9):
        yield torch.tensor([x] * 4)


class TestIterableDataset(IterableDataset):
    def __iter__(self):
        return iter(generator())


gen_feat = generator()
iter_ds = TestIterableDataset()

subset_idx = np.sort(np.random.choice(list(range(7)), 5, replace=False))

# Brute-force coverage! >:D

CASES = ParameterGrid(
    {
        "feature": [
            np_feat,
            torch_feat,
            ds_feat,
            db.feat,
            gen_feat,
            iter_ds,
            (np_feat, db.feat),  # valid
            (np_feat, db.feat, list_feat),  # valid
            (np_feat, gen_feat),  # NOT valid
            (gen_feat, iter_ds),  # valid
            (iter_ds, ds_feat),  # NOT valid
        ],

        "subset_idx": [None,
                       subset_idx,
                       list(subset_idx),
                       db.metadata,
                       db.metadata.iloc[[1, 2, 3]]],

        "malloc_device": [None, "cpu"],

        "ds_wrapper": [
            None,
            InputEqualTargetWrapper()
        ],

        "train_val_split": [1., .5],

        "loader_kwargs": [
            {
                "batch_size": 3,
                "num_workers": 0,
            },
            {
                "batch_size": None,
                "num_workers": 0,

            },
            {
                "collate_fn": None,
            },
        ]
    }, )


class TestDataModule(unittest.TestCase):

    def test_data_module_setup(self):

        for i, case in enumerate(CASES):

            with self.subTest(stage="setup", case=case):
                kwargs = case["loader_kwargs"]
                case.pop("loader_kwargs")
                feat = case["feature"]
                dm = DataModuleBase(**case, **kwargs)

                # catch invalid device-allocations of any type of datasets and generator
                if case["malloc_device"] is not None and \
                        (isinstance(feat, TestDataset) or
                         isinstance(feat, TestIterableDataset) or
                         isinstance(feat, Generator) or
                         (isinstance(feat, tuple) and
                          (any(x is ds_feat for x in feat) or any(x is iter_ds for x in feat)))):
                    with self.assertRaises(ValueError):
                        dm.setup()
                    continue
                # catch invalid (subset & iterable/generator) cases
                elif case["subset_idx"] is not None and \
                        (isinstance(feat, TestIterableDataset) or
                         isinstance(feat, Generator) or
                         (isinstance(feat, tuple) and
                          (any(x is gen_feat for x in feat) or any(x is iter_ds for x in feat)))):
                    with self.assertRaises(ValueError):
                        dm.setup()
                    continue
                # catch features mixing styles
                elif isinstance(feat, tuple) and \
                        ((any(x is ds_feat for x in feat) and any(x is iter_ds for x in feat)) or
                         (any(x is gen_feat for x in feat) and any(x is np_feat for x in feat))):
                    with self.assertRaises(ValueError):
                        dm.setup()
                    continue
                # catch invalid wrapping of iterables in a map-style
                elif case["ds_wrapper"] is not None and \
                        (isinstance(feat, TestIterableDataset) or
                         isinstance(feat, Generator) or
                         isinstance(feat, tuple)):
                    with self.assertRaises(ValueError):
                        dm.setup()
                    continue
                else:
                    dm.setup()

            with self.subTest(stage="loaders", case=case, kwargs=kwargs):

                tr_loader = dm.train_dataloader()

            with self.subTest(stage="batch", case=case, kwargs=kwargs):

                for _, batch in zip(range(2), iter(tr_loader)):

                    if isinstance(feat, tuple):

                        self.assertEqual(len(batch), len(feat), str([batch, feat]))
                        self.assertTrue(all(isinstance(x, torch.Tensor) for x in batch))

                    else:
                        self.assertTrue(all(isinstance(x, torch.Tensor) for x in batch),
                                        "Got :" + str(batch))

    def tearDown(self):
        os.remove(test_db_path)


if __name__ == '__main__':
    unittest.main()
