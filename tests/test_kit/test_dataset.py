import pytest
from ...kit.dataset import Dataset
import numpy as np
import torch


class Case:

    @property
    def ds(self):
        if self._ds is None:
            self._ds = Dataset(self.data_object)
        return self._ds

    def __init__(self, data_object, expected):
        self.data_object = data_object
        self.expected = expected
        self._ds = None

    def __repr__(self):
        return str(self.data_object)


valid_data_objects = [
    Case(np.random.randn(10, 10),
         dict(shape=((10, 10),),
              style=("map",),
              dtype=(torch.float64,),
              device=("cpu",),
              )),
    Case(torch.randn(10, 10, requires_grad=False),
         dict(shape=((10, 10),),
              style=("map",),
              dtype=(torch.float32,),
              device=("cpu",),
              )),
    Case((np.random.randn(10, 10), np.random.randn(10, 10)),
         dict(shape=((10, 10), (10, 10)),
              style=("map", "map"),
              dtype=(torch.float64, torch.float64),
              device=("cpu", "cpu"),
              )),
    Case([list(range(10)) for _ in range(10)],
         dict(shape=((10, 10),),
              style=("map",),
              dtype=(torch.int64,),
              device=("cpu",),
              )
         )
]


@pytest.fixture(params=valid_data_objects)
def init_valid_case(request):
    return request.param


def test_computed_properties_of_valid_cases(init_valid_case):
    case = init_valid_case
    to_check = case.expected.keys()
    for attr in to_check:
        computed = getattr(case.ds, attr)
        expected = case.expected[attr]
        assert computed == expected, (computed, expected)


def test_methods_on_valid_cases(init_valid_case):
    # test `to_tensor`

    # test `to(device)`

    # test `select`

    # test `split`

    # test `random_example`

    # test `load`

    pass


invalid_data_objects = [
    Case((np.random.randn(10, 10), range(10)),
         dict(shape=((10, 10),),
              style=("map",),
              dtype=(torch.float32,),
              device=("cpu",),
              )
         )
]


@pytest.fixture(params=invalid_data_objects)
def init_invalid_case(request):
    return request.param


def test_invalid_case_raises_correct_exception(init_invalid_case):
    pass