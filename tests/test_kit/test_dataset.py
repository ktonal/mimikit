import pytest
from mmk.kit.dataset import Dataset
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
        assert computed == expected, (attr, computed, expected)


def test_to_tensor(init_valid_case):
    pass


def test_to_device():
    pass


def test_select():
    pass


def test_split():
    pass


def test_to_datamodule():
    pass


invalid_data_objects = [
    Case((np.random.randn(10, 10), (x for x in range(10))),
         dict(style=TypeError, msg="same style")
         ),
    Case(np.random.randn,
         dict(style=ValueError, msg="implement")
         ),
    Case((np.random.randn(12, 10), np.random.randn(10, 10)),
         dict(style=ValueError, msg="same lengths")
         ),
    Case(["this string's dtype isn't recognized"],
         dict(dtype=TypeError, msg="numerical")
         )
]


@pytest.fixture(params=invalid_data_objects)
def init_invalid_case(request):
    return request.param


def test_constructor_exceptions(init_invalid_case):
    case = init_invalid_case
    exception = [e for key, e in case.expected.items() if isinstance(e, type)][0]
    msg = case.expected["msg"]
    with pytest.raises(exception, match=r".* " + msg + r".*"):
        assert case.ds is not None
