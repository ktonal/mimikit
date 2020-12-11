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

    def __init__(self, data_object):
        self.data_object = data_object
        self._ds = None

    def should_be_multi(self):
        return isinstance(self.data_object, tuple) and len(self.data_object) > 1

    def assert_ds_is_not_None(self):
        assert self.ds is not None

    def assert_attr_is_multi(self, attr):
        ds = self.ds
        result = getattr(ds, attr)
        assert isinstance(result, tuple) and len(result) == len(self.data_object), (result, len(self.data_object))

    def assert_attr_is_not_None(self, attr):
        ds = self.ds
        result = getattr(ds, attr)
        assert result is not None


valid_data_objects = [
    np.random.randn(10, 10),
    torch.randn(10, 10),
    (np.random.randn(10, 10), np.random.randn(10, 10)),
    [list(range(10)) for _ in range(10)],
]


@pytest.fixture(params=valid_data_objects)
def init_valid_case(request):
    return Case(request.param)


def test_computed_properties_of_valid_cases(init_valid_case):
    case = init_valid_case
    attrs = ["shape", "dtype", "device", "style"]
    case.assert_ds_is_not_None()
    if not case.should_be_multi():
        for attr in attrs:
            case.assert_attr_is_not_None(attr)
    else:
        for attr in attrs:
            case.assert_attr_is_multi(attr)


def test_methods_on_valid_cases(init_valid_case):
    # test `to_tensor`

    # test `to(device)`

    # test `select`

    # test `split`

    # test `random_example`

    # test `load`

    pass


invalid_data_objects = [
    (np.random.randn(10, 10), range(10)),
]


@pytest.fixture(params=invalid_data_objects)
def init_invalid_case(request):
    return Case(request.param)


def test_invalid_case_raises_correct_exception(init_invalid_case):
    pass