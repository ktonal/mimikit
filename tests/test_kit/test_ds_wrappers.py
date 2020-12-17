import numpy as np

from ...mmk.kit import Dataset, DSWrapper, InputEqualTarget, ShiftedSeqsPair


class TestBaseWrapper:
    data = np.random.randn(10, 10)
    ds = Dataset(data)
    wrapper = DSWrapper()

    def test_wraps_without_error(self):
        ds = self.wrapper(self.ds)
        assert ds.data is self.data

    def test_wraps_returns_new_copy(self):
        ds = self.wrapper(self.ds)
        assert self.ds.__class__ is not ds.__class__


class TestInputEqualTarget:
    data = np.random.randn(10, 10)
    ds = Dataset(data)
    wrapper = InputEqualTarget()

    def test_wraps_without_error(self):
        ds = self.wrapper(self.ds)
        assert ds.data is self.data

    def test_overrides_getitem(self):
        ds = self.wrapper(self.ds)
        returned = ds[0]
        assert len(returned) == 2, returned
        assert np.all(returned[0] == returned[1])


class TestShiftedSeqsPair:
    data = np.random.randn(10, 4)
    ds = Dataset(data)
    wrapper = ShiftedSeqsPair(sequence_length=(2, 3), shift=1)

    def test_wraps_without_error(self):
        ds = self.wrapper(self.ds)
        assert ds.data is self.data

    def test_overrides_getitem(self):
        ds = self.wrapper(self.ds)
        returned = ds[0]
        # returns input and target
        assert len(returned) == 2, returned
        # returns sequences
        assert returned[0].shape == (self.wrapper.sequence_length[0], self.data.shape[-1]), returned[0].shape
        # shifts sequences correctly
        assert np.all(returned[0][self.wrapper.shift] == returned[1][0]), (returned[0][self.wrapper.shift],
                                                                           returned[1][0])

    def test_overrides_len(self):
        ds = self.wrapper(self.ds)
        assert len(ds) < len(self.ds), (len(ds), len(self.ds))
