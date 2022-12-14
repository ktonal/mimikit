import pytest
from assertpy import assert_that

from mimikit.features.ifeature import Batch
from mimikit.features.audio import MuLawSignal, AudioSignal, Spectrogram


def test_should_implement_config_api_correctly():
    inputs = [AudioSignal(), MuLawSignal()]
    targets = [Spectrogram(n_fft=1024), Spectrogram(n_fft=512)]

    under_test = Batch(inputs=inputs, targets=targets)

    serialized = under_test.serialize()
    loaded = Batch.deserialize(serialized)

    assert_that(type(loaded)).is_equal_to(Batch)
    assert_that(loaded).is_equal_to(under_test)


