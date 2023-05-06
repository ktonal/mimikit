import numpy as np
from assertpy import assert_that
import matplotlib.pyplot as plt
import pytest

import mimikit as mmk


def test_should_fail_with_magspec():
    n_fft, hop_length, center = 2048, 512, True
    alignment = "end"
    fft = mmk.MagSpec(n_fft, hop_length, center=center, alignment=alignment)
    ifft = fft.inv

    n_frames = 8
    extra = 104
    x = mmk.Normalize()(np.random.randn((n_frames-1)*hop_length+extra))

    S = fft(x)

    assert_that(S.shape[0]).is_equal_to(n_frames)

    y = ifft(S)
    with pytest.raises(AssertionError):
        assert_that(np.allclose(x[-y.shape[0]:], y)).is_true()


def test_should_end_align_with_center_true():
    n_fft, hop_length, center = 2048, 512, True
    alignment = "end"
    fft = mmk.STFT(n_fft, hop_length, center=center, alignment=alignment)
    ifft = fft.inv

    n_frames = 8
    extra = 104
    x = mmk.Normalize()(np.random.randn((n_frames-1)*hop_length+extra))

    S = fft(x)

    assert_that(S.shape[0]).is_equal_to(n_frames)

    y = ifft(S)

    assert_that(np.allclose(x[-y.shape[0]:], y)).is_true()


def test_should_end_align_with_center_false():
    n_fft, hop_length, center = 2048, 512, False
    alignment = "end"
    fft = mmk.STFT(n_fft, hop_length, center=center, alignment=alignment,
                   window="hann"  # not None values break the test for the 1st sample (!...)
                   )
    ifft = fft.inv
    n_frames = 8
    extra = 105
    x = mmk.Normalize()(np.random.randn((n_fft-hop_length)+(n_frames*hop_length)+extra))

    S = fft(x)

    assert_that(S.shape[0]).is_equal_to(n_frames)

    y = ifft(S)

    # plt.figure(figsize=(40, 6))
    # plt.plot(x[-y.shape[0]+1:][:300])
    # plt.plot(y[1:300])
    # plt.show()

    # not None window break the test for the 1st sample (!...)
    assert_that(np.allclose(x[-y.shape[0]+1:], y[1:])).is_true()


def test_should_start_align_with_center_true():
    n_fft, hop_length, center = 2048, 512, True
    alignment = "start"
    fft = mmk.STFT(n_fft, hop_length, center=center, alignment=alignment)
    ifft = fft.inv

    n_frames = 8
    extra = 87
    x = mmk.Normalize()(np.random.randn((n_frames-1)*hop_length+extra))

    S = fft(x)

    assert_that(S.shape[0]).is_equal_to(n_frames)

    y = ifft(S)

    assert_that(np.allclose(x[:y.shape[0]], y)).is_true()


def test_should_start_align_with_center_false():
    n_fft, hop_length, center = 2048, 512, False
    alignment = "start"
    fft = mmk.STFT(n_fft, hop_length, center=center, alignment=alignment,
                   window="hann"  # not None values break the test for the 1st sample (!...)
                   )
    ifft = fft.inv
    n_frames = 8
    extra = 99
    x = mmk.Normalize()(np.random.randn((n_fft-hop_length)+(n_frames*hop_length)+extra))

    S = fft(x)

    assert_that(S.shape[0]).is_equal_to(n_frames)

    y = ifft(S)

    # plt.figure(figsize=(40, 6))
    # plt.plot(x[:y.shape[0]][:300])
    # plt.plot(y[:300])
    # plt.show()

    # not None window break the test for the 1st sample (!...)
    assert_that(np.allclose(x[1:y.shape[0]], y[1:])).is_true()