from typing import Tuple

import pytest
from assertpy import assert_that

import torch
from torch.nn import Sigmoid

from mimikit.networks.wavenet_v2 import WNLayer


def inputs_(b=8, t=32, d=16):
    return torch.randn(b, d, t)


@pytest.mark.parametrize(
    "with_gate",
    [True, False]
)
@pytest.mark.parametrize(
    "feed_skips",
    [True, False]
)
@pytest.mark.parametrize(
    "given_input_dim",
    [None, 7]
)
@pytest.mark.parametrize(
    "given_pad",
    [0, 1]
)
@pytest.mark.parametrize(
    "given_residuals",
    [None,]  # TODO: broken if not None
)
@pytest.mark.parametrize(
    "given_skips",
    [None, 34]
)
@pytest.mark.parametrize(
    "given_1x1",
    [(), (3,), (8, 2,), (4, 9, 64)]
)
@pytest.mark.parametrize(
    "given_dil",
    [(16,), (32,), (8,)]
)
def test_layer_should_support_various_graphs(
    given_dil: Tuple[int], given_1x1: Tuple[int], given_skips, given_residuals,
        given_pad, given_input_dim, feed_skips, with_gate
):
    under_test = WNLayer(
        input_dim=given_input_dim,
        dims_dilated=given_dil,
        dims_1x1=given_1x1,
        skips_dim=given_skips,
        residuals_dim=given_residuals,
        pad_side=given_pad,
        act_g=Sigmoid() if with_gate else None
    )
    B, T = 1, 8
    input_dim = given_dil[0] if given_input_dim is None else given_input_dim
    skips = None if not feed_skips or given_skips is None else inputs_(B, T, given_skips)
    given_inputs = (
        (inputs_(B, T, input_dim), ), tuple(inputs_(B, T, d) for d in given_1x1), skips
    )
    expected_dim = given_residuals if given_residuals is not None else given_dil[0]

    outputs = under_test(*given_inputs)

    assert_that(type(outputs)).is_equal_to(tuple)
    assert_that(len(outputs)).is_equal_to(2)

    assert_that(outputs[0].size(1)).is_equal_to(expected_dim)
    if given_skips is not None:
        assert_that(outputs[1].size(1)).is_equal_to(given_skips)

    if bool(given_pad):
        assert_that(outputs[0].size(-1)).is_equal_to(T)
        if given_skips is not None:
            assert_that(outputs[1].size(-1)).is_equal_to(T)
            assert_that(outputs[1].size(-1)).is_equal_to(outputs[0].size(-1))
    else:
        assert_that(outputs[0].size(-1)).is_less_than(T)
        if given_skips is not None:
            assert_that(outputs[1].size(-1)).is_less_than(T)
            assert_that(outputs[1].size(-1)).is_equal_to(outputs[0].size(-1))
