import pytest

import mimikit as mmk
from assertpy import assert_that
import torch


@pytest.mark.parametrize(
    "given_kernel_sizes",
    [
        (3, 5, 7),
        (7, 5, 3),
        (3, 3, 3),
    ]
)
@pytest.mark.parametrize(
    "given_pad",
    [True, False]
)
def test_forward(given_pad, given_kernel_sizes):
    under_test = mmk.TiedAE.from_config(
        mmk.TiedAE.Config(
            io_spec=mmk.IOSpec.magspec_io(
                mmk.IOSpec.MagSpecIOConfig(),
            ),
            kernel_sizes=given_kernel_sizes,
            dims=(64, 32, 8),
            independence_reg=0.25,
            causal_pad=given_pad
        )
    )

    assert_that(under_test).is_instance_of(mmk.TiedAE)

    input = torch.randn(4, 8192)
    input = under_test.config.io_spec.inputs[0].transform(input)

    output = under_test.forward((input,))

    assert_that(output).is_instance_of(tuple)
    assert_that(output[0]).is_instance_of(torch.Tensor)
    assert_that(output[0].size()).is_equal_to(input.size())