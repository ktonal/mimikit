import pytest
import mimikit as mmk
import torch
from assertpy import assert_that


@pytest.mark.parametrize(
    "in_dim",
    [256, 53, 12, 128],
)
@pytest.mark.parametrize(
    "hidden_dim",
    [256, 53, 12, 128],
)
@pytest.mark.parametrize(
    "out_dim",
    [128, 53, 13, 63]
)
def test_vector_mix(in_dim, hidden_dim, out_dim):
    under_test = mmk.VectorMixIO(hidden_dim=hidden_dim)
    under_test = under_test.set(in_dim=in_dim, out_dim=out_dim).module()
    input = torch.randn(4, 61, in_dim)

    output = under_test(input)

    assert_that(output.size(0)).is_equal_to(input.size(0))
    assert_that(output.size(1)).is_equal_to(input.size(1))
    assert_that(output.size(2)).is_equal_to(out_dim)
