import pytest
import torch
from pyassert import assert_that
from typing import get_args

from mimikit.modules.inputer import FramedInput, ProjectionType

torch.set_grad_enabled(False)


@pytest.mark.parametrize(
    "given_type, expect_exception", [
        *[(arg, False) for arg in get_args(ProjectionType)],
        ("not-a-type", True), ("fiir", True), ("linea", True)
    ])
def test_should_accept_all_and_only_projection_type_args(given_type, expect_exception):
    if expect_exception:
        with pytest.raises(TypeError):
            mod = FramedInput(
                class_size=2,
                projection_type=given_type,
                hidden_dim=1,
                frame_size=1,
                unfold_step=None
            )
    else:
        mod = FramedInput(
            class_size=2,
            projection_type=given_type,
            hidden_dim=1,
            frame_size=1,
            unfold_step=None
        )
        assert_that(type(mod)).is_equal_to(FramedInput)
        assert_that(mod.projection_type).is_equal_to(given_type)


@pytest.mark.parametrize(
    "given_type", [
        "embedding", "fir_embedding"
    ])
def test_should_not_accept_unset_class_size(
        given_type: ProjectionType
):
    with pytest.raises(AssertionError):
        under_test = FramedInput(
            class_size=None,
            projection_type=given_type,
            hidden_dim=1,
            frame_size=1,
            unfold_step=None
        )


@pytest.mark.parametrize(
    "given_type", [
        "linear", "embedding", "fir", "fir_embedding"
    ])
def test_type_should_unfold_and_linearize_or_embed_and_project(
        given_type: ProjectionType
):
    given_class_size = 12
    given_hidden_dim = 16
    given_frame_size = 32
    given_batch_size = 2
    given_input_length = given_frame_size * 4
    given_input = torch.randint(
        0, given_class_size, (given_batch_size, given_input_length)
    )

    under_test = FramedInput(
        class_size=given_class_size,
        projection_type=given_type,
        hidden_dim=given_hidden_dim,
        frame_size=given_frame_size,
        unfold_step=given_frame_size
    )

    output = under_test(given_input)

    assert_that(output.shape).is_equal_to(
        (given_batch_size, given_input_length // given_frame_size, given_hidden_dim)
    )


@pytest.mark.parametrize(
    "given_type", [
        "linear", "fir"
    ])
def test_type_should_just_project(
        given_type: ProjectionType
):
    given_class_size = None
    given_hidden_dim = 16
    given_frame_size = 32
    given_batch_size = 2
    given_unfold_step = None
    given_input = torch.randn(given_batch_size, 10, given_frame_size)

    under_test = FramedInput(
        class_size=given_class_size,
        projection_type=given_type,
        hidden_dim=given_hidden_dim,
        frame_size=given_frame_size,
        unfold_step=given_unfold_step
    )

    output = under_test(given_input)

    assert_that(output.shape).is_equal_to(
        (given_batch_size, given_input.size(1), given_hidden_dim)
    )


@pytest.mark.parametrize(
    "given_type", [
        "linear", "fir"
    ])
def test_type_should_unfold_and_project(
        given_type: ProjectionType
):
    given_class_size = None
    given_hidden_dim = 16
    given_frame_size = 32
    given_batch_size = 2
    given_input_length = given_frame_size * 4
    given_input = torch.randn(given_batch_size, given_input_length)

    under_test = FramedInput(
        class_size=given_class_size,
        projection_type=given_type,
        hidden_dim=given_hidden_dim,
        frame_size=given_frame_size,
        unfold_step=given_frame_size
    )

    output = under_test(given_input)

    assert_that(output.shape).is_equal_to(
        (given_batch_size, 4, given_hidden_dim)
    )


@pytest.mark.parametrize(
    "given_type", [
        "embedding", "fir_embedding"
    ])
def test_type_should_embed_and_project(
        given_type: ProjectionType
):
    given_class_size = 8
    given_hidden_dim = 16
    given_frame_size = 32
    given_batch_size = 2
    given_unfold_step = None
    given_input = torch.randint(
        0, given_class_size, (given_batch_size, 4, given_frame_size)
    )

    under_test = FramedInput(
        class_size=given_class_size,
        projection_type=given_type,
        hidden_dim=given_hidden_dim,
        frame_size=given_frame_size,
        unfold_step=given_unfold_step
    )

    output = under_test(given_input)

    assert_that(output.shape).is_equal_to(
        (given_batch_size, given_input.size(1), given_hidden_dim)
    )


@pytest.mark.parametrize(
    "given_type", [
        "embedding", "fir_embedding"
    ])
def test_type_should_embed_and_unfold_and_project(
        given_type: ProjectionType
):
    given_class_size = 8
    given_hidden_dim = 16
    given_frame_size = 32
    given_batch_size = 2
    given_input_length = given_frame_size * 4
    given_input = torch.randint(
        0, given_class_size, (given_batch_size, given_input_length)
    )

    under_test = FramedInput(
        class_size=given_class_size,
        projection_type=given_type,
        hidden_dim=given_hidden_dim,
        frame_size=given_frame_size,
        unfold_step=given_frame_size
    )

    output = under_test(given_input)

    assert_that(output.shape).is_equal_to(
        (given_batch_size, 4, given_hidden_dim)
    )