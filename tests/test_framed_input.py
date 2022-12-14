import pytest
import torch
from assertpy import assert_that
from typing import get_args

from mimikit.modules.io import ModuleFactory

torch.set_grad_enabled(False)


@pytest.mark.parametrize(
    "given_type, expect_exception", [
        *[(arg, False) for arg in get_args(ProjectionType)],
        ("not-a-type", True), ("fiir", True), ("linea", True)
    ])
def test_should_accept_all_and_only_projection_type_args(given_type, expect_exception):
    if expect_exception:
        with pytest.raises(TypeError):
            mod = ModuleFactory(
                class_size=2,
                module_type=given_type,
                hidden_dim=1,
                frame_size=1,
                unfold_step=None
            )
    else:
        mod = ModuleFactory(
            class_size=2,
            module_type=given_type,
            hidden_dim=1,
            frame_size=1,
            unfold_step=None
        )
        assert_that(type(mod)).is_equal_to(ModuleFactory)
        assert_that(mod.module_type).is_equal_to(given_type)


@pytest.mark.parametrize(
    "given_type", [
        "embedding_bag", "embedding_conv1d"
    ])
def test_should_not_accept_unset_class_size(
        given_type: ProjectionType
):
    with pytest.raises(AssertionError):
        under_test = ModuleFactory(
            class_size=None,
            module_type=given_type,
            hidden_dim=1,
            frame_size=1,
            unfold_step=None
        )


@pytest.mark.parametrize(
    "given_type", [
        "framed_linear",
    ])
def test_should_project_and_not_unfold(given_type):
    given_class_size = None
    given_frame_size = 32
    given_hidden_dim = 16
    given_batch_size = 2

    given_input = torch.randn(
        given_batch_size, given_frame_size
    )

    under_test = ModuleFactory(
        class_size=given_class_size,
        module_type=given_type,
        hidden_dim=given_hidden_dim,
        frame_size=given_frame_size,
        unfold_step=None
    )

    output = under_test(given_input)

    assert_that(output.shape).is_equal_to(
        (given_batch_size, given_hidden_dim)
    )


@pytest.mark.parametrize(
    "given_type", [
        "framed_linear", "embedding_bag", "framed_conv1d", "embedding_conv1d"
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

    under_test = ModuleFactory(
        class_size=given_class_size,
        module_type=given_type,
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
        "framed_linear", "framed_conv1d"
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

    under_test = ModuleFactory(
        class_size=given_class_size,
        module_type=given_type,
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
        "framed_linear", "framed_conv1d"
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

    under_test = ModuleFactory(
        class_size=given_class_size,
        module_type=given_type,
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
        "embedding_bag", "embedding_conv1d"
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

    under_test = ModuleFactory(
        class_size=given_class_size,
        module_type=given_type,
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
        "embedding_bag", "embedding_conv1d"
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

    under_test = ModuleFactory(
        class_size=given_class_size,
        module_type=given_type,
        hidden_dim=given_hidden_dim,
        frame_size=given_frame_size,
        unfold_step=given_frame_size
    )

    output = under_test(given_input)

    assert_that(output.shape).is_equal_to(
        (given_batch_size, 4, given_hidden_dim)
    )