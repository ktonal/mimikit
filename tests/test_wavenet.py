from typing import Tuple

import pytest
from assertpy import assert_that

import torch
from torch.nn import Sigmoid

from mimikit.checkpoint import Checkpoint
from mimikit.networks.wavenet_v2 import WNLayer, WaveNet


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
    [None, 5, 7]  # TODO: broken if not None
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
    # if given_residuals is not None and given_input_dim is not None
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
    # HOW INPUT_DIM WORKS:
    if given_input_dim is None:
        if given_residuals is None:
            input_dim = given_dil[0]
        else:
            input_dim = given_residuals
    else:
        input_dim = given_input_dim

    skips = None if not feed_skips or given_skips is None else inputs_(B, T, given_skips)

    given_inputs = (
        (inputs_(B, T, input_dim), ), tuple(inputs_(B, T, d) for d in given_1x1), skips
    )
    # HOW OUTPUT DIM WORKS:
    if given_residuals is not None:
        if given_input_dim is not None and given_input_dim != given_residuals:
            # RESIDUALS ARE SKIPPED!
            expected_out_dim = given_dil[0]
        else:
            expected_out_dim = given_residuals
    else:
        expected_out_dim = given_dil[0]

    outputs = under_test(*given_inputs)

    assert_that(type(outputs)).is_equal_to(tuple)
    assert_that(len(outputs)).is_equal_to(2)

    assert_that(outputs[0].size(1)).is_equal_to(expected_out_dim)
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


def test_should_instantiate_from_default_config():
    given_config = WaveNet.Config()

    under_test = WaveNet.from_config(given_config)

    assert_that(type(under_test)).is_equal_to(WaveNet)
    assert_that(len(under_test.layers)).is_equal_to(given_config.blocks[0])


def test_should_load_when_saved(tmp_path_factory):
    given_config = WaveNet.Config()
    root = str(tmp_path_factory.mktemp("ckpt"))
    wn = WaveNet.from_config(given_config)
    ckpt = Checkpoint(id="123", epoch=1, root_dir=root)

    ckpt.create(model_config=wn.config, network=wn)
    loaded = ckpt.network

    assert_that(type(loaded)).is_equal_to(WaveNet)


@pytest.mark.parametrize(
    "given_temp",
    [None, 0.5, (1.,)]
)
def test_generate(
        given_temp
):
    given_config = WaveNet.Config()
    q_levels = given_config.batch.inputs[0].q_levels
    wn = WaveNet.from_config(given_config)

    given_prompt = torch.randint(0, q_levels, (1, 128,))
    wn.eval()
    # For now, prompts are just Tuple[T] (--> Tuple[Tuple[T, ...]] for multi inputs!)
    wn.before_generate((given_prompt, ), batch_index=0)
    output = wn.generate_step(
        (given_prompt[:, -wn.rf:], ),
        t=given_prompt.size(1),
        temperature=given_temp)
    wn.after_generate(output, batch_index=0)

    assert_that(type(output)).is_equal_to(tuple)
    assert_that(output[0].size(0)).is_equal_to(given_prompt.size(0))
    assert_that(output[0].ndim).is_equal_to(given_prompt.ndim)