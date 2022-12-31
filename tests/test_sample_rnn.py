from typing import Tuple

import pytest
import torch
from assertpy import assert_that

from mimikit import MuLawSignal, IOFactory, MLPParams, LinearParams
from mimikit.networks.io_spec import *
from mimikit.networks.sample_rnn_v2 import SampleRNN
from mimikit.checkpoint import Checkpoint


def test_should_instantiate_from_default_config():
    given_config = SampleRNN.Config(io_spec=SampleRNN.qx_io)

    under_test = SampleRNN.from_config(given_config)

    assert_that(type(under_test)).is_equal_to(SampleRNN)
    assert_that(len(under_test.tiers)).is_equal_to(len(given_config.frame_sizes))


# @pytest.mark.skip("Not really supported yet...")
def test_should_take_n_unfolded_inputs():
    given_frame_sizes = (16, 4, 2,)
    given_config = SampleRNN.Config(
        frame_sizes=given_frame_sizes,
        io_spec=SampleRNN.qx_io,
        inputs_mode='sum',
    )
    given_inputs = (torch.arange(128).reshape(2, 64), )
    # given_inputs[1] -= 64
    under_test = SampleRNN.from_config(given_config)
    outputs = under_test(given_inputs)

    assert_that(type(outputs)).is_equal_to(tuple)
    assert_that(outputs[0].shape).is_equal_to(
        (2, given_inputs[0].size(1)-given_frame_sizes[0], given_config.io_spec.inputs[0].feature.q_levels)
    )


def test_should_load_when_saved(tmp_path_factory):
    given_config = SampleRNN.Config(io_spec=SampleRNN.qx_io)
    root = str(tmp_path_factory.mktemp("ckpt"))
    srnn = SampleRNN.from_config(given_config)
    ckpt = Checkpoint(id="123", epoch=1, root_dir=root)

    ckpt.create(model_config=srnn.config, network=srnn)
    loaded = ckpt.network

    assert_that(type(loaded)).is_equal_to(SampleRNN)


@pytest.mark.parametrize(
    "given_temp",
    [None, 0.5, (1.,)]
)
def test_generate(
        given_temp
):
    given_config = SampleRNN.Config(io_spec=SampleRNN.qx_io)
    q_levels = given_config.io_spec.inputs[0].feature.q_levels
    srnn = SampleRNN.from_config(given_config)

    given_prompt = (torch.randint(0, q_levels, (1, 32,)),)
    srnn.eval()
    # For now, prompts are just Tuple[T] (--> Tuple[Tuple[T, ...]] for multi inputs!)
    srnn.before_generate(given_prompt, batch_index=0)
    output = srnn.generate_step(
        tuple(p[:, -srnn.rf:] for p in given_prompt),
        t=given_prompt[0].size(1),
        temperature=given_temp)
    srnn.after_generate(output, batch_index=0)

    assert_that(type(output)).is_equal_to(tuple)
    assert_that(output[0].size(0)).is_equal_to(given_prompt[0].size(0))
    assert_that(output[0].ndim).is_equal_to(given_prompt[0].ndim)
