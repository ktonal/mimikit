import os
from typing import Tuple

import pytest
import torch
from assertpy import assert_that

from mimikit import GenerateLoopV2, TrainARMLoop, TrainARMConfig
from .test_utils import TestDB, tmp_db

from mimikit.networks.sample_rnn_v2 import SampleRNN
from mimikit.checkpoint import Checkpoint
from mimikit.io_spec import IOSpec


def test_should_instantiate_from_default_config():
    given_config = SampleRNN.Config(io_spec=IOSpec.mulaw_io(
        IOSpec.MuLawIOConfig()
    ))

    under_test = SampleRNN.from_config(given_config)

    assert_that(type(under_test)).is_equal_to(SampleRNN)
    assert_that(len(under_test.tiers)).is_equal_to(len(given_config.frame_sizes))


def test_should_take_n_unfolded_inputs():
    given_frame_sizes = (16, 4, 2,)
    given_config = SampleRNN.Config(
        frame_sizes=given_frame_sizes,
        io_spec=IOSpec.mulaw_io(
            IOSpec.MuLawIOConfig()
        ),
        inputs_mode='sum',
    )
    given_inputs = (torch.arange(128).reshape(2, 64),)
    # given_inputs[1] -= 64
    under_test = SampleRNN.from_config(given_config)
    outputs = under_test(given_inputs)

    assert_that(type(outputs)).is_equal_to(tuple)
    assert_that(outputs[0].shape).is_equal_to(
        (2, given_inputs[0].size(1) - given_frame_sizes[0],
         given_config.io_spec.inputs[0].elem_type.class_size)
    )


def test_should_load_when_saved(tmp_path_factory):
    given_config = SampleRNN.Config(io_spec=IOSpec.mulaw_io(
        IOSpec.MuLawIOConfig()
    ))
    root = str(tmp_path_factory.mktemp("ckpt"))
    srnn = SampleRNN.from_config(given_config)
    ckpt = Checkpoint(id="123", epoch=1, root_dir=root)

    ckpt.create(network=srnn)
    loaded = ckpt.network

    assert_that(type(loaded)).is_equal_to(SampleRNN)


@pytest.mark.parametrize(
    "given_temp",
    [None, 0.5, (1.,)]
)
def test_generate(
        given_temp
):
    given_config = SampleRNN.Config(io_spec=IOSpec.mulaw_io(
        IOSpec.MuLawIOConfig()
    ))
    q_levels = given_config.io_spec.inputs[0].elem_type.class_size
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


def test_generate_loop_integration(tmp_db):
    given_config = SampleRNN.Config(io_spec=IOSpec.mulaw_io(
        IOSpec.MuLawIOConfig()
    ))
    srnn = SampleRNN.from_config(given_config)

    db: TestDB = tmp_db("gen-test.h5")

    loop = GenerateLoopV2.from_config(
        GenerateLoopV2.Config(
            prompts_length_sec=512 / 16000,
            output_duration_sec=512 / 16000,
            prompts_position_sec=(None, None,),
            batch_size=2,
            parameters=dict(temperature=(1.,))
        ),
        db, srnn
    )

    for outputs in loop.run():
        assert_that(outputs).is_not_none()
        assert_that(outputs[0].shape).is_equal_to((2, 1024))
        assert_that(outputs[0].dtype).is_equal_to(torch.float)


def test_should_train(tmp_db, tmp_path):
    given_config = SampleRNN.Config(io_spec=IOSpec.mulaw_io(
        IOSpec.MuLawIOConfig()
    ), frame_sizes=(4, 2, 2))
    srnn = SampleRNN.from_config(given_config)
    db = tmp_db("train-loop.h5")
    config = TrainARMConfig(
        root_dir=str(tmp_path),
        limit_train_batches=2,
        batch_size=2,
        batch_length=8,
        tbptt_chunk_length=128,
        max_epochs=2,
        every_n_epochs=1,
        oversampling=4,
        CHECKPOINT_TRAINING=True,
        MONITOR_TRAINING=True,
        OUTPUT_TRAINING=True,
    )

    loop = TrainARMLoop.from_config(
        config, dataset=db, network=srnn
    )

    loop.run()

    content = os.listdir(os.path.join(str(tmp_path), loop.hash_))
    assert_that(content).contains("hp.yaml", "outputs", "epoch=1.h5")

    outputs = os.listdir(os.path.join(str(tmp_path), loop.hash_, "outputs"))
    assert_that([os.path.splitext(o)[-1] for o in outputs]).contains(".mp3")