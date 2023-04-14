import os

import pytest
from assertpy import assert_that

import torch

from mimikit import IOSpec, TrainARMConfig, TrainARMLoop, GenerateLoopV2
from mimikit.networks.s2s_lstm_v2 import EncoderLSTM, DecoderLSTM, Seq2SeqLSTMNetwork

from .test_utils import tmp_db


def inputs_(b=8, t=32, d=16):
    return torch.randn(b, t, d)


@pytest.mark.parametrize(
    "weight_norm",
    [True, False]
)
@pytest.mark.parametrize(
    "downsampling",
    ['edge_sum', 'edge_mean', 'sum', 'mean', 'linear_resample']
)
@pytest.mark.parametrize(
    "output_dim",
    [32, 64, 128]
)
@pytest.mark.parametrize(
    "num_layers",
    [1, 2, 3, 4]
)
@pytest.mark.parametrize(
    "apply_residuals",
    [True, False]
)
@pytest.mark.parametrize(
    "input_dim",
    [32, 64, 128]
)
@pytest.mark.parametrize(
    "hop",
    [2, 4, 8]
)
def test_encoder_forward(
        hop, input_dim, apply_residuals, num_layers, output_dim, downsampling, weight_norm
):
    given_input = inputs_(4, hop, input_dim)
    under_test = EncoderLSTM(
        downsampling=downsampling, input_dim=input_dim, output_dim=output_dim,
        num_layers=num_layers, apply_residuals=apply_residuals, hop=hop,
        weight_norm=weight_norm
    )

    y, (hidden, h_c) = under_test.forward(given_input)

    assert_that(y).is_instance_of(torch.Tensor)
    assert_that(y.size(0)).is_equal_to(given_input.size(0))
    assert_that(y.size(1)).is_equal_to(1)
    assert_that(y.size(2)).is_equal_to(output_dim)

    assert_that(hidden).is_instance_of(torch.Tensor)
    assert_that(hidden.size(1)).is_equal_to(given_input.size(0))
    assert_that(hidden.size(0)).is_equal_to(2)
    assert_that(hidden.size(2)).is_equal_to(output_dim)


@pytest.mark.parametrize(
    "weight_norm",
    [True, False]
)
@pytest.mark.parametrize(
    "upsampling",
    ['repeat', 'interp', 'linear_resample']
)
@pytest.mark.parametrize(
    "num_layers",
    [1, 2, 3, 4]
)
@pytest.mark.parametrize(
    "apply_residuals",
    [True, False]
)
@pytest.mark.parametrize(
    "model_dim",
    [32, 64, 128]
)
@pytest.mark.parametrize(
    "hop",
    [2, 4, 8]
)
def test_decoder_forward(
        hop, model_dim, apply_residuals, num_layers, upsampling, weight_norm
):
    B = 4
    x = torch.randn(B, 1, model_dim)
    hidden = torch.randn(2, B, model_dim), torch.randn(2, B, model_dim)
    under_test = DecoderLSTM(
        upsampling=upsampling, model_dim=model_dim, weight_norm=weight_norm,
        num_layers=num_layers, apply_residuals=apply_residuals, hop=hop
    )

    y = under_test.forward(x, hidden)

    assert_that(y).is_instance_of(torch.Tensor)
    assert_that(y.size(0)).is_equal_to(x.size(0))
    assert_that(y.size(1)).is_equal_to(hop)
    assert_that(y.size(2)).is_equal_to(model_dim)


def test_seq2seq_forward():
    under_test = Seq2SeqLSTMNetwork.from_config(
        Seq2SeqLSTMNetwork.Config(
            io_spec=IOSpec.magspec_io(IOSpec.MagSpecIOConfig())
        )
    )
    given_inputs = (inputs_(
        4, under_test.config.hop, under_test.config.io_spec.inputs[0].elem_type.size),)

    outputs = under_test.forward(given_inputs)

    assert_that(outputs).is_instance_of(torch.Tensor)
    assert_that(outputs.size()).is_equal_to(given_inputs[0].size())


def test_should_generate(tmp_db):
    db = tmp_db("train-loop.h5")
    s2s = Seq2SeqLSTMNetwork.from_config(
        Seq2SeqLSTMNetwork.Config(
            io_spec=IOSpec.magspec_io(IOSpec.MagSpecIOConfig()),
            hop=2
        )
    )
    loop = GenerateLoopV2.from_config(
        GenerateLoopV2.Config(
            prompts_position_sec=(None,),
            batch_size=1,
        ),
        db, s2s
    )

    for outputs in loop.run():
        assert_that(len(outputs)).is_equal_to(1)
        assert_that(outputs[0]).is_instance_of(torch.Tensor)
        assert_that(torch.all(outputs[0][:, -loop.n_steps:] != 0)).is_true()


@pytest.mark.parametrize(
    "given_io",
    [
        IOSpec.magspec_io(IOSpec.MagSpecIOConfig()),
        IOSpec.mulaw_io(IOSpec.MuLawIOConfig(input_module_type="embedding"))
    ]
)
def test_should_train(tmp_db, tmp_path, given_io):
    s2s = Seq2SeqLSTMNetwork.from_config(
        Seq2SeqLSTMNetwork.Config(
            io_spec=given_io,
            hop=2
        )
    )

    db = tmp_db("train-loop.h5")
    config = TrainARMConfig(
        root_dir=str(tmp_path),
        limit_train_batches=2,
        batch_size=2,
        batch_length=s2s.config.hop,
        downsampling=64,
        max_epochs=2,
        every_n_epochs=1,
        CHECKPOINT_TRAINING=True,
        MONITOR_TRAINING=True,
        OUTPUT_TRAINING=True,
    )

    loop = TrainARMLoop.from_config(
        config, dataset=db, network=s2s
    )

    loop.run()

    content = os.listdir(os.path.join(str(tmp_path), loop.hash_))
    assert_that(content).contains("hp.yaml", "outputs", "epoch=1.ckpt")

    outputs = os.listdir(os.path.join(str(tmp_path), loop.hash_, "outputs"))
    assert_that([os.path.splitext(o)[-1] for o in outputs]).contains(".mp3")
