import pytest
import torch
from assertpy import assert_that

from mimikit.modules.inputs import FramedInput
from mimikit.networks.sample_rnn_v2 import SampleRNN
from mimikit.checkpoint import Checkpoint


def test_should_instantiate_from_default_config():
    given_config = SampleRNN.Config()

    under_test = SampleRNN.from_config(given_config)

    assert_that(type(under_test)).is_equal_to(SampleRNN)
    assert_that(len(under_test.tiers)).is_equal_to(len(given_config.frame_sizes))


@pytest.skip("Not really supported yet...")
def test_should_take_n_unfolded_inputs():
    given_frame_sizes = (16, 4, 8,)
    given_config = SampleRNN.Config(
        frame_sizes=given_frame_sizes,
        inputs=(
            # 3 inputs of different dtypes
            FramedInput.Config(class_size=None, projection_type="fir"),
            FramedInput.Config(class_size=None, projection_type="linear"),
            FramedInput.Config(class_size=8, projection_type="embedding"),
        ),
        inputs_mode='static_mix',
        unfold_inputs=False
    )
    n_frames = (1, given_frame_sizes[0]//given_frame_sizes[1])
    given_inputs = tuple(
        (torch.randn(8, n, fs), torch.randn(8, n, fs), torch.randint(0, 8, (8, n, fs)))
        for fs, n in zip(given_frame_sizes[:-1], n_frames)
    )
    given_inputs = (*given_inputs,
                    (
                        torch.randn(8, given_frame_sizes[0], given_frame_sizes[-1]),
                        torch.randn(8, given_frame_sizes[0], given_frame_sizes[-1]),
                        torch.randint(0, 8, (8, given_frame_sizes[0], given_frame_sizes[-1])))
                    )

    under_test = SampleRNN.from_config(given_config)
    outputs: torch.Tensor = under_test(given_inputs)

    assert_that(type(outputs)).is_equal_to(torch.Tensor)
    assert_that(outputs.shape).is_equal_to((8, given_frame_sizes[0], given_config.hidden_dim))


def test_should_load_when_saved(tmp_path_factory):
    given_config = SampleRNN.Config()
    root = str(tmp_path_factory.mktemp("ckpt"))
    srnn = SampleRNN.from_config(given_config)
    ckpt = Checkpoint(id="123", epoch=1, root_dir=root)

    ckpt.create(model_config=srnn.config, network=srnn)
    loaded = ckpt.network

    assert_that(type(loaded)).is_equal_to(SampleRNN)
