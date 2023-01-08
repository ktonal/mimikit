import os

import pytest
from assertpy import assert_that

from .test_utils import tmp_db, TestARM
import mimikit as mmk


def test_should_run(tmp_db, tmp_path):
    db = tmp_db("train-loop.h5")
    extractor = mmk.Extractor("snd", mmk.FileToSignal(16000))
    net = TestARM(
        TestARM.Config(io_spec=mmk.IOSpec(
            inputs=(
                mmk.InputSpec(
                    extractor=extractor,
                    transform=mmk.Normalize(),
                    module=mmk.IOFactory("linear")
                ),
            ),
            targets=(
                mmk.TargetSpec(
                    extractor=extractor,
                    transform=mmk.Normalize(),
                    module=mmk.IOFactory("linear"),
                    objective=mmk.Objective("reconstruction")
                ),
            )
        ))
    )
    config = mmk.TrainARMConfig(
        root_dir=str(tmp_path),
        limit_train_batches=4,
        max_epochs=4,
        every_n_epochs=1,
        CHECKPOINT_TRAINING=True,
        MONITOR_TRAINING=True,
        OUTPUT_TRAINING=True,
    )

    loop = mmk.TrainLoop.from_config(
        config, soundbank=db, network=net
    )

    loop.run()

    content = os.listdir(os.path.join(str(tmp_path), loop.hash_))
    assert_that(content).contains("hp.yaml", "outputs", "epoch=1.h5")

    outputs = os.listdir(os.path.join(str(tmp_path), loop.hash_, "outputs"))
    assert_that([os.path.splitext(o)[-1] for o in outputs]).contains(".mp3")
