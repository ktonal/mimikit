import os
from assertpy import assert_that

from .test_utils import tmp_db, TestARM
import mimikit as mmk


def test_should_run(tmp_db, tmp_path):
    db = tmp_db("train-loop.h5")
    extractor = mmk.Extractor("signal", mmk.FileToSignal(16000))
    net = TestARM(
        TestARM.Config(io_spec=mmk.IOSpec(
            inputs=(
                mmk.InputSpec(
                    extractor_name=extractor.name,
                    transform=mmk.Normalize(),
                    module=mmk.IOFactory("linear")
                ).bind_to(extractor),
            ),
            targets=(
                mmk.TargetSpec(
                    extractor_name=extractor.name,
                    transform=mmk.Normalize(),
                    module=mmk.IOFactory("linear"),
                    objective=mmk.Objective("reconstruction")
                ).bind_to(extractor),
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

    loop = mmk.TrainARMLoop.from_config(
        config, dataset=db, network=net
    )

    loop.run()

    content = os.listdir(os.path.join(str(tmp_path), loop.hash_))
    assert_that(content).contains("hp.yaml", "outputs", "epoch=1.h5")

    outputs = os.listdir(os.path.join(str(tmp_path), loop.hash_, "outputs"))
    assert_that([os.path.splitext(o)[-1] for o in outputs]).contains(".mp3")
