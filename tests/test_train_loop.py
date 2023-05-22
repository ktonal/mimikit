import os

import pytest
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
                    module=mmk.LinearIO()
                ).bind_to(extractor),
            ),
            targets=(
                mmk.TargetSpec(
                    extractor_name=extractor.name,
                    transform=mmk.Normalize(),
                    module=mmk.LinearIO(),
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
    assert_that(content).contains("hp.yaml", "outputs", "epoch=1.ckpt")

    outputs = os.listdir(os.path.join(str(tmp_path), loop.hash_, "outputs"))
    assert_that([os.path.splitext(o)[-1] for o in outputs]).contains(".mp3")


@pytest.mark.parametrize(
    "save_optimizer",
    [True, False]
)
def test_should_resume_from_checkpoint(tmp_db, tmp_path, save_optimizer):
    db = tmp_db("train-loop.h5")
    extractor = mmk.Extractor("signal", mmk.FileToSignal(16000))
    ds_config = mmk.DatasetConfig(filename=db.filename, sources=["0", "1"], extractors=[extractor])
    db.config = ds_config
    net = TestARM(
        TestARM.Config(io_spec=mmk.IOSpec(
            inputs=(
                mmk.InputSpec(
                    extractor_name=extractor.name,
                    transform=mmk.Normalize(),
                    module=mmk.LinearIO()
                ).bind_to(extractor),
            ),
            targets=(
                mmk.TargetSpec(
                    extractor_name=extractor.name,
                    transform=mmk.Normalize(),
                    module=mmk.LinearIO(),
                    objective=mmk.Objective("reconstruction")
                ).bind_to(extractor),
            )
        ))
    )
    config = mmk.TrainARMConfig(
        root_dir=str(tmp_path),
        limit_train_batches=2,
        max_epochs=2,
        every_n_epochs=1,
        save_optimizer=save_optimizer,
        CHECKPOINT_TRAINING=True,
        MONITOR_TRAINING=True,
        OUTPUT_TRAINING=True,
    )

    loop = mmk.TrainARMLoop.from_config(
        config, dataset=db, network=net
    )

    def on_epoch_end(*args):
        raise KeyboardInterrupt()

    loop.on_train_epoch_end = on_epoch_end

    loop.run()
    loop.teardown("fit")

    content = os.listdir(os.path.join(str(tmp_path), loop.hash_))
    must_contain = ["epoch=1.ckpt"]
    if save_optimizer:
        must_contain += ["epoch=1.opt"]
    assert_that(content).contains("hp.yaml", "outputs", *must_contain)

    ckpt = mmk.Checkpoint(id=loop.hash_, epoch=1, root_dir=str(tmp_path))

    if save_optimizer:
        assert_that(ckpt.optimizer_state).is_not_none()

    assert_that(ckpt.trainer_state).is_not_none()

    ckpt_loop = mmk.TrainARMLoop.from_checkpoint(ckpt)

    assert_that(ckpt_loop).is_instance_of(mmk.TrainARMLoop)

    ckpt_loop.run()
    must_contain = ["epoch=2.ckpt"]
    if save_optimizer:
        must_contain += ["epoch=2.opt"]
    content = os.listdir(os.path.join(str(tmp_path), loop.hash_))
    assert_that(content).contains("hp.yaml", "outputs", *must_contain)
