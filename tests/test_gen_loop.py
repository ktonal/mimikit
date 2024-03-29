import torch
from assertpy import assert_that

import mimikit.features.extractor
from .test_utils import TestARM, TestDB, tmp_db
import mimikit as mmk


def test_should_run(tmp_db):
    db: TestDB = tmp_db("gen-test.h5")
    extractor = mimikit.features.extractor.Extractor("signal", mmk.FileToSignal(16000))
    net = TestARM(
        TestARM.Config(io_spec=mmk.IOSpec(
            inputs=(
                mmk.InputSpec(
                    extractor_name=extractor.name,
                    transform=mmk.Normalize(),
                    module=mmk.LinearIO()
                ).bind_to(extractor),
                mmk.InputSpec(
                    extractor_name=extractor.name,
                    transform=mmk.MuLawCompress(256),
                    module=mmk.LinearIO()
                ).bind_to(extractor),
            ),
            targets=(
                mmk.TargetSpec(
                    extractor_name=extractor.name,
                    transform=mmk.Normalize(),
                    module=mmk.LinearIO(),
                    objective=mmk.Objective("none")
                ).bind_to(extractor),
                mmk.TargetSpec(
                    extractor_name=extractor.name,
                    transform=mmk.MuLawCompress(256),
                    module=mmk.LinearIO(),
                    objective=mmk.Objective("none")
                ).bind_to(extractor),
            )
        ))
    )

    assert_that(net).is_instance_of(TestARM)

    loop = mmk.GenerateLoopV2.from_config(
        mmk.GenerateLoopV2.Config(
            prompts_position_sec=(None,),
            batch_size=1,
        ),
        db, net
    )

    assert_that(loop).is_instance_of(mmk.GenerateLoopV2)
    for outputs in loop.run():
        assert_that(len(outputs)).is_equal_to(2)
        assert_that(outputs[0]).is_instance_of(torch.Tensor)
        assert_that(torch.all(outputs[0][:, -loop.n_steps:] != 0)).is_true()
