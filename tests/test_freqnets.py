import pytest
import numpy as np
import torch

from ..mmk.kit import get_trainer
from ..mmk.freqnet import *


def test_base(tmp_path):
    class TestModel(FreqNetModel):
        def __init__(self, **kwargs):
            super(TestModel, self).__init__(**kwargs)
            self.fc = torch.nn.Linear(1025, 1025)
            self.loss_fn = torch.nn.MSELoss()
            # torch automatically converts numpy to torch.float64
            # but then the parameters are torch.float32 by default...
            self.datamodule.ds.to_tensor()
            self.datamodule.ds.to(torch.float32)

        def forward(self, x):
            return self.fc(x)

        def targets_shifts_and_lengths(self, input_length):
            # shift and length of each target
            return [(4, input_length), ]

    base = TestModel(inputs=np.random.randn(32, 1025),
                     input_seq_length=16,
                     batch_size=10,
                     to_gpu=False,
                     splits=[1.])

    assert isinstance(base.datamodule, FreqData)
    assert isinstance(base.optim, FreqOptim)

    trainer = get_trainer(tmp_path, max_epochs=1)
    trainer.fit(base)

    assert base.datamodule.train_dataloader() is not None
    loader = base.datamodule.train_dataloader()
    batch = next(iter(loader))
    assert isinstance(batch, list), type(batch)
    assert len(batch) == 2, len(batch)
    assert isinstance(batch[0], torch.Tensor), type(batch[0])
    assert isinstance(batch[1], torch.Tensor), type(batch[1])

    optims = base.configure_optimizers()
    assert optims is not None
    assert len(optims[0][0].param_groups[0]) > 0


def test_layer_computed_properties():
    cases = [
        (FreqLayer(layer_index=0, ),
         dict(shift=2, receptive_field=2, rel_shift=1)),
        (FreqLayer(layer_index=0, strict=True),
         dict(shift=2, receptive_field=2, rel_shift=2)),
        (FreqLayer(layer_index=1, ),
         dict(shift=4, receptive_field=4, rel_shift=2)),
        (FreqLayer(layer_index=1, strict=True),
         dict(shift=5, receptive_field=4, rel_shift=3)),
        (FreqLayer(layer_index=2, ),
         dict(shift=8, receptive_field=8, rel_shift=4)),
        (FreqLayer(layer_index=2, strict=True),
         dict(shift=10, receptive_field=8, rel_shift=5)),
    ]

    for i, (layer, expected) in enumerate(cases):

        for attr in expected.keys():
            assert getattr(layer, attr)() == expected[attr], \
                ("case_" + str(i), attr, getattr(layer, attr)(), expected[attr])


def test_single_layer_output_length():
    input_length = 32
    cases = [
        (FreqLayer(layer_index=0, ), input_length - 1),
        (FreqLayer(layer_index=0, strict=True), input_length - 1),
        (FreqLayer(layer_index=1, ), input_length - 2),
        (FreqLayer(layer_index=2, strict=True), input_length - 4),
        (FreqLayer(layer_index=0, concat_outputs="left"), input_length),
        # strict && concat always ADDS a time-step (concat appends MORE past)
        (FreqLayer(layer_index=1, strict=True, concat_outputs="left"), input_length + 1),
        (FreqLayer(layer_index=1, pad_input="left"), input_length),
        # strict && pad returns same shape : NOT LIKE CONCAT !
        (FreqLayer(layer_index=2, strict=True, pad_input="left"), input_length),
    ]
    for i, (layer, expected) in enumerate(cases):
        assert layer.output_length(input_length) == expected, \
            ("computed case_" + str(i), layer.output_length(input_length), expected)

        inpt = torch.randn(1, layer.layer_dim, input_length)
        outpt, _ = layer(inpt)
        assert outpt.size(-1) == expected, \
            ("forward case_" + str(i), outpt.size(-1), expected, layer.padding)


def test_freqnet_computed_properties():
    input_length = 16
    inputs = torch.randn(1, input_length, 1025)

    cases = [
        (FreqNet(inputs=inputs, input_seq_length=input_length,
                 n_layers=(1, 1, 1),
                 ),
         dict(shift=4,
              receptive_field=4,
              all_rel_shifts=(1, 1, 1),
              all_shifts=(2, 3, 4),
              targets=[(4, input_length-3)],
              out_length=input_length-3,
              )),
        (FreqNet(inputs=inputs, input_seq_length=input_length,
                 n_layers=(1, 1, 1),
                 strict=True),
         dict(shift=6,
              receptive_field=4,
              all_rel_shifts=(2, 2, 2),
              all_shifts=(2, 4, 6),
              targets=[(6, input_length-3)],
              out_length=input_length-3)),
        (FreqNet(inputs=inputs, input_seq_length=input_length,
                 n_layers=(3,),
                 ),
         dict(shift=8,
              receptive_field=8,
              all_rel_shifts=(1, 2, 4),
              all_shifts=(2, 4, 8),
              targets=[(8, input_length-7)],
              out_length=input_length-7)),
        (FreqNet(inputs=inputs, input_seq_length=input_length,
                 n_layers=(3,),
                 strict=True),
         dict(shift=8+2,  # strict shifts (n_layers -1) MORE than not strict
              receptive_field=8,
              all_rel_shifts=(2, 3, 5),
              all_shifts=(2, 5, 10),
              targets=[(8+2, input_length-7)],
              out_length=input_length-7)),
        (FreqNet(inputs=inputs, input_seq_length=input_length,
                 n_layers=(3, 3),
                 ),
         dict(shift=15,
              receptive_field=15,
              all_rel_shifts=(1, 2, 4, 1, 2, 4),
              all_shifts=(2, 4, 8, 9, 11, 15),
              targets=[(15, input_length-14)],
              out_length=input_length-14)),
        (FreqNet(inputs=inputs, input_seq_length=input_length,
                 n_layers=(3, 3),
                 strict=True),
         dict(shift=20,
              receptive_field=15,
              all_rel_shifts=(2, 3, 5, 2, 3, 5),
              all_shifts=(2, 5, 10, 12, 15, 20),
              targets=[(20, input_length-14)],
              out_length=input_length-14)),
    ]

    for i, (net, expected) in enumerate(cases):

        for attr in expected.keys():
            if attr not in ("out_length", "targets"):
                assert getattr(net, attr)() == expected[attr], \
                    ("case_" + str(i), attr, getattr(net, attr)(), expected[attr])
            if attr == "out_length":
                # verify that computed and returned lengths are both equal to the one that is expected
                # computed
                assert net.output_length(input_length) == expected[attr], \
                    ("computed case_" + str(i), net.output_length(input_length), expected[attr])
                # returned
                outpt = net(inputs)
                assert outpt.size(1) == expected[attr], \
                    ("forward case_" + str(i), outpt.size(1), expected[attr])

            if attr == "targets":
                result = net.targets_shifts_and_lengths(input_length)
                assert result == expected[attr], \
                    (attr, "case_" + str(i), result, expected[attr])


def test_models_train(tmp_path):

    inputs = torch.randn(32, 1025)
    kwargs = dict(splits=None, input_seq_length=8)
    models = [

        FreqNet(inputs=inputs, **kwargs),
        FreaksNet(inputs=inputs, **kwargs),
        HKFreqNet(inputs=inputs, **kwargs),
        FrexpertMixture(inputs=inputs, **kwargs),
        LayerWiseLossFreqNet(inputs=inputs, **kwargs)
    ]

    for mdl in models:

        trainer = get_trainer(tmp_path, max_epochs=1)

        trainer.fit(mdl)

        assert trainer.global_step > 0, type(mdl)