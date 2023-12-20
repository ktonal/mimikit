import pytest

import mimikit as mmk
from assertpy import assert_that
import torch


def test_generate():
    input_module = mmk.ZipReduceVariables(mode="sum",
                                          modules=(mmk.FramedLinearIO().set(
                                              frame_size=4, hop_length=4, out_dim=2, class_size=16
                                          ).module(),))
    tier = mmk.TransformerTier(input_module=input_module, model_dim=2, n_heads=2,
                               feedforward_dim=8, num_layers=3,
                               with_layer_norm=False, positional_encoding=None)

    inpt = torch.randint(0, 16, (1, 8))
    extras = torch.randint(0, 16, (1, 4))
    no_caching = tier(((inpt,), None))
    no_caching_extras = tier(((torch.cat((inpt, extras), dim=1),), None))[:, -1:]
    tier.train(False)
    caching = tier(((inpt,), None))
    caching_extras = tier(((extras,), None))

    no_caching = torch.cat((no_caching, no_caching_extras), dim=1)
    caching = torch.cat((caching, caching_extras), dim=1)

    assert_that((no_caching == caching).all())