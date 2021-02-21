import torch.nn as nn
import numpy as np
from pytorch_lightning import LightningModule

from ..modules import DilatedConv1d
from mimikit.kit.modules.ops import Transpose


class DilatedConv1dBlocks(LightningModule):
    LAYER_KWARGS = ["groups", "strict", "accum_outputs", "concat_outputs", "kernel_size",
                    "pad_input", "learn_padding", "with_skip_conv", "with_residual_conv",
                    "local_cond_dim", "global_cond_dim"]

    def __init__(self,
                 input_dim=None,
                 model_dim=512,
                 n_layers=(2,),
                 groups=1,
                 kernel_size=2,
                 strict=False,
                 accum_outputs=0,
                 concat_outputs=0,
                 pad_input=0,
                 learn_padding=False,
                 with_skip_conv=True,
                 with_residual_conv=True,
                 local_cond_dim=None,
                 global_cond_dim=None
                 ):
        super(DilatedConv1dBlocks, self).__init__()
        self.model_dim = model_dim
        if input_dim is None:
            raise ValueError("input_dim can not be None")
        self.input_dim = input_dim
        self.groups = groups
        self.n_layers = n_layers
        self.kernel_size = kernel_size
        self.strict = strict
        self.accum_outputs = accum_outputs
        self.concat_outputs = concat_outputs
        self.pad_input = pad_input
        self.learn_padding = learn_padding
        self.with_skip_conv = with_skip_conv
        self.with_residual_conv = with_residual_conv
        self.local_cond_dim = local_cond_dim
        self.global_cond_dim = global_cond_dim

        layer_kwargs = {attr: getattr(self, attr) for attr in self.LAYER_KWARGS}
        # for simplicity we keep all the layers in a flat list
        self.layers = nn.ModuleList(
            [Transpose(1, 2)] + [
                DilatedConv1d(layer_index=i, input_dim=model_dim, layer_dim=model_dim, **layer_kwargs)
                for n_layers in self.n_layers for i in range(n_layers)
            ] + [Transpose(1, 2)])

        self.save_hyperparameters()

    def forward(self, x, local_cond=None, global_cond=None):
        """
        """
        skips = None
        for layer in self.layers:
            x, skips = layer(x, skips, local_cond, global_cond)
        return x

    def all_rel_shifts(self):
        """sequence of shifts from one layer to the next"""
        return tuple(layer.rel_shift() for layer in self.layers)

    def shift(self):
        """total shift of the network"""
        if not self.strict and (self.pad_input == 1 or self.concat_outputs == 1):
            return 1
        elif self.pad_input == -1 or self.concat_outputs == -1:
            return self.receptive_field()
        elif self.strict and self.concat_outputs == 1:
            return 0
        elif self.strict and self.pad_input == 1:
            return len(self.layers)
        else:
            return sum(self.all_rel_shifts()) + int(not self.strict)

    def all_shifts(self):
        """the accumulated shift at each layer"""
        return tuple(np.cumsum(self.all_rel_shifts()) + int(not self.strict))

    def receptive_field(self):
        block_rf = []
        for i, layer in enumerate(self.layers[:-1]):
            if self.layers[i + 1].layer_index == 0:
                block_rf += [layer.receptive_field() - 1]
        block_rf += [self.layers[-1].receptive_field()]
        return sum(block_rf)

    def output_length(self, input_length):
        return self.all_output_lengths(input_length)[-1]

    def all_output_lengths(self, input_length):
        out_length = input_length
        lengths = []
        for layer in self.layers:
            out_length = layer.output_length(out_length)
            lengths += [out_length]
        return tuple(lengths)
