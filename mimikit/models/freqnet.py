from abc import ABC

import numpy as np
import torch
import torch.nn as nn
from pytorch_lightning import LightningModule

from ..audios import transforms as T
from ..kit.db_dataset import DBDataset
from ..kit.ds_utils import ShiftedSequences

from ..kit.modules import GatedLinearUnit, mean_L1_prop, DilatedConv1d
from mimikit.kit.modules.ops import Abs, Transpose

from ..kit.sub_models.optim import SuperAdam
from ..kit.sub_models.sequence_model import SequenceModel


class FreqNetDB(DBDataset):
    features = ["fft"]
    fft = None

    @staticmethod
    def extract(path, n_fft=2048, hop_length=512, sr=22050):
        params = dict(n_fft=n_fft, hop_length=hop_length, sr=sr)
        fft = T.FileTo.mag_spec(path, **params)
        return dict(fft=(params, fft.T, None))

    def prepare_dataset(self, model):
        args = model.targets_shifts_and_lengths(model.hparams["input_seq_length"])
        self.slicer = ShiftedSequences(len(self.fft), [(0, model.hparams["input_seq_length"])] + args)

    def __getitem__(self, item):
        slices = self.slicer(item)
        return tuple(self.fft[sl] for sl in slices)

    def __len__(self):
        return len(self.slicer)


class FreqNetBase(SequenceModel,
                  SuperAdam,
                  LightningModule,
                  ABC):

    def __init__(self,
                 db=None,
                 files=None,
                 input_seq_length=64,
                 batch_size=64,
                 in_mem_data=True,
                 splits=[.8, .2],
                 max_lr=1e-3,
                 betas=(.9, .9),
                 div_factor=3.,
                 final_div_factor=1.,
                 pct_start=.25,
                 cycle_momentum=False,
                 **loaders_kwargs):
        super(LightningModule, self).__init__()
        SequenceModel.__init__(self)
        SuperAdam.__init__(self, max_lr, betas, div_factor, final_div_factor, pct_start, cycle_momentum)
        self.save_hyperparameters()

    def setup(self, stage: str):
        # SuperAdam needs setup
        SuperAdam.setup(self, stage)


class FreqNet(FreqNetBase):
    db_class = FreqNetDB

    LAYER_KWARGS = ["groups", "strict", "accum_outputs", "concat_outputs", "kernel_size",
                    "pad_input", "learn_padding", "with_skip_conv", "with_residual_conv"]

    def __init__(self,
                 loss_fn=mean_L1_prop,
                 input_dim=None,
                 model_dim=512,
                 groups=1,
                 n_layers=(2,),
                 kernel_size=2,
                 strict=False,
                 accum_outputs=0,
                 concat_outputs=0,
                 pad_input=0,
                 learn_padding=False,
                 with_skip_conv=True,
                 with_residual_conv=True,
                 **data_and_optim_kwargs):
        super(FreqNet, self).__init__(**data_and_optim_kwargs)
        self._loss_fn = loss_fn
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

        # Input Encoder (input = ( B x T x D ) )
        self.inpt = nn.Sequential(
            GatedLinearUnit(self.input_dim, self.model_dim), Transpose(1, 2)
        )

        # Auto-regressive Part (input = ( B x D x T ) )
        layer_kwargs = {attr: getattr(self, attr) for attr in self.LAYER_KWARGS}
        # for simplicity we keep all the layers in a flat list
        self.layers = nn.ModuleList([
            DilatedConv1d(layer_index=i, input_dim=model_dim, layer_dim=model_dim, **layer_kwargs)
            for n_layers in self.n_layers for i in range(n_layers)
        ])

        # Output Decoder (output = ( B x T x D ) )
        self.outpt = nn.Sequential(
            Transpose(1, 2), nn.Linear(self.model_dim, self.input_dim), Abs()
        )

        self.save_hyperparameters()

    def forward(self, x):
        """
        """
        x = self.inpt(x)
        skips = None
        for layer in self.layers:
            x, skips = layer(x, skips)
        x = self.outpt(skips)
        return x

    def loss_fn(self, predictions, targets):
        return self._loss_fn(predictions, targets)

    def batch_info(self, input_length):
        return [(self.shift(), self.output_length(input_length))]

    def generation_slices(self):
        # input is always the last receptive field
        input_slice = slice(-self.receptive_field(), None)
        if not self.strict and (self.pad_input == 1 or self.concat_outputs == 1):
            # then there's only one future time-step : the last output
            output_slice = slice(-1, None)
        elif self.strict and (self.pad_input == 1 or self.concat_outputs == 1):
            # then there are as many future-steps as they are layers and they all are
            # at the end of the outputs
            output_slice = slice(-len(self.layers), None)
        else:
            # for all other cases, the first output has the shift of the whole network
            output_slice = slice(None, 1)
        return input_slice, output_slice


class LayerWiseLossFreqNet(FreqNet):

    def forward(self, x):
        """
        """
        x = self.inpt(x)
        # we collect all the layers outputs
        skips = None
        outputs = []
        for layer in self.layers:
            x, skips = layer(x, skips)
            # decode each layer's output
            outputs += [self.outpt(skips)]
        if self.training:
            # pass them all to the loss function
            return tuple(outputs)
        # else redirect to a special method for inference
        return self.infer_layer_wise(outputs)

    def training_step(self, batch, batch_idx):
        batch = batch[0], batch[1:]
        return super(LayerWiseLossFreqNet, self).training_step(batch, batch_idx)

    def infer_layer_wise(self, outputs):
        """
        Place to implement the inference method when trained layer-wise.
        this default method returns the last output if the network isn't strict, otherwise, it collects the future
        time-steps where they are first predicted in the network (Xt+1 @ layer_1, Xt+2 @ layer_2...)
        @param outputs: list containing all the layers outputs
        @return: the prediction of the network
        """
        if not self.strict:
            return outputs[-1][-1]
        # we collect one future time step pro layer
        rf = self.receptive_field()
        future_steps = []
        for layer_out, shift in zip(outputs, self.all_shifts()):
            # we keep everything from the last layer, hence the condition on the slice
            step = slice(rf - shift, rf - shift + 1) if rf - shift > 0 else slice(None, None)
            future_steps += [layer_out[:, step]]
        return torch.cat(future_steps, dim=1)

    def loss_fn(self, predictions, targets):
        denominator = len(self.layers)
        return sum(self._loss_fn(pred, trg) / denominator for pred, trg in zip(predictions, targets))

    def batch_info(self, input_length):
        shifts, lengths = self.all_shifts(), self.all_output_lengths(input_length)
        return list(zip(shifts, lengths))


class HKFreqNet(FreqNet):
    """
    Add a hierarchical quality to the conditioning sequence by combining FreqLayers in a
    final Conv2d making depth in the network proportional to the distance to the predicted time-step.

    N.B : Could be fun to try the other way around : the closer to now -> the deeper in the net.
    """

    def __init__(self,
                 loss_fn=mean_L1_prop,
                 input_dim=None,
                 model_dim=512,
                 groups=1,
                 n_layers=2,
                 strict=False,
                 kernel_size=2,
                 accum_outputs=0,
                 concat_outputs=0,
                 pad_input=0,
                 learn_padding=False,
                 with_skip_conv=True,
                 with_residual_conv=True,
                 **data_optim_kwargs):
        super(HKFreqNet, self).__init__(loss_fn, input_dim, model_dim, groups, (n_layers,), kernel_size, strict,
                                        accum_outputs, concat_outputs, pad_input, learn_padding,
                                        with_skip_conv, with_residual_conv,
                                        **data_optim_kwargs)
        # the 4th dimension in the final conv2d is 2 encoder outputs + n_conv_layers
        self.conv_f = nn.Conv2d(model_dim, model_dim, (1, n_layers + 2), groups=groups)
        self.conv_g = nn.Conv2d(model_dim, model_dim, (1, n_layers + 2), groups=groups)

    def forward(self, x):
        """

        """
        x = self.inpt(x)

        # the conv2d layer adds a power of 2 to the n standard layers before it
        n_outs = self.output_length(x.size(-1))

        # stack the 2 encoder outputs
        hk_input = torch.stack((x[:, :, -n_outs - 1:-1], x[:, :, -n_outs:]), dim=-1)
        skip = None

        for layer in self.layers:
            x, skip = layer(x, skip)
            # we only keep n_outs outputs finishing at the -`layer.shift()` step
            to_stack = slice(-n_outs - layer.shift(), -layer.shift())
            hk_input = torch.cat((hk_input, skip[:, :, to_stack].unsqueeze(-1)), dim=-1)

        # Do the Gated2dConv on the gathered pieces of layers
        f = nn.Tanh()(self.conv_f(hk_input))
        g = nn.Sigmoid()(self.conv_g(hk_input))

        y = self.outpt((f * g).squeeze(-1))
        return y

    def receptive_field(self):
        # rf of the standard layers
        rf = super(HKFreqNet, self).receptive_field()
        # rf WITH HK layer :
        return rf * 2

    def all_rel_shifts(self):
        """sequence of shifts from one layer to the next"""
        stack = super(HKFreqNet, self).all_rel_shifts()
        # add the last HK layer
        return (*stack, 2 * stack[-1])

    def all_output_lengths(self, input_length):
        out_length = input_length
        lengths = []
        for layer in self.layers:
            out_length = layer.output_length(out_length)
            lengths += [out_length]
        # add the last HK layer
        lengths += [out_length - (2 ** (len(self.layers)))]
        return tuple(lengths)
