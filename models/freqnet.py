from ..kit import ShiftedSeqsPair, MMKHooks, EpochEndPrintHook, Dataset
from ..modules import GatedLinearInput, AbsLinearOutput, GatedConv, mean_L1_prop
from pytorch_lightning import LightningModule
from functools import partial
import torch
import torch.nn as nn
import numpy as np
from multiprocessing import cpu_count


class Padder(nn.Module):
    def __init__(self, dim, length, learn_padding=False):
        super(Padder, self).__init__()
        if learn_padding:
            self.p = nn.Parameter(torch.randn(dim, length))
        else:
            self.p = torch.zeros(dim, length)

    def forward(self, x, side):
        self.p = self.p.to(x)
        return torch.cat((x, torch.stack([self.p] * x.size(0)))[::-side], dim=-1)


class FreqLayer(nn.Module):
    def __init__(self, model_dim, strict=False, learn_padding=False, **kwargs):
        super(FreqLayer, self).__init__()
        self.gate = GatedConv(model_dim, model_dim, **kwargs)
        if "kernel_size" in kwargs:
            kwargs.pop("kernel_size")
        self.residuals = nn.Conv1d(model_dim, model_dim, 1, **kwargs)
        self.skips = nn.Conv1d(model_dim, model_dim, 1, **kwargs)
        self.pad_y = Padder(model_dim, self.output_len_diff(strict), learn_padding)
        self.pad_skip = Padder(model_dim, self.output_len_diff(strict), learn_padding)

    def forward(self, x):
        y = self.gate(x)
        skip = self.skips(y)
        y = self.residuals(y)
        return y, skip

    def output_len_diff(self, strict):
        return (self.gate.kernel - 1) * self.gate.dilation + int(strict)


class FreqBlock(nn.Module):
    def __init__(self, model_dim, n_layers, layer_func, collect_skips, **kwargs):
        super(FreqBlock, self).__init__()
        self.block = nn.ModuleList(
            [FreqLayer(model_dim,
                       dilation=2 ** i, **kwargs) for i in range(n_layers)])
        self.layer_func = layer_func
        self.collect_skips = collect_skips

    def forward(self, x, skip):
        skips = [x]
        for i, layer in enumerate(self.block):
            x, skip = self.layer_func(layer, x, skip)
            if self.collect_skips:
                skips += [skip]
        return x, skips if self.collect_skips else skip


def accum(x, y, shift=1):
    if shift == 0:
        if x.size(-1) == y.size(-1):
            return x + y
        else:
            raise ValueError("zero shift with size: %i and %i" % (x.size(-1), y.size(-1)))
    side = np.sign(shift)
    aligned_x = slice(*((shift, None)[::side]))
    n_aligned = x.size(-1) - abs(shift)
    aligned_y = slice(*((None, n_aligned * side)[::side]))
    compl_y = slice(*((n_aligned * side, None)[::side]))
    aligned = x[:, :, aligned_x] + y[:, :, aligned_y]
    rest = y[:, :, compl_y]
    return torch.cat((aligned, rest)[::side], dim=-1)


def concat(x, y, shift=1):
    if shift == 0:
        if x.size(-1) != y.size(-1):
            return concat(x, y, - x.size(-1) - y.size(-1))
        return y
    side = np.sign(shift)
    compl_x = slice(*((None, shift)[::side]))
    return torch.cat((x[:, :, compl_x], y)[::side], dim=-1)


def layer_func(layer, x, skip, strict=True,
               accum_outputs=0, concat_outputs=0,
               pad_input=0):
    if pad_input != 0:
        inpt = layer.pad_y(x, pad_input)
    else:
        inpt = x

    y, h = layer(inpt)
    shift = x.size(-1) - y.size(-1) + int(strict)

    if accum_outputs:
        if skip is None:
            skip = torch.zeros_like(h).to(h)
        y, skip = accum(x, y, shift * accum_outputs), accum(skip, h, shift * accum_outputs)

    if concat_outputs:
        if skip is None:
            skip = torch.zeros_like(h).to(h)
        y, skip = concat(x, y, shift * concat_outputs), concat(skip, h, shift * concat_outputs)

    if skip is None or (not concat_outputs and not accum_outputs):
        skip = h

    return y, skip


layer_funcs = dict(

    strict_recursive=partial(layer_func, strict=True),
    strict_concat_left=partial(layer_func, strict=True, concat_outputs=1),
    strict_concat_left_residuals_right=partial(layer_func, strict=True, accum_outputs=-1, concat_outputs=1),

    standard_recursive=partial(layer_func, strict=False),
    standard_concat_left=partial(layer_func, strict=False, concat_outputs=1),
    # concat right doesn't make much sense since it means that, counting from the input, lower layers would have to
    # shift the input more than later layers... but whatever...
    standard_concat_right=partial(layer_func, strict=False, concat_outputs=-1),

    residuals_left=partial(layer_func, strict=False, accum_outputs=1),
    residuals_left_concat_right=partial(layer_func, strict=False, accum_outputs=1, concat_outputs=-1),

    residuals_right=partial(layer_func, strict=False, accum_outputs=-1),
    residuals_right_concat_left=partial(layer_func, strict=False, accum_outputs=-1, concat_outputs=1),

    padded_left=partial(layer_func, strict=False, pad_input=1),
    padded_right=partial(layer_func, strict=False, pad_input=-1),

    padded_left_residuals=partial(layer_func, strict=False, pad_input=1, accum_outputs=1),
    padded_right_residuals=partial(layer_func, strict=False, pad_input=-1, accum_outputs=1),

)

# for some reasons, partials don't get pickled in the hparams, unless they are in a tuple...
for k, v in layer_funcs.items():
    layer_funcs[k] = (v,)


class FreqNet(MMKHooks,
              EpochEndPrintHook,
              LightningModule):
    model_defaults = dict(
        loss_fn=mean_L1_prop,
        model_dim=512,
        conv_kwargs=dict(groups=1),
        lf=layer_funcs["residuals_left"],
        layers=(int(np.log2(8)),),
        layer_wise_loss=False,
        learn_padding=False,
    )
    optim_defaults = dict(
        max_lr=5e-4,
        betas=(.9, .9),
        div_factor=3.,
        final_div_factor=1.,
        pct_start=.25,
        cycle_momentum=False,
        max_epochs=100,
    )
    # group all data-related params (splits, loaders- and wrappers-args, preparation steps...)
    # N.B.: this would mean that each model would have to implement the steps
    # that transform raw data to a, for him, consumable loader or DataModule.
    data_defaults = dict(
        batch_size=64,
        sequence_length=64,
        splits=[.85, .15],
        wrappers=None,
        to_tensor=True,
        to_gpu=True,
    )

    def __init__(self,
                 inputs,
                 model_args: dict,
                 optim_args: dict,
                 data_args: dict):
        super(FreqNet, self).__init__()
        # validate and inject params als attributes
        self._init_hparams(model_args, optim_args, data_args)
        # build the datamodule
        ds = Dataset(inputs)
        self.input_dim = ds.shape[0][-1]
        ds = ShiftedSeqsPair(self.sequence_length, self.shift)(ds)
        self.datamodule = ds.to_datamodule(self.to_gpu,
                                           self.to_tensor,
                                           self.splits,
                                           batch_size=self.batch_size,
                                           num_workers=1 if self.to_gpu else cpu_count() // 2
                                           )
        # Now we can build the model
        model_dim = self.model_dim
        layer_f, learn_padding = self.lf[0], self.learn_padding
        conv_kwargs = self.conv_kwargs
        layer_wise_loss = self.layer_wise_loss

        # Input Encoder
        self.inpt = GatedLinearInput(self.input_dim, model_dim)

        # Auto-regressive Blocks
        self.blocks = nn.ModuleList([
            FreqBlock(model_dim, n_layers, layer_f, layer_wise_loss,
                      strict=self.is_strict(), learn_padding=learn_padding, **conv_kwargs)
            for n_layers in self.layers
        ])

        # Output Decoder
        self.outpt = AbsLinearOutput(model_dim, self.input_dim)

    def forward(self, x):
        """
        input x has shape (Batch x Time x Dimensionality)
        depending on the layer_function, output has more, less, or as much time steps
        """
        x = self.inpt(x)
        skips = None
        for block in self.blocks:
            x, skips = block(x, skips)
        if self.layer_wise_loss:
            x = [self.outpt(skp) for skp in skips]
            return x if self.training else x[-1]
        x = self.outpt(skips)
        return x

    def training_step(self, batch, batch_idx):
        batch, target = batch

        if self.is_strict():
            # discard the last time steps of the input to get an output of same length as target
            batch = batch[:, :-sum(i for i in self.layers)]

        if self.layer_wise_loss:
            targets_ = torch.cat((batch[:, :self.shift], target), dim=1)
            targets_ = [targets_[:, shift:shift+self.sequence_length] for layer in self.layers_shifts for shift in layer]
            output = self.forward(batch)
            recon = sum(self.loss_fn(out, trgt[:, :out.size(1)]) for out, trgt in zip(output, targets_))
            recon = recon / len(targets_)
            self.log("recon", recon, on_step=False, on_epoch=True)
            return {"loss": recon}

        output = self.forward(batch)
        # since the output can be shorter, longer or equal to target, we have to make the shapes right...
        n_out, n_target = output.size(1), target.size(1)

        if n_out < n_target:
            target = target[:, :n_out]
        elif n_out > n_target:
            if self.concat_side > 0:
                target = torch.cat((batch[:, :self.shift], target), dim=1)
            else:
                output = output[:, :n_target]

        recon = self.loss_fn(output, target)
        self.log("recon", recon, on_step=False, on_epoch=True)
        return {"loss": recon}

    def configure_optimizers(self):
        # for warm restarts
        if self.trainer is not None and self.trainer.optimizers is not None \
                and len(self.trainer.optimizers) >= 1:
            return self.trainer.optimizers
        self.opt = torch.optim.Adam(self.parameters(), lr=self.max_lr, betas=self.betas)
        self.sched = torch.optim.lr_scheduler.OneCycleLR(self.opt,
                                                         steps_per_epoch=len(self.datamodule.train_dataloader()),
                                                         epochs=self.max_epochs,
                                                         max_lr=self.max_lr, div_factor=self.div_factor,
                                                         final_div_factor=self.final_div_factor,
                                                         pct_start=self.pct_start,
                                                         cycle_momentum=self.cycle_momentum
                                                         )
        return [self.opt], [{"scheduler": self.sched, "interval": "step", "frequency": 1}]

    def is_strict(self):
        return self.lf[0].keywords.get("strict", False)

    @property
    def shift(self):
        return sum(2 ** i + i * int(self.is_strict()) for i in self.layers)

    @property
    def receptive_field(self):
        return sum(2 ** i for i in self.layers)

    @property
    def layers_shifts(self):
        shifts = []
        for layer in self.layers:
            # 2 following lines are the shifts to compute the layer_wise_loss
            # (including the first encoding layer) of a strict block
            tpl = tuple((2 ** (i+1)) + (i * int(self.is_strict())) for i in range(layer))
            shifts += [(0, *tpl[:-1], tpl[-1] + int(self.is_strict()))]
        return tuple(shifts)

    @property
    def concat_side(self):
        return self.lf[0].keywords.get("concat_outputs", 0)

    def generation_slices(self, step_length=1):
        rf = self.receptive_field
        if self.concat_side in (0, -1):
            return slice(-rf, None), slice(None, step_length)
        return slice(-rf, None), slice(rf, rf + step_length)

    def _init_hparams(self, model_args, optim_args, data_args):
        hparams = dict()
        # inject hparams as attributes
        # model part
        for key in self.model_defaults.keys():
            setattr(self, key, model_args.get(key, self.model_defaults[key]))
            hparams.update({key: getattr(self, key)})
        # optim part
        for key in self.optim_defaults.keys():
            setattr(self, key, optim_args.get(key, self.optim_defaults[key]))
            hparams.update({key: getattr(self, key)})
        # data part
        for key in self.data_defaults.keys():
            setattr(self, key, data_args.get(key, self.data_defaults[key]))
            hparams.update({key: getattr(self, key)})
        # validate
        self._validate_hparams(hparams)
        # save
        self._set_hparams(hparams)

    def _validate_hparams(self, hparams):
        if hparams["sequence_length"] < self.receptive_field:
            raise ValueError("Expected `sequence_length` to be >= to self.receptive_field."
                             "%i < %i" % (hparams["sequence_length"], self.receptive_field))
