from ..model_base import Model
from .modules import *
from music_modelling_kit.modules import GatedLinearInput, AbsLinearOutput

from functools import partial


layer_funcs = dict(

    strict_recursive=partial(layer_func, strict=True),
    strict_concat_left=partial(layer_func, strict=True, concat_outputs=1),
    strict_concat_left_residuals_right=partial(layer_func, strict=True, accum_outputs=-1, concat_outputs=1),

    standard_recursive=partial(layer_func, strict=False),
    standard_concat_left=partial(layer_func, strict=False, concat_outputs=1),
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


class FreqNet(Model):

    @property
    def strict(self):
        return self.lf[0].keywords.get("strict", False)

    @property
    def shift(self):
        return sum(2 ** i + i * int(self.strict) for i in self.layers)

    @property
    def receptive_field(self):
        return self.shift - sum(i * int(self.strict) for i in self.layers)

    @property
    def concat_side(self):
        return self.lf[0].keywords.get("concat_outputs", 0)

    def __init__(self, **kwargs):
        super(FreqNet, self).__init__(**kwargs)

        gate_c, residuals_c, skips_c = self.gate_c, self.residuals_c, self.skips_c
        layer_f, strict, learn_values = self.lf[0], self.strict, self.learn_values

        # Input Encoder
        self.inpt = GatedLinearInput(1025, residuals_c)

        # Autoregressive Part
        self.blocks = nn.ModuleList([
            FreqBlock(gate_c, residuals_c, skips_c, n_layers, layer_f,
                      strict=strict, learn_values=learn_values)
            for n_layers in self.layers
        ])

        # Output Decoder
        self.outpt = AbsLinearOutput(skips_c, 1025)

    def forward(self, x):
        """
        input x has shape (Batch x Time x Dimensionality)
        depending on the layer_function, output has more, less, or as much time steps
        """
        x = self.inpt(x)
        skips = None
        for block in self.blocks:
            x, skips = block(x, skips)
        x = self.outpt(skips)
        return x

    def training_step(self, batch, batch_idx):
        batch, target = batch

        if self.strict:
            # discard the last time steps of the input to get an output of same length as target
            batch = batch[:, :-sum(i for i in self.layers)]

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
        self.ep_losses += [recon.item()]
        return {"loss": recon}

    # TODO: def generate_slice(self)

