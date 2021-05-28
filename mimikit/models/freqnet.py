import torch
from torchaudio.transforms import GriffinLim
import dataclasses as dtc

from ..data import Input, Target, AsSlice
from ..loss_functions import mean_L1_prop
from .parts import SuperAdam, SequenceModel
from .model import model
from ..networks import FreqNetNetwork, WNNetwork

__all__ = [
    'FreqNet',
    'main'
]


@model
@dtc.dataclass(repr=False)
class FreqNet(SequenceModel,
              SuperAdam,
              FreqNetNetwork,
              # we have to specify this parent here to avoid infinite recursion in __init__
              WNNetwork
              ):

    batch_size: int = 32
    batch_seq_length: int = 64

    @staticmethod
    def loss_fn(output, target):
        return {"loss": mean_L1_prop(output, target)}

    def setup(self, stage: str):
        SequenceModel.setup(self, stage)
        SuperAdam.setup(self, stage)

    def batch_signature(self):
        inpt = Input('fft', AsSlice(shift=0, length=self.hparams.batch_seq_length))
        trgt = Target('fft', AsSlice(shift=self.shift,
                                     length=self.output_shape((-1, self.hparams.batch_seq_length, -1))[1]))
        return inpt, trgt

    def generation_slices(self):
        # input is always the last receptive field
        input_slice = slice(-self.receptive_field, None)
        if self.pad_input == 1:
            # then there's only one future time-step : the last output
            output_slice = slice(-1, None)
        else:
            # for all other cases, the first output has the shift of the whole network
            output_slice = slice(None, 1)
        return input_slice, output_slice

    def decode_outputs(self, outputs: torch.Tensor):
        gla = GriffinLim(n_fft=self.hparams.n_fft, hop_length=self.hparams.hop_length, power=1.,
                         wkwargs=dict(device=outputs.device))
        return gla(outputs.transpose(-1, -2).contiguous())

    def get_prompts(self, n_prompts, prompt_length=None):
        return next(iter(self.datamodule.train_dataloader()))[0][:n_prompts, :prompt_length]

    def generate(self, prompt, n_steps, decode_outputs=False, **kwargs):
        self.before_generate()

        output = self.prepare_prompt(prompt, n_steps, at_least_nd=3)
        prior_t = prompt.size(1)
        rf = self.receptive_field
        _, out_slc = self.generation_slices()

        for t in self.generate_tqdm(range(prior_t, prior_t + n_steps)):
            output.data[:, t:t + 1] = self.forward(output[:, t - rf:t])[:, out_slc]

        if decode_outputs:
            output = self.decode_outputs(output)

        self.after_generate()

        return output


def main(
        sources='./gould',
        sr=22050, n_fft=2048, hop_length=512,
        segments_labels=False, files_labels=False):
    import os
    import mimikit as mmk

    schema = {"fft": mmk.Spectrogram(sr=sr, n_fft=n_fft, hop_length=hop_length,
                                     magspec=True)}
    if segments_labels:
        schema.update({'s_labels': mmk.SegmentLabels(base_repr=schema['fft'])})
    if files_labels:
        schema.update({'f_labels': None})

    db_path = '/tmp/freqnet_db.h5'
    if not os.path.exists(db_path):
        db = mmk.Database.build(db_path, sources, schema)
    else:
        db = mmk.Database(db_path)

    net = mmk.FreqNet(
        input_dim=db.fft.shape[-1]
    )

    dm = mmk.DataModule(net, db,
                        loader_kwargs=dict(
                            batch_size=net.batch_size,
                            drop_last=False,
                            shuffle=True
                        ))
    trainer = mmk.get_trainer(root_dir=None, max_epochs=32)

    trainer.fit(net, datamodule=dm)


if __name__ == '__main__':
    main()
