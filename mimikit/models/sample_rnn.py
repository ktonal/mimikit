import inspect
import click
import torch
import torch.nn as nn
import numpy as np
from random import randint
import dataclasses as dtc

from ..audios.features import MuLawSignal
from ..data import Database, DataModule, TBPTTSampler
from mimikit.data.datamodule import AsSlice, AsFramedSlice, Input, Target
from .parts import SuperAdam, SequenceModel
from ..networks.sample_rnn import SampleRNNNetwork

from mimikit.models.model import model

__all__ = [
    'SampleRNN',
    'main'
]


@model
@dtc.dataclass(repr=False)
class SampleRNN(SequenceModel,
                SuperAdam,
                SampleRNNNetwork):
    sr: int = 16000
    batch_size: int = 16
    chunk_len: int = 16000 * 8
    batch_seq_len: int = 512

    @staticmethod
    def loss_fn(output, target):
        criterion = nn.CrossEntropyLoss(reduction="mean")
        return {"loss": criterion(output.view(-1, output.size(-1)), target.view(-1))}

    @property
    def db_schema(self):
        return {'qx': MuLawSignal(sr=self.sr, q_levels=self.q_levels)}

    def batch_signature(self):
        batch_seq_len, frame_sizes = tuple(getattr(self, key) for key in ["batch_seq_len", "frame_sizes"])
        shifts = [frame_sizes[0] - size for size in frame_sizes]
        inputs = []
        for fs, shift in zip(frame_sizes[:-1], shifts[:-1]):
            inputs.append(
                Input('qx', AsFramedSlice(shift, batch_seq_len, frame_size=fs,
                                          as_strided=False)))
        inputs.append(
            Input('qx', AsFramedSlice(shifts[-1], batch_seq_len, frame_size=frame_sizes[-1],
                                      as_strided=True)))
        targets = Target('qx', AsSlice(shift=frame_sizes[0], length=batch_seq_len))
        return inputs, targets

    def loader_kwargs(self, datamodule):
        batch_sampler = TBPTTSampler(len(datamodule.train_ds),
                                     self.batch_size,
                                     self.chunk_len,
                                     self.batch_seq_len)
        return dict(batch_sampler=batch_sampler)

    def setup(self, stage: str):
        SequenceModel.setup(self, stage)
        SuperAdam.setup(self, stage)

    def on_train_batch_start(self, batch, batch_idx, dataloader_idx):
        if (batch_idx * self.hparams.batch_seq_len) % self.hparams.chunk_len == 0:
            self.reset_h0()

    def encode_inputs(self, inputs: torch.Tensor):
        return self.db_schema['qx'].encode(inputs)

    def decode_outputs(self, outputs: torch.Tensor):
        return self.db_schema['qx'].decode(outputs)

    def get_prompts(self, n_prompts, prompt_length=None):
        # TODO : add idx=None arg, move method to SequenceModel and use self.input_feature
        if prompt_length is None:
            prompt_length = self.hparams.batch_seq_len
        N = len(self.db.qx) - prompt_length
        idx = sorted([randint(0, N) for _ in range(n_prompts)])
        if isinstance(self.db.qx, torch.Tensor):
            stack = lambda t: torch.stack(t, dim=0)
        else:
            stack = lambda t: np.stack(t, axis=0)
        return stack(tuple(self.db.qx[i:i + prompt_length].squeeze() for i in idx))

    def generate(self, prompt, n_steps=16000, decode_outputs=False, temperature=.5):
        # prepare model
        self.before_generate()
        output = self.prepare_prompt(prompt, n_steps, at_least_nd=2)
        # trim to start with a whole number of top frames
        output = output[:, prompt.size(1) % self.frame_sizes[0]:]
        prior_t = prompt.size(1) - (prompt.size(1) % self.frame_sizes[0])

        # init variables
        fs = [*self.frame_sizes]
        outputs = [None] * (len(fs) - 1)
        # hidden are reset if prompt.size(0) != self.hidden.size(0)
        hiddens = self.hidden
        tiers = self.tiers

        for t in self.generate_tqdm(range(fs[0], n_steps + prior_t)):
            for i in range(len(tiers) - 1):
                if t % fs[i] == 0:
                    inpt = output[:, t - fs[i]:t].unsqueeze(1)

                    if i == 0:
                        prev_out = None
                    else:
                        prev_out = outputs[i - 1][:, (t // fs[i]) % (fs[i - 1] // fs[i])].unsqueeze(1)

                    out, h = tiers[i](inpt, prev_out, hiddens[i])
                    hiddens[i] = h
                    outputs[i] = out
            if t < prior_t:  # only used for warming-up
                continue
            prev_out = outputs[-1]
            inpt = output[:, t - fs[-1]:t].reshape(-1, 1, fs[-1])

            out, _ = tiers[-1](inpt, prev_out[:, (t % fs[-1]) - fs[-1]].unsqueeze(1))
            if temperature is None:
                pred = (nn.Softmax(dim=-1)(out.squeeze(1))).argmax(dim=-1)
            else:
                # great place to implement dynamic cooling/heating !
                pred = torch.multinomial(nn.Softmax(dim=-1)(out.squeeze(1) / temperature), 1)
            output.data[:, t] = pred.squeeze()

        if decode_outputs:
            output = self.decode_outputs(output)

        self.after_generate()

        return output


def make_click_command(f):
    aspec = inspect.getfullargspec(f)
    if aspec.defaults is not None:
        for arg, default in zip(reversed(aspec.args), reversed(aspec.defaults)):
            opt = click.option('--' + arg.replace("_", "-"), default=default)
            f = opt(f)
    return click.command()(f)


@make_click_command
def main(sources=['./gould'], sr=16000, q_levels=256):
    import mimikit as mmk

    net = mmk.SampleRNN(sr=sr, q_levels=q_levels)

    dm = mmk.DataModule(net, "/tmp/srnn.h5",
                             sources=sources, schema=net.db_schema,
                             splits=tuple(),
                             in_mem_data=True)
    trainer = mmk.get_trainer(root_dir='./',
                              max_epochs=32)

    trainer.fit(net, datamodule=dm)


if __name__ == '__main__':
    main()
