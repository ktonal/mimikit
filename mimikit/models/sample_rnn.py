from mimikit.kit import TBPTTSampler
from numpy.lib.stride_tricks import as_strided as np_as_strided
from pytorch_lightning import LightningModule
import torch
import torch.nn as nn
import numpy as np
from random import randint

from ..kit import DBDataset
from ..audios.features import QuantizedSignal
from ..kit.ds_utils import ShiftedSequences
from ..kit import SuperAdam, SequenceModel, DataSubModule
from ..kit.networks.sample_rnn import SampleRNNNetwork

from . import model


class FramesDB(DBDataset):
    qx = None

    @staticmethod
    def extract(path, sr=16000, q_levels=255, emphasis=0.):
        return QuantizedSignal.extract(path, sr, q_levels, emphasis)

    def prepare_dataset(self, model, datamodule):
        batch_size, chunk_len, batch_seq_len, frame_sizes = model.batch_info()
        shifts = [frame_sizes[0] - size for size in frame_sizes + (0,)]  # (0,) for the target
        lengths = [batch_seq_len for _ in frame_sizes[:-1]]
        lengths += [frame_sizes[0] + batch_seq_len]
        lengths += [batch_seq_len]
        self.slicer = ShiftedSequences(len(self.qx), list(zip(shifts, lengths)))
        self.frame_sizes = frame_sizes
        self.seq_len = batch_seq_len

        # the slicer knows how many batches it can build, so we pass its length to the sampler
        batch_sampler = TBPTTSampler(len(self.slicer),
                                     batch_size,
                                     chunk_len,
                                     batch_seq_len)
        datamodule.loader_kwargs.update(dict(batch_sampler=batch_sampler))
        for k in ["batch_size", "shuffle", "drop_last"]:
            if k in datamodule.loader_kwargs:
                datamodule.loader_kwargs.pop(k)
        datamodule.loader_kwargs["sampler"] = None

    def __getitem__(self, item):
        if type(self.qx) is not torch.Tensor:
            itemsize = self.qx.dtype.itemsize
            as_strided = lambda slc, fs: np_as_strided(self.qx[slc],
                                                       shape=(self.seq_len, fs),
                                                       strides=(itemsize, itemsize))
        else:
            as_strided = lambda slc, fs: torch.as_strided(self.qx[slc],
                                                          size=(self.seq_len, fs),
                                                          stride=(1, 1))

        slices = self.slicer(item)
        tiers_slc, bottom_slc, target_slc = slices[:-2], slices[-2], slices[-1]
        inputs = [self.qx[slc].reshape(-1, fs) for slc, fs in zip(tiers_slc, self.frame_sizes[:-1])]
        # ugly but necessary if self.qx became a tensor...
        with torch.no_grad():
            inputs += [as_strided(bottom_slc, self.frame_sizes[-1])]
        target = self.qx[target_slc]

        return tuple(inputs), target

    def __len__(self):
        return len(self.qx)


@model
class SampleRNN:
    __parts__ = (SequenceModel,
                 DataSubModule,
                 SuperAdam,
                 SampleRNNNetwork)

    @staticmethod
    def loss_fn(output, target):
        criterion = nn.CrossEntropyLoss(reduction="mean")
        return {"loss": criterion(output.view(-1, output.size(-1)), target.view(-1))}

    db_class = FramesDB

    def batch_info(self, *args, **kwargs):
        return tuple(self.hparams[key] for key in ["batch_size", "chunk_len", "batch_seq_len", "frame_sizes"])

    def setup(self, stage: str):
        SuperAdam.setup(self, stage)

    def on_train_batch_start(self, batch, batch_idx, dataloader_idx):
        if self.hparams.reset_hidden and (batch_idx * self.hparams.batch_seq_len) % self.hparams.chunk_len == 0:
            self.reset_h0()

    def encode_inputs(self, inputs: torch.Tensor):
        # TODO : automatically get from property self.input_feature = {db_key: Feature(), ...}
        return QuantizedSignal.encode(inputs, self.hparams.q_levels, self.hparams.emphasis)

    def decode_outputs(self, outputs: torch.Tensor):
        # TODO : automatically get from property self.output_feature = (Feature(), ...)
        return QuantizedSignal.decode(outputs, self.hparams.q_levels, self.hparams.emphasis)

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

    def load_state_dict(self, state_dict, strict=True):
        to_pop = [k for k in state_dict.keys() if "h0" in k or "c0" in k]
        for k in to_pop:
            state_dict.pop(k)
        return super(SampleRNN, self).load_state_dict(state_dict, strict=False)


sp = SampleRNN()