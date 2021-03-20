import torch
import torch.nn as nn
import pytorch_lightning as pl

from ..kit import DBDataset, ShiftedSequences
from ..audios.features import QuantizedSignal

from ..kit import SuperAdam, SequenceModel, DataSubModule

from ..kit.networks.wavenet import WNNetwork


class WaveNetDB(DBDataset):
    qx = None

    @staticmethod
    def extract(path, sr=16000, q_levels=255, emphasis=0.):
        return QuantizedSignal.extract(path, sr, q_levels, emphasis)

    def prepare_dataset(self, model, datamodule):
        prm = model.batch_info()
        self.slicer = ShiftedSequences(len(self.qx), list(zip(prm["shifts"], prm["lengths"])))
        datamodule.loader_kwargs.setdefault("drop_last", False)
        datamodule.loader_kwargs.setdefault("shuffle", True)

    def __getitem__(self, item):
        slices = self.slicer(item)
        return tuple(self.qx[sl].reshape(-1) for sl in slices)

    def __len__(self):
        return len(self.slicer)


class WaveNet(WNNetwork,
              DataSubModule,
              SuperAdam,
              SequenceModel,
              pl.LightningModule):

    @staticmethod
    def loss_fn(output, target):
        criterion = nn.CrossEntropyLoss(reduction="mean")
        return criterion(output.view(-1, output.size(-1)), target.view(-1))

    db_class = WaveNetDB

    def __init__(self,
                 n_layers=(4,),
                 cin_dim=None,
                 gin_dim=None,
                 layers_dim=128,
                 kernel_size=2,
                 groups=1,
                 accum_outputs=0,
                 pad_input=0,
                 skip_dim=None,
                 residuals_dim=None,
                 max_lr=1e-3,
                 betas=(.9, .9),
                 div_factor=3.,
                 final_div_factor=1.,
                 pct_start=.25,
                 cycle_momentum=False,
                 batch_seq_length=64,
                 db=None,
                 batch_size=64,
                 in_mem_data=True,
                 splits=[.8, .2],
                 **loaders_kwargs
                 ):
        super(pl.LightningModule, self).__init__()
        SequenceModel.__init__(self)
        DataSubModule.__init__(self, db, in_mem_data, splits, batch_size=batch_size, **loaders_kwargs)
        SuperAdam.__init__(self, max_lr, betas, div_factor, final_div_factor, pct_start, cycle_momentum)
        self.hparams.q_levels = db.params.qx["q_levels"]
        self.hparams.sr = db.params.qx["sr"]
        self.hparams.emphasis = db.params.qx["emphasis"]
        if hasattr(db, "labels") and cin_dim is not None:
            n_cin_classes = db.params.labels["n_classes"]
        else:
            n_cin_classes = None
        if hasattr(db, "g_labels") and gin_dim is not None:
            n_gin_classes = db.params.g_labels["n_classes"]
        else:
            n_gin_classes = None
        # noinspection PyArgumentList
        WNNetwork.__init__(self, n_layers=n_layers, q_levels=self.hparams.q_levels,
                           n_cin_classes=n_cin_classes, cin_dim=cin_dim,
                           n_gin_classes=n_gin_classes, gin_dim=gin_dim,
                           layers_dim=layers_dim, kernel_size=kernel_size, groups=groups, accum_outputs=accum_outputs,
                           pad_input=pad_input,
                           skip_dim=skip_dim, residuals_dim=residuals_dim)
        self.save_hyperparameters()

    def setup(self, stage: str):
        super().setup(stage)

    def batch_info(self, *args, **kwargs):
        lengths = (self.hparams.batch_seq_length, self.output_shape((-1, self.hparams.batch_seq_length, -1))[1])
        shifts = (0, self.shift)
        return dict(shifts=shifts, lengths=lengths)

    def encode_inputs(self, inputs: torch.Tensor):
        return QuantizedSignal.encode(inputs, self.hparams.q_levels, self.hparams.emphasis)

    def decode_outputs(self, outputs: torch.Tensor):
        return QuantizedSignal.decode(outputs, self.hparams.q_levels, self.hparams.emphasis)

    def generate(self, prompt, n_steps, decode_outputs=False, temperature=None, **kwargs):
        # prepare device, mode and turn off autograd
        self.before_generate()

        output = self.prepare_prompt(prompt, n_steps, at_least_nd=2)
        prior_t = prompt.size(1)

        def predict(outpt, temp):
            if temp is None:
                return nn.Softmax(dim=-1)(outpt).argmax(dim=-1, keepdims=True)
            else:
                return torch.multinomial(nn.Softmax(dim=-1)(outpt / temp), 1)

        inpt = prompt[:, -self.receptive_field:]
        z, cin, gin = self.inpt(inpt, None, None)
        qs = [(z.clone(), None)]
        # initialize queues with one full forward pass
        skips = None
        for layer in self.layers:
            z, _, _, skips = layer((z, cin, gin, skips))
            qs += [(z.clone(), skips.clone() if skips is not None else skips)]

        outpt = self.outpt(skips if skips is not None else z)[:, -1:].squeeze()
        outpt = predict(outpt, temperature)
        output.data[:, prior_t:prior_t+1] = outpt

        qs = {i: q for i, q in enumerate(qs)}

        # disable padding and dilation
        dilations = {}
        for mod in self.modules():
            if isinstance(mod, nn.Conv1d) and mod.dilation != (1,):
                dilations[mod] = mod.dilation
                mod.dilation = (1,)
        for layer in self.layers:
            layer.pad_input = 0

        # cache the indices of the inputs for each layer
        lyr_slices = {}
        for l, layer in enumerate(self.layers):
            d, k = layer.dilation, layer.kernel_size
            rf = d * k
            indices = [qs[l][0].size(2) - 1 - n for n in range(0, rf, d)][::-1]
            lyr_slices[l] = torch.tensor(indices).long().to(self.device)

        for t in self.generate_tqdm(range(prior_t + 1, prior_t + n_steps)):

            x, cin, gin = self.inpt(output[:, t-1:t], None, None)
            q, _ = qs[0]
            q = q.roll(-1, 2)
            q[:, :, -1:] = x
            qs[0] = (q, None)

            for l, layer in enumerate(self.layers):
                z, skips = qs[l]
                zi = torch.index_select(z, 2, lyr_slices[l])
                if skips is not None:
                    # we only need one skip : the first or the last of the kernel's indices
                    i = lyr_slices[l][0 if layer.accum_outputs > 0 else -1].item()
                    skips = skips[:, :, i:i + 1]

                z, _, _, skips = layer((zi, cin, gin, skips))

                if l < len(self.layers) - 1:
                    q, skp = qs[l + 1]

                    q = q.roll(-1, 2)
                    q[:, :, -1:] = z
                    if skp is not None:
                        skp = skp.roll(-1, 2)
                        skp[:, :, -1:] = skips

                    qs[l + 1] = (q, skp)
                else:
                    y, skips = z, skips

            outpt = self.outpt(skips if skips is not None else y).squeeze()
            outpt = predict(outpt, temperature)
            output.data[:, t:t+1] = outpt

        if decode_outputs:
            output = self.decode_outputs(output)

        # reset the layers' parameters
        for mod, d in dilations.items():
            mod.dilation = d
        for layer in self.layers:
            layer.pad_input = self.pad_input

        self.after_generate()

        return output

