import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from functools import reduce
import operator as op


class ConvTranspose1dWaveGanGen(nn.Module):
    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            up_sample=4,
            up_sampling="conv",
            use_batch_norm=False,
    ):
        super(ConvTranspose1dWaveGanGen, self).__init__()
        self.up_sample = up_sample
        self.up_sampling = up_sampling

        if up_sampling == "nearest":
            reflection_pad = nn.ReflectionPad1d(
                (kernel_size // 2) if kernel_size % 2 == 1 else (kernel_size // 2 - 1, kernel_size // 2)
            )
            conv1d = nn.Conv1d(in_channels, out_channels, kernel_size, stride=(1,))
            ops = [reflection_pad, conv1d]
        else:
            Conv1dTrans = nn.ConvTranspose1d(
                in_channels, out_channels, kernel_size, stride=(up_sample,),
                padding=(math.ceil((kernel_size - up_sample) / 2) if kernel_size > up_sample else 0,),
                output_padding=(up_sample - kernel_size if kernel_size < up_sample else int(kernel_size % 2 == 1),)
            )
            ops = [Conv1dTrans]

        if use_batch_norm:
            ops.append(nn.BatchNorm1d(out_channels))

        self.ops = nn.Sequential(*ops)

    def forward(self, x):
        if self.up_sampling == "nearest":
            # recommended by wavgan paper to use nearest up_sampling
            x = nn.functional.interpolate(x, scale_factor=self.up_sample, mode="nearest")
        return self.ops(x)


class WaveGANGenerator(nn.Module):
    def __init__(
            self,
            latent_dim=100,
            model_size=64,
            kernel_size=25,
            t0=16,
            up_sample=(4, ),
            up_sampling="conv",  # or "nearest"
            n_layers=5,
            n_channels=1,
            post_proc_filt_len=512,
            use_batch_norm=False,
            verbose=False,
    ):
        super(WaveGANGenerator, self).__init__()
        self.latent_dim = latent_dim
        self.model_size = model_size  # d
        self.kernel_size = kernel_size
        self.t0 = t0
        self.n_channels = n_channels  # c
        self.post_proc_filt_len = post_proc_filt_len
        self.use_batch_norm = use_batch_norm
        self.verbose = verbose
        self.dim_mul = dim_mul = reduce(op.mul, up_sample)
        self.fc1 = nn.Linear(latent_dim, model_size * dim_mul * t0)
        self.bn1 = nn.BatchNorm1d(num_features=model_size * dim_mul) if use_batch_norm else None

        deconv_layers = [
            ConvTranspose1dWaveGanGen(
                (dim_mul * model_size) // (2 ** downscaling_factor),
                (dim_mul * model_size) // (2 ** (downscaling_factor + 1)),
                kernel_size,
                up_sample=up,
                up_sampling=up_sampling,
                use_batch_norm=use_batch_norm,
            ) for up, downscaling_factor in zip(up_sample, range(n_layers - 1))] + \
            [
                ConvTranspose1dWaveGanGen(
                    (dim_mul * model_size) // (2 ** (n_layers - 1)),
                    n_channels,
                    kernel_size,
                    up_sample=up_sample[-1],
                    up_sampling=up_sampling,
                )
            ]

        self.deconv_list = nn.ModuleList(deconv_layers)
        self.ppfilter1 = nn.Conv1d(n_channels, n_channels, (post_proc_filt_len,)) if post_proc_filt_len else None

        for m in self.modules():
            if isinstance(m, nn.ConvTranspose1d) or isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight.data)

    def forward(self, x):
        x = self.fc1(x).view(-1, self.dim_mul * self.model_size, self.t0)
        if self.use_batch_norm:
            x = self.bn1(x)
        x = F.relu(x)
        if self.verbose:
            print(self.fc1.__class__.__qualname__, x.shape)

        for deconv in self.deconv_list[:-1]:
            x = F.relu(deconv(x))
            if self.verbose:
                print(deconv.__class__.__qualname__, x.shape)
        if self.ppfilter1 is None:
            output = torch.tanh(self.deconv_list[-1](x))
        else:
            x = F.relu(self.deconv_list[-1](x))
            # Pad for "same" filtering
            if (self.post_proc_filt_len % 2) == 0:
                pad_left = self.post_proc_filt_len // 2
                pad_right = pad_left - 1
            else:
                pad_left = (self.post_proc_filt_len - 1) // 2
                pad_right = pad_left
            x = F.pad(x, (pad_left, pad_right))
            output = torch.tanh(self.ppfilter1(x))
        return output


class PhaseShuffle(nn.Module):
    """
    Performs phase shuffling, i.e. shifting feature axis of a 3D tensor
    by a random integer in {-n, n} and performing reflection padding where
    necessary.
    """

    # Copied from https://github.com/jtcramer/wavegan/blob/master/wavegan.py#L8
    def __init__(self, shift_factor):
        super(PhaseShuffle, self).__init__()
        self.shift_factor = shift_factor

    def forward(self, x):
        if self.shift_factor == 0:
            return x
        # uniform in (L, R)
        k_list = (
                torch.Tensor(x.shape[0]).random_(0, 2 * self.shift_factor + 1)
                - self.shift_factor
        )
        k_list = k_list.numpy().astype(int)

        # Combine sample indices into lists so that less shuffle operations
        # need to be performed
        k_map = {}
        for idx, k in enumerate(k_list):
            k = int(k)
            if k not in k_map:
                k_map[k] = []
            k_map[k].append(idx)

        # Make a copy of x for our output
        x_shuffle = x.clone()

        # Apply shuffle to each sample
        for k, idxs in k_map.items():
            if k > 0:
                x_shuffle[idxs] = F.pad(x[idxs][..., :-k], (k, 0), mode="reflect")
            else:
                x_shuffle[idxs] = F.pad(x[idxs][..., -k:], (0, -k), mode="reflect")

        assert x_shuffle.shape == x.shape, "{}, {}".format(x_shuffle.shape, x.shape)
        return x_shuffle


class Conv1DWaveGanDisc(nn.Module):
    def __init__(
            self,
            input_channels,
            output_channels,
            kernel_size,
            down_sample=4,
            alpha=0.2,
            shift_factor=2,

            use_batch_norm=False,
            drop_prob=0,
    ):
        super(Conv1DWaveGanDisc, self).__init__()
        self.conv1d = nn.Conv1d(
            input_channels, output_channels, kernel_size, stride=(down_sample,),
            padding=(math.ceil((kernel_size - down_sample) / 2) if kernel_size > down_sample else 0,)
        )
        self.batch_norm = nn.BatchNorm1d(output_channels)
        self.phase_shuffle = PhaseShuffle(shift_factor)
        self.alpha = alpha
        self.use_batch_norm = use_batch_norm
        self.use_phase_shuffle = shift_factor == 0
        self.use_drop = drop_prob > 0
        self.dropout = nn.Dropout2d(drop_prob)

    def forward(self, x):
        x = self.conv1d(x)
        if self.use_batch_norm:
            x = self.batch_norm(x)
        x = F.leaky_relu(x, negative_slope=self.alpha)
        if self.use_phase_shuffle:
            x = self.phase_shuffle(x)
        if self.use_drop:
            x = self.dropout(x)
        return x


class WaveGANDiscriminator(nn.Module):
    def __init__(
            self,
            model_size=64,
            kernel_size=25,
            down_sample=(4, ),
            n_layers=5,
            n_channels=1,
            shift_factor=2,
            alpha=0.2,
            verbose=False,
            use_batch_norm=False,
    ):
        super(WaveGANDiscriminator, self).__init__()

        self.model_size = model_size  # d
        self.use_batch_norm = use_batch_norm
        self.n_channels = n_channels  # c
        self.shift_factor = shift_factor  # n
        self.alpha = alpha
        self.verbose = verbose

        conv_layers = [
            Conv1DWaveGanDisc(
                n_channels,
                model_size,
                kernel_size,
                down_sample=down_sample[0],
                use_batch_norm=use_batch_norm,
                alpha=alpha,
                shift_factor=shift_factor,
            )
            ] + [
            Conv1DWaveGanDisc(
                model_size * (2 ** up_scaling_factor),
                model_size * (2 ** (up_scaling_factor + 1)),
                kernel_size,
                down_sample=down,
                use_batch_norm=use_batch_norm,
                alpha=alpha,
                shift_factor=shift_factor,
            )
            for down, up_scaling_factor in zip(down_sample, range(n_layers-1))]
        # self.fc_input_size = (down_sample ** (n_layers-1)) * model_size
        self.fc_input_size = reduce(op.mul, down_sample)

        self.conv_layers = nn.ModuleList(conv_layers)

        self.fc1 = nn.Linear(self.fc_input_size, 1)

        for m in self.modules():
            if isinstance(m, nn.Conv1d) or isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight.data)

    def forward(self, x):
        for conv in self.conv_layers:
            x = conv(x)
            if self.verbose:
                print(conv.__class__.__qualname__, x.shape)
        x = x.view(-1, self.fc_input_size)
        if self.verbose:
            print(self.fc1.__class__.__qualname__, x.shape)

        return self.fc1(x)
