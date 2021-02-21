import torch.nn as nn


class TimeUpscalerLinear(nn.Module):

    def __init__(self, in_dim, out_dim, upscaling, **kwargs):
        super(TimeUpscalerLinear, self).__init__()
        self.upscaling = upscaling
        self.out_dim = out_dim
        self.fc = nn.Linear(in_dim, out_dim * upscaling, **kwargs)

    def forward(self, x):
        B, T, _ = x.size()
        return self.fc(x).reshape(B, T * self.upscaling, self.out_dim)


class TimeUpscalerStridedConv1d(nn.Module):

    def __init__(self, c_in, c_out, upscaling, **kwargs):
        super(TimeUpscalerStridedConv1d, self).__init__()
        self.conv = nn.ConvTranspose1d(c_in, c_out, kernel_size=upscaling,
                                       stride=upscaling, **kwargs)

    def forward(self, x):
        return self.conv(x)


class TimeUpscalerPaddedConv1d(nn.Module):

    def __init__(self, c_in, c_out, upscaling, **kwargs):
        super(TimeUpscalerPaddedConv1d, self).__init__()
        self.conv = nn.ConvTranspose1d(c_in, c_out, kernel_size=upscaling,
                                       padding=upscaling-1, **kwargs)

    def forward(self, x):
        return self.conv(x)
