import torch
import torch.nn as nn
from mimikit.kit.networks.parametrized_gaussian import ParametrizedGaussian


class EncoderLSTM(nn.Module):
    def __init__(self, input_d, model_dim, num_layers, bottleneck="add", n_fc=1):
        super(EncoderLSTM, self).__init__()
        self.bottleneck = bottleneck
        self.dim = model_dim
        self.lstm = nn.LSTM(input_d, self.dim if bottleneck == "add" else self.dim // 2, bias=False,
                            num_layers=num_layers, batch_first=True, bidirectional=True)
        self.fc = nn.Sequential(
            *[nn.Sequential(nn.Linear(self.dim, self.dim), nn.Tanh()) for _ in range(n_fc - 1)],
            nn.Linear(self.dim, self.dim, bias=False),  # NO ACTIVATION !
        )
#         for name, p in dict(self.lstm.named_parameters()).items():
#             if "weight" in name:
#                 torch.nn.utils.weight_norm(self.lstm, name)

    def forward(self, x, hiddens=None, cells=None):
        if hiddens is None or cells is None:
            states, (hiddens, cells) = self.lstm(x)
        else:
            states, (hiddens, cells) = self.lstm(x, (hiddens, cells))
        states = self.first_and_last_states(states)
        return self.fc(states), (hiddens, cells)

    def first_and_last_states(self, sequence):
        sequence = sequence.view(*sequence.size()[:-1], self.dim, 2).sum(dim=-1)
        first_states = sequence[:, 0, :]
        last_states = sequence[:, -1, :]
        if self.bottleneck == "add":
            return first_states + last_states
        else:
            return torch.cat((first_states, last_states), dim=-1)


class DecoderLSTM(nn.Module):
    def __init__(self, model_dim, num_layers, bottleneck="add"):
        super(DecoderLSTM, self).__init__()
        self.dim = model_dim
        self.lstm1 = nn.LSTM(self.dim, self.dim if bottleneck == "add" else self.dim // 2, bias=False,
                             num_layers=num_layers, batch_first=True, bidirectional=True)
        self.lstm2 = nn.LSTM(self.dim, self.dim if bottleneck == "add" else self.dim // 2, bias=False,
                             num_layers=num_layers, batch_first=True, bidirectional=True)
        for name, p in dict(self.lstm1.named_parameters()).items():
            if "weight" in name:
                torch.nn.utils.weight_norm(self.lstm1, name)
#         for name, p in dict(self.lstm2.named_parameters()).items():
#             if "weight" in name:
#                 torch.nn.utils.weight_norm(self.lstm2, name)

    def forward(self, x, hiddens, cells):
        if hiddens is None or cells is None:
            output, (_, _) = self.lstm1(x)
        else:
            # decoder does get hidden states from encoder !
            output, (_, _) = self.lstm1(x, (hiddens, cells))
        # sum forward and backward nets
        output = output.view(*output.size()[:-1], self.dim, 2).sum(dim=-1)

        if hiddens is None or cells is None:
            output2, (hiddens, cells) = self.lstm2(output)
        else:
            # V1 residual does get hidden states from encoder !
            output2, (hiddens, cells) = self.lstm2(output, (hiddens, cells))
        # sum forward and backward nets
        output2 = output2.view(*output2.size()[:-1], self.dim, 2).sum(dim=-1)

        # sum the outputs
        return output + output2, (hiddens, cells)


class Seq2SeqLSTM(nn.Module):
    def __init__(self,
                 input_dim,
                 model_dim,
                 num_layers=1,
                 bottleneck="add",
                 n_fc=1):
        super(Seq2SeqLSTM, self).__init__()
        self.enc = EncoderLSTM(input_dim, model_dim, num_layers, bottleneck, n_fc)
        self.dec = DecoderLSTM(model_dim, num_layers, bottleneck)
        self.sampler = ParametrizedGaussian(model_dim, model_dim)
        self.fc_out = nn.Linear(model_dim, input_dim, bias=False)

    def forward(self, x, output_length=None):
        coded, (h_enc, c_enc) = self.enc(x)
        if output_length is None:
            output_length = x.size(1)
        coded = coded.unsqueeze(1).repeat(1, output_length, 1)
        residuals, _, _ = self.sampler(coded)
        coded = coded + residuals
        output, (_, _) = self.dec(coded, h_enc, c_enc)
        return self.fc_out(output).abs()
