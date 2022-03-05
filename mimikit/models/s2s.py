import h5mapper as h5m
import torch
import torch.nn as nn

from .io_modules import *
from ..features import Spectrogram, MuLawSignal
from ..networks import Seq2SeqLSTMNetwork as S2SNet, MultiSeq2SeqLSTM
from ..modules import Flatten


__all__ = [
    "Seq2SeqLSTMv0",
    "Seq2SeqLSTM",
    "Seq2SeqMuLaw"
]


class Seq2SeqLSTMv0(S2SNet):
    feature = Spectrogram(sr=22050, n_fft=2048, hop_length=512, coordinate='mag')
    """no input module"""
    def __init__(self,
                 feature=None,
                 output_heads=4,
                 scaled_activation=True,
                 phs="a",
                 **net_hp):
        model_dim = net_hp['model_dim']
        fft_dim = (feature.n_fft // 2 + 1)
        if feature.coordinate == "mag":
            _, outpt_mod = mag_spec_io(fft_dim, model_dim, 1, output_heads, scaled_activation)
        elif feature.coordinate == "pol":
            _, outpt_mod = pol_spec_io(fft_dim//2, model_dim, 1, output_heads, scaled_activation, phs)
            net_hp["input_module"] = Flatten(2)
        else:
            raise ValueError(f"Seq2SeqLSTM doesn't support coordinate of type {feature.coordinate}")
        net_hp["input_dim"] = fft_dim
        net_hp["output_module"] = outpt_mod
        super(Seq2SeqLSTMv0, self).__init__(**net_hp)
        self.hp.feature = self.feature = feature
        self.hp.output_heads = output_heads
        self.hp.scaled_activation = scaled_activation
        self.hp.phs = phs


class Seq2SeqLSTM(S2SNet):
    feature = Spectrogram(sr=22050, n_fft=2048, hop_length=512, coordinate='mag')

    def __init__(self,
                 feature=None,
                 input_heads=1,
                 output_heads=4,
                 scaled_activation=True,
                 phs="a",
                 **net_hp):
        fft_dim = feature.n_fft // 2 + 1
        model_dim = net_hp['model_dim']
        if feature.coordinate == "mag":
            inpt_mod, outpt_mod = mag_spec_io(fft_dim, model_dim, input_heads, output_heads, scaled_activation)
        elif feature.coordinate == "pol":
            inpt_mod, outpt_mod = pol_spec_io(fft_dim, model_dim, input_heads, output_heads, scaled_activation, phs)
        else:
            raise ValueError(f"Seq2SeqLSTM doesn't support coordinate of type {feature.coordinate}")
        net_hp["input_dim"] = model_dim
        net_hp["input_module"] = inpt_mod
        net_hp["output_module"] = outpt_mod
        super(Seq2SeqLSTM, self).__init__(**net_hp)
        self.hp.feature = self.feature = feature
        self.hp.input_heads = input_heads
        self.hp.output_heads = output_heads
        self.hp.scaled_activation = scaled_activation
        self.hp.phs = phs


class Seq2SeqMuLaw(Seq2SeqLSTM):
    feature = MuLawSignal(sr=16000, q_levels=256, normalize=True)

    def __init__(self, feature=None, mlp_dim=128, **net_hp):
        net_hp["input_module"] = feature.input_module(net_hp["input_dim"])
        out_d = net_hp["model_dim"]
        net_hp["output_module"] = feature.output_module(out_d, mlp_dim=mlp_dim)
        super(Seq2SeqMuLaw, self).__init__(**net_hp)
        self.hp.feature = self.feature = feature
        self.hp.mlp_dim = mlp_dim


class MultiSeq2SeqFFT(MultiSeq2SeqLSTM):
    pass


class MultiSeq2SeqMuLaw(MultiSeq2SeqLSTM):
    pass


class NoNANNEt(nn.Module):

    def attach_hooks(self):

        def frw_hook(module, inpt, output):
            is_tensor = lambda x: isinstance(x, torch.Tensor)
            h5m.process_batch(inpt, is_tensor,
                              lambda x: print("INPUT NAN", module) if torch.any(torch.isnan(x)) else None
                              )
            h5m.process_batch(output, is_tensor,
                              lambda x: print("OUTPUT NAN", module) if torch.any(torch.isnan(x)) else None
                              )
            return output

        def grad_hook(module, inpt, output):
            is_tensor = lambda x: isinstance(x, torch.Tensor)
            h5m.process_batch(inpt, is_tensor,
                              lambda x: print("INPUT GRAD NAN", module) if torch.any(torch.isnan(x)) else None
                              )
            h5m.process_batch(output, is_tensor,
                              lambda x: print("OUTPUT GRAD NAN", module) if torch.any(torch.isnan(x)) else None
                              )
            return inpt

        for mod in self.modules():
            mod.register_full_backward_hook(grad_hook)
            mod.register_forward_hook(frw_hook)

        def hook(grad):
            is_nans = torch.isnan(grad)
            if torch.any(is_nans):
                return torch.where(is_nans, torch.zeros_like(grad), grad)
            return grad
        for p in self.parameters():
            p.register_hook(hook)