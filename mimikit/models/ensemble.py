import h5mapper as h5m
import torch.nn as nn
import torch
from pprint import pprint

from ..features import Resample, MuLawSignal
from ..loops import GenerateLoop
from ..checkpoint import Checkpoint
from .nnn import NearestNextNeighbor


__all__ = [
    "Ensemble"
]


class VotingEnsemble(nn.Module):
    device = property(lambda self: next(self.parameters()).device)

    def __init__(self, networks, weights=None):
        super(VotingEnsemble, self).__init__()
        self.nets = nn.ModuleList(networks)
        N = len(networks)
        W = [1/N for _ in range(N)] if weights is None else weights
        if len(W) != N:
            raise ValueError(f"Expected `weights` to be of length {N} but got {len(W)}")
        self.weights = torch.tensor(W) / sum(W)

    def before_generate(self, loop, batch, batch_idx):
        for net in self.nets:
            net.before_generate(loop, batch, batch_idx)
        return {}

    def generate_step(self, t, inputs, ctx):
        self.weights.to(inputs.device)
        out = None
        for w, net in zip(self.nets, self.weights):
            if out is None:
                out = net.generate_step(t, inputs, ctx) * w
            else:
                out += net.generate_step(t, inputs, ctx) * w
        return out

    def after_generate(self, *args, **kwargs):
        for net in self.nets:
            net.after_generate(*args, **kwargs)
        return self


class Ensemble(nn.Module):
    """
    generate form a prompt by chaining checkpoints/models
    """
    device = property(lambda self: next(self.parameters()).device)

    def __init__(self,
                 max_seconds,
                 base_sr,
                 stream,  # a pbind stream of events
                 print_events=False
                 ):
        super(Ensemble, self).__init__()
        self.max_seconds = max_seconds
        self.base_sr = base_sr
        self.stream = stream
        self.print_events = print_events
        # just to make self.device settable/gettable
        self._param = nn.Parameter(torch.ones(1))

    def generate_step(self, t, inputs, ctx):
        """called from outer GenerateLoop"""
        if t >= int(self.max_seconds * self.base_sr):
            return None
        event, net, feature, n_steps, params = self.next_event()
        if hasattr(net, 'to'):
            net = net.to("cuda")
        if hasattr(net, "use_fast_generate"):
            net.use_fast_generate = False

        if (t / self.base_sr + event['seconds']) < self.max_seconds:
            if self.print_events:
                event.update({"start": t / self.base_sr})
                pprint(event)
            out = self.run_event(inputs[0], net, feature, n_steps, *params)
            return out
        return torch.zeros(1, int(self.max_seconds * self.base_sr - t)).to("cuda")

    def run_event(self, inputs, net, feature, n_steps, *params):
        resample = Resample(self.base_sr, feature.sr)
        inputs_resampled = resample(inputs)
        prompt = feature.t(inputs_resampled)
        n_input_samples = inputs.shape[1]

        output = [[]]

        def process_outputs(outputs, _):
            inv_resample = Resample(feature.sr, self.base_sr)
            # prompt + generated in base_sr :
            y = feature.inv(outputs[0])
            out = inv_resample(y)
            output[0] = out[:, n_input_samples:]

        loop = GenerateLoop(
            net,
            dataloader=[(prompt,)],
            inputs=(h5m.Input(None,
                              getter=h5m.AsSlice(dim=1, shift=-net.shift, length=net.shift),
                              setter=h5m.Setter(dim=1)),
                    *tuple(h5m.Input(p, h5m.AsSlice(dim=1 + int(hasattr(net.hp, 'hop')), length=1),
                                     setter=None) for p in params)
                    ),
            n_steps=n_steps,
            add_blank=True,
            process_outputs=process_outputs,
            time_hop=getattr(net, 'hop', 1)
        )
        loop.run()
        return output[0]

    @staticmethod
    def get_network(event):
        return None

    @staticmethod
    def get_feature(event):
        return None

    @staticmethod
    def get_n_steps(event, net, feature):
        return None

    @staticmethod
    def seconds_to_n_steps(seconds, net, feature):
        return int(seconds * feature.sr) if isinstance(feature, MuLawSignal) \
            else int(seconds * (feature.sr // feature.hop_length)) // getattr(net, "hop", 1)

    @staticmethod
    def get_params(event):
        return None

    def next_event(self):
        event = next(self.stream)
        if "Checkpoint" in str(event["type"]):
            ck = Checkpoint(event['id'], event['epoch'], event.get("root_dir", "./"))
            net, feature = ck.network, ck.feature
        elif "NearestNextNeighbor" in str(event["type"]):
            feature = event['feature']
            data = event["soundbank"]
            path_length = event.get("path_length", 10)
            net = NearestNextNeighbor(feature, data.snd, path_length)
        elif event["type"] == "Parallel":
            pass

        else:
            raise TypeError(f"event type '{event['type']}' not recognized")
        n_steps = self.seconds_to_n_steps(event['seconds'], net, feature)
        if "temperature" in event:
            temp = event['temperature']
            if isinstance(temp, float):
                params = (torch.tensor([[temp]]).to(self.device).repeat(1, n_steps),)
            elif isinstance(temp, tuple):
                params = (torch.linspace(temp[0], temp[1], n_steps, device=self.device),)
            else:
                params = tuple()
        else:
            params = tuple()
        return event, net, feature, n_steps, params
