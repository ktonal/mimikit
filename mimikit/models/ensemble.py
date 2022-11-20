import h5mapper as h5m
import torch.nn as nn
import torch
import math
from pprint import pprint

from ..networks import TierNetwork
from ..features import Resample, MuLawSignal
from ..loops import GenerateLoop

from .nnn import NearestNextNeighbor
from .checkpoint import Checkpoint

__all__ = [
    "Ensemble"
]


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

    def run_event(self, inputs, net, feature, n_steps, *params):
        prompt_getter = feature.batch_item(
            shift=0, length=inputs.size(-1) if isinstance(net, TierNetwork) else net.rf  # TODO: NNN needs 1 instead of net.rf...
        )
        resample = Resample(self.base_sr, feature.sr)
        n_input_samples = math.ceil(prompt_getter.getter.length * self.base_sr / feature.sr)
        prompt_getter.data = resample(inputs[:, -n_input_samples:])
        prompt = prompt_getter(0)
        output = [[]]
        print("length", prompt_getter.getter.length, "n_inputs", n_input_samples, "PROMPT", prompt.shape)
        def process_outputs(outputs, _):
            print("OUTPUT", outputs[0].shape)
            inv_resample = Resample(feature.sr, self.base_sr)
            # prompt + generated in base_sr :
            y = feature.inverse_transform(outputs[0])
            print("Y", y.shape, feature.sr, self.base_sr)
            out = inv_resample(y)
            output[0] = out[:, n_input_samples:]
            print("OUTPUT", out.shape)

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
    def seconds_to_n_steps(seconds, net, feature):
        return int(seconds * feature.sr) if isinstance(feature, MuLawSignal) \
            else int(seconds * (feature.sr // feature.hop_length)) // getattr(net, "hop", 1)

    def generate_step(self, t, inputs, ctx):
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

    def next_event(self):
        event = next(self.stream)
        if "Checkpoint" in str(event["type"]):
            ck = Checkpoint(event['id'], event['epoch'], event.get("root_dir", "./"))
            net, feature = ck.network, ck.feature
        elif "NearestNextNeighbor" in str(event["type"]):
            feature = event['feature']
            data = event["soundbank"]
            net = NearestNextNeighbor(feature, data.snd)
        else:
            raise TypeError(f"event type '{event['type']}' not recognized")
        n_steps = self.seconds_to_n_steps(event['seconds'], net, feature)
        print("N_STEPS", n_steps)
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
