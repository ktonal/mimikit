from typing import Optional, Union, Generator

import torch.nn as nn
import torch
from pprint import pprint
import dataclasses as dtc

from ..utils import default_device
from ..networks import ARM
from ..features.functionals import Resample
from ..loops import GenerateLoopV2
from ..checkpoint import Checkpoint
from .nnn import NearestNextNeighbor

__all__ = [
    "EnsembleGenerator"
]


class VotingEnsemble(nn.Module):
    device = property(lambda self: next(self.parameters()).device)

    def __init__(self, networks, weights=None):
        super(VotingEnsemble, self).__init__()
        self.nets = nn.ModuleList(networks)
        N = len(networks)
        W = [1 / N for _ in range(N)] if weights is None else weights
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


@dtc.dataclass
class Event:
    generator: Union[ARM, Checkpoint, NearestNextNeighbor]
    seconds: float
    temperature: Optional[float] = None


class EnsembleGenerator:
    """
    generate form a prompt by chaining checkpoints/models
    """

    def __init__(self,
                 prompt: torch.Tensor,
                 max_seconds: float = 10.,
                 base_sr: int = 22050,
                 stream: Generator = (),
                 print_events: bool = False,
                 device=default_device()
                 ):
        super(EnsembleGenerator, self).__init__()
        self.prompt = prompt.to(device)
        self.max_seconds = max_seconds
        self.base_sr = base_sr
        self.stream = stream
        self.print_events = print_events
        self.device = device

    def run(self):
        prompt_length = t = self.prompt.size(-1)
        n_samples = int(self.max_seconds * self.base_sr)
        output = torch.zeros(self.prompt.size(0), n_samples,
                             dtype=self.prompt.dtype).to(self.device)
        output[:, :t] = self.prompt
        while t < n_samples:
            prompt = output[:, t-prompt_length:t]
            step_output = self.generate_step(t, prompt)
            if step_output is None:
                break
            output[:, t:t+step_output.size(1)] = step_output
            t += step_output.size(1)
        return output

    def generate_step(self, t, inputs):
        if t >= int(self.max_seconds * self.base_sr):
            return None
        event, net, n_steps, params = self.next_event()
        if hasattr(net, 'to'):
            net = net.to(self.device)

        if (t / self.base_sr + event.seconds) < self.max_seconds:
            if self.print_events:
                e = dtc.asdict(event)
                e.update({"start": t / self.base_sr})
                pprint(e)
            out = self.run_event(inputs, net, n_steps, params)
            return out
        return torch.zeros(inputs.size(0), int(self.max_seconds * self.base_sr - t)).to(self.device)

    def run_event(self,
                  inputs: torch.Tensor,
                  net: ARM,
                  n_steps: int,
                  params: dict
                  ):
        network_sr = net.config.io_spec.sr
        resample = Resample(self.base_sr, network_sr)
        inputs_resampled = resample(inputs)
        prompt = tuple(in_spec.transform(inputs_resampled) for in_spec in net.config.io_spec.inputs)
        n_input_samples = inputs.shape[1]

        cfg = GenerateLoopV2.Config(
            parameters=params,
            display_waveform=False,
            write_waveform=False,
            yield_inversed_outputs=True
        )
        loop = GenerateLoopV2(
            cfg,
            network=net,
            n_steps=n_steps,
            # the loop needs the indices of the prompt before the prompt for logging...
            dataloader=[[torch.ones(1), *prompt]],
            logger=None,
        )
        for outputs in loop.run():
            inv_resample = Resample(network_sr, self.base_sr)
            # prompt + generated in base_sr :
            out = inv_resample(outputs[0])
            return out[:, n_input_samples:]

    def next_event(self):
        event = Event(**next(self.stream))
        if isinstance(event.generator, Checkpoint):
            ck = event.generator
            net = ck.network
        elif isinstance(event.generator, NearestNextNeighbor):
            net = event.generator
        # elif event["type"] == "Parallel":
        #     pass
        else:
            raise TypeError(f"event generator type '{type(event.generator)}' not supported")
        cfg = GenerateLoopV2.Config(output_duration_sec=event.seconds)
        n_steps = GenerateLoopV2.get_n_steps(cfg, net)
        if event.temperature is not None:
            params = dict(temperature=event.temperature)
        else:
            params = {}
        return event, net, n_steps, params
