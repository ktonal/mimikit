from typing import Optional, Union, Set, Tuple, Generator

import h5mapper as h5m
import torch.nn as nn
import torch
from pprint import pprint
import dataclasses as dtc

from ..config import NetworkConfig
from ..networks import ARM
from ..features.functionals import Resample
from ..features.item_spec import ItemSpec
from ..loops import GenerateLoop, GenerateLoopV2
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


@dtc.dataclass
class Event:
    generator: Union[ARM, Checkpoint, NearestNextNeighbor]
    seconds: float
    temperature: Optional[float] = None


class Ensemble(ARM):
    """
    generate form a prompt by chaining checkpoints/models
    """
    class Config(NetworkConfig):
        io_spec = None
        max_seconds: float = 10.
        base_sr: int = 22050
        stream: Generator = ()
        print_events: bool = False

    @classmethod
    def from_config(cls, config: Config):
        return Ensemble(config.max_seconds, config.base_sr,
                        config.stream, config.print_events)

    @property
    def config(self) -> NetworkConfig:
        return self.Config()

    @property
    def rf(self):
        return None

    def train_batch(self, item_spec: ItemSpec) -> Tuple[Tuple[h5m.Input, ...], Tuple[h5m.Input, ...]]:
        pass

    def test_batch(self, item_spec: ItemSpec) -> Tuple[Tuple[h5m.Input, ...], Tuple[h5m.Input, ...]]:
        pass

    def before_generate(self, prompts: Tuple[torch.Tensor, ...], batch_index: int) -> None:
        pass

    def after_generate(self, final_outputs: Tuple[torch.Tensor, ...], batch_index: int) -> None:
        pass

    @property
    def generate_params(self) -> Set[str]:
        return {}

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
        event, net, n_steps, params = self.next_event()
        if hasattr(net, 'to'):
            net = net.to("cuda")
        if hasattr(net, "use_fast_generate"):
            net.use_fast_generate = False

        if (t / self.base_sr + event.seconds) < self.max_seconds:
            if self.print_events:
                e = dtc.asdict(event)
                e.update({"start": t / self.base_sr})
                pprint(e)
            out = self.run_event(inputs[0], net, n_steps, params)
            return out
        return torch.zeros(1, int(self.max_seconds * self.base_sr - t)).to("cuda")

    def run_event(self, inputs, net, feature, n_steps, params):
        resample = Resample(self.base_sr, feature.sr)
        inputs_resampled = resample(inputs)
        prompt = feature.t(inputs_resampled)
        n_input_samples = inputs.shape[1]

        cfg = GenerateLoopV2.Config(
            parameters=params
        )
        loop = GenerateLoopV2(
            cfg,
            network=net,
            n_steps=n_steps,
            dataloader=[(prompt,)],
            logger=None
        )
        for outputs in loop.run():
            inv_resample = Resample(feature.sr, self.base_sr)
            # prompt + generated in base_sr :
            y = feature.inv(outputs[0])
            out = inv_resample(y)
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
