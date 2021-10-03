from typing import Optional, Any, Callable, Tuple, Iterable
import numpy as np
import torch
import dataclasses as dtc
from tqdm import tqdm
from h5mapper import AsSlice, Getter, process_batch


__all__ = [
    'DynamicDataInterface',
    'GenerateLoop',
    'Setter',
    'prepare_prompt',
    'generate_tqdm',
]


def prepare_prompt(device, prompt, n_steps, at_least_nd=2):
    def _prepare(prmpt):
        if isinstance(prmpt, np.ndarray):
            prmpt = torch.from_numpy(prmpt)
        while len(prmpt.shape) < at_least_nd:
            prmpt = prmpt.unsqueeze(0)
        prmpt = prmpt.to(device)
        blank_shapes = prmpt.size(0), n_steps, *prmpt.size()[2:]
        return torch.cat((prmpt, torch.zeros(*blank_shapes).to(prmpt)), dim=1)

    return process_batch(prompt, lambda x: isinstance(x, (np.ndarray, torch.Tensor)), _prepare)


def generate_tqdm(rng):
    return tqdm(rng, desc="Generate", dynamic_ncols=True,
                leave=False, unit="step", mininterval=0.25)


@dtc.dataclass
class Setter:

    dim: int = 0

    def __post_init__(self):
        self.pre_slices = (slice(None),) * self.dim

    def __call__(self, data, item, value):
        data.data[self.pre_slices + (slice(item, item + value.shape[self.dim]),)] = value
        return data


# noinspection PyArgumentList
@dtc.dataclass
class DynamicDataInterface:
    source: Optional[Any] = None

    prepare: Callable[[Any], Any] = lambda src: None

    getter: Getter = AsSlice(shift=-1, length=1)
    input_transform: Optional[Callable] = lambda x: x

    output_transform: Optional[Callable] = lambda x: x
    setter: Optional[Setter] = None

    def __post_init__(self):
        result = self.prepare(self.source)
        if result is not None:
            self.source = result

    def get(self, t):
        return self.input_transform(self.getter(self.source, t))

    def set(self, t, value):
        return self.setter(self.source, t, self.output_transform(value))

    def wrap(self, source):
        return DynamicDataInterface(source, self.prepare, self.getter, self.input_transform,
                                    self.output_transform, self.setter)


class GenerateLoop:
    """
    interfaces' length must equal the number of items in the batches of the DataLoader
    if an interfaces has None as setter, it is considered a parameter, not a "generable"
    """

    def __init__(self,
                 network: torch.nn.Module = None,
                 dataloader: torch.utils.data.dataloader.DataLoader = None,
                 interfaces: Iterable[DynamicDataInterface] = tuple(),
                 n_batches: Optional[int] = None,
                 n_steps: int = 1,
                 time_hop: int = 1,
                 disable_grads: bool = True,
                 device: str = 'cuda:0',
                 process_outputs: Callable[[Tuple[Any], int], None] = lambda x, i: None
                 ):
        self.net = network
        self.dataloader = dataloader
        self.interfaces = interfaces
        self.n_batches = n_batches
        self.n_steps = n_steps
        self.time_hop = time_hop
        self.disable_grads = disable_grads
        self.device = device
        self.process_outputs = process_outputs

        self._was_training = False
        self._initial_device = device

    def setup(self):
        net = self.net
        self._was_training = net.training
        self._initial_device = net.device
        net.eval()
        net.to(self.device if 'cuda' in self.device and torch.cuda.is_available()
               else "cpu")
        if self.disable_grads:
            torch.set_grad_enabled(False)

    def teardown(self):
        self.net.to(self._initial_device)
        self.net.train() if self._was_training else None
        if self.disable_grads:
            torch.set_grad_enabled(True)

    def run(self):

        self.setup()

        if self.n_batches is not None:
            epoch_iterator = zip(range(self.n_batches), self.dataloader)
        else:
            epoch_iterator = enumerate(self.dataloader)

        for batch_idx, batch in epoch_iterator:
            # prepare
            batch = tuple(x.to(self.device) for x in batch)
            if getattr(self.net, 'before_generate', False):
                ctx = self.net.before_generate(self, batch, batch_idx)
            else:
                ctx = {}
            prior_t = len(batch[0][0])
            inputs_itf = []
            for x, interface in zip(batch + ((None,) * (len(self.interfaces) - len(batch))),
                                    self.interfaces):
                if isinstance(x, torch.Tensor):
                    x = prepare_prompt(self.device, x, self.n_steps, len(x.shape))
                inputs_itf += [interface.wrap(x)]
            outputs_itf = tuple(x for x in inputs_itf if x.setter is not None)

            # generate
            for t in generate_tqdm(range(0, self.n_steps, self.time_hop)):
                inputs = tuple(x.get(t + (prior_t if x.setter is not None else 0))
                               for x in inputs_itf)
                outputs = self.net.generate_step(t+prior_t, inputs, ctx)
                if not isinstance(outputs, tuple):
                    outputs = outputs,
                for x, out in zip(outputs_itf, outputs):
                    x.set(t+prior_t, out)

            # wrap up
            final_outputs = tuple(x.source for x in outputs_itf)
            if getattr(self.net, 'after_generate', False):
                self.net.after_generate(final_outputs, ctx, batch_idx)
            else:
                pass

            self.process_outputs(final_outputs, batch_idx)

        self.teardown()
