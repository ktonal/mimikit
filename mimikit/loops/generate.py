from typing import Optional, Any, Callable, Tuple, Iterable
import numpy as np
import torch
import h5mapper as h5m

from .callbacks import tqdm

__all__ = [
    'GenerateLoop',
    'prepare_prompt',
    'generate_tqdm',
]


def prepare_prompt(device, prompt, n_blanks, at_least_nd=2):
    def _prepare(prmpt):
        if isinstance(prmpt, np.ndarray):
            prmpt = torch.from_numpy(prmpt)
        while len(prmpt.shape) < at_least_nd:
            prmpt = prmpt.unsqueeze(0)
        prmpt = prmpt.to(device)
        if n_blanks > 0:
            blank_shapes = prmpt.size(0), n_blanks, *prmpt.size()[2:]
            return torch.cat((prmpt, torch.zeros(*blank_shapes).to(prmpt)), dim=1)
        else:
            return prmpt

    return h5m.process_batch(prompt, lambda x: isinstance(x, (np.ndarray, torch.Tensor)), _prepare)


def generate_tqdm(rng):
    return tqdm(rng, desc="Generate", dynamic_ncols=True,
                leave=False, unit="step", mininterval=0.25)


class GenInput(h5m.Input):
    def __init__(self, data=None, length=1, transform=lambda x: x):
        super(GenInput, self).__init__(
            data=data, getter=h5m.AsSlice(dim=1, shift=-length, length=length),
            setter=h5m.Setter(dim=1),
            transform=transform
        )


class Parameter(h5m.Input):
    def __init__(self, data):
        super(Parameter, self).__init__(
            data=data, getter=h5m.AsSlice(dim=1, length=1)
        )


class GenerateLoop:
    """
    interfaces' length must equal the number of items in the batches of the DataLoader
    if an interfaces has None as setter, it is considered a parameter, not a "generable"
    """

    def __init__(self,
                 network: torch.nn.Module = None,
                 dataloader: torch.utils.data.dataloader.DataLoader = None,
                 inputs: Iterable[h5m.Input] = tuple(),
                 n_batches: Optional[int] = None,
                 n_steps: int = 1,
                 time_hop: int = 1,
                 disable_grads: bool = True,
                 device: str = 'cuda:0',
                 process_outputs: Callable[[Tuple[Any], int], None] = lambda x, i: None,
                 add_blank=True,
                 ):
        self.net = network
        self.dataloader = dataloader
        self.inputs = inputs
        self.n_batches = n_batches
        self.n_steps = n_steps
        self.time_hop = time_hop
        self.disable_grads = disable_grads
        self.device = device
        self.process_outputs = process_outputs
        self.add_blank = add_blank
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

            prior_t = len(batch[0][0]) if self.add_blank else getattr(self.net, 'shift', 0)
            inputs_itf = []
            for x, interface in zip(batch + ((None,) * (len(self.inputs) - len(batch))), self.inputs):
                if isinstance(x, torch.Tensor):
                    x = prepare_prompt(self.device, x,
                                       self.n_steps * self.time_hop if self.add_blank else 0,
                                       len(x.shape))
                    interface.data = x
                    inputs_itf += [interface]
                elif x is None and interface.data is not None:  # e.g. parameter
                    inputs_itf += [interface]

            outputs_itf = tuple(interface for interface in self.inputs if interface.setter is not None)
            # generate
            until = 0
            for t in generate_tqdm(range(0, self.n_steps * self.time_hop, self.time_hop)):
                if t < until:
                    continue
                inputs = tuple(input(t + (prior_t if input.setter is not None else 0)) for input in inputs_itf)
                outputs = self.net.generate_step(t+prior_t, inputs, ctx)
                if not isinstance(outputs, tuple):
                    outputs = outputs,
                for interface, out in zip(outputs_itf, outputs):
                    # let the net return None when ignoring this step
                    if out is not None:
                        until = interface.set(t+prior_t, out) + t

            # wrap up
            final_outputs = tuple(x.data for x in inputs_itf[:len(outputs_itf)])
            if getattr(self.net, 'after_generate', False):
                self.net.after_generate(final_outputs, ctx, batch_idx)
            else:
                pass

            self.process_outputs(final_outputs, batch_idx)

        self.teardown()
