from typing import Optional, Any, Callable, Tuple, Iterable, Dict, overload, Sized
import numpy as np
import torch
import h5mapper as h5m

from .callbacks import tqdm

__all__ = [
    'GenerateLoop',
    'prepare_prompt',
    'generate_tqdm',
]

from .logger import AudioLogger

from ..config import Config
from ..networks.arm import ARM
from .samplers import IndicesSampler


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


class GenerateLoop:
    """
    interfaces' length must equal the number of items in the batches of the DataLoader
    if an interfaces has None as setter, it is considered a parameter, not a "generable"
    """

    class Config(Config):
        output_duration_sec: float = 1.
        prompts_length_sec: float = 1.
        prompts_position_sec: Tuple[Optional[float]] = (None,)  # random if None
        parameters: Optional[Dict[str, Any]] = None
        batch_size: int = 1

        output_file_template: Optional[str] = None
        display_waveform: bool = True
        # either GenerateLoop exposes self.add_callback(...)
        # or evaluators must wrap networks and implement after_generate...
        # or the loop changes to run(prompts, **parameters) -> outputs
        # maybe even:
        # for prompt in loop.prompts:
        #    output = loop.run(prompt, **parameters)
        #    logger.write(output)
        # --> finer factoring...

    @classmethod
    def from_config(cls, config: Config, soundbank, network: ARM):
        prompt_batch, _ = network.train_batch(config.prompts_length_sec, "second")
        max_len = max(inpt.length for inpt in prompt_batch)
        sr = soundbank.snd.sr
        indices = tuple(int(x * sr) if x is not None else x for x in config.prompts_position_sec)
        dataloader = soundbank.serve(
            prompt_batch,
            sampler=IndicesSampler(N=len(indices),
                                   indices=indices,
                                   max_i=soundbank.snd.shape[0] - max_len,
                                   redraw=True),
            shuffle=False,
            batch_size=config.batch_size
        )
        targets = network.config.io_spec.targets
        n_steps = {t.feature.seconds_to_steps(config.output_duration_sec) for t in targets}
        assert len(n_steps) == 1
        n_steps = n_steps.pop()
        inputs_interfaces = None
        targets_interfaces = None
        parameters = None

        logger = AudioLogger(file_template=config.output_file_template,
                             title_template=config.output_file_template)

    def __init__(self,
                 # config: "GenerateLoop.Config",
                 network: ARM = None,
                 dataloader: torch.utils.data.dataloader.DataLoader = None,
                 inputs: Iterable[h5m.Input] = tuple(),
                 n_batches: Optional[int] = None,
                 n_steps: int = 1,
                 time_hop: int = 1,
                 disable_grads: bool = True,
                 device: str = 'cuda:0',
                 process_outputs: Callable[[Tuple[Any], int], None] = lambda x, i: None,
                 ):
        # self.config = config
        self.net: ARM = network
        self.dataloader = dataloader
        self.inputs = inputs
        self.n_batches = n_batches
        self.n_steps = n_steps
        self.time_hop = time_hop
        self.disable_grads = disable_grads
        self.device = device
        self.process_outputs = process_outputs
        self._was_training = network.training
        self._initial_device = network.device

    def setup(self):
        net = self.net
        self._initial_device = net.device
        self._was_training = net.training
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

    def _process_outputs(self, final_outputs, batch_idx):
        for output in final_outputs:
            if self.config.output_file_template:
                self.logger.write_batch(output, batch_idx=batch_idx)
            if self.config.display_waveform:
                self.logger.display_batch(output, batch_idx=batch_idx)

    def run_epoch(self):
        pass

    def run(self):

        self.setup()

        if self.n_batches is not None:
            epoch_iterator = zip(range(self.n_batches), self.dataloader)
        else:
            epoch_iterator = enumerate(self.dataloader)

        for batch_idx, batch in epoch_iterator:
            # prepare
            batch = tuple(x.to(self.device) for x in batch)
            self.net.before_generate(batch, batch_idx)

            prior_t = len(batch[0][0])
            inputs_itf = []
            for x, interface in zip(batch + ((None,) * (len(self.inputs) - len(batch))), self.inputs):
                if isinstance(x, torch.Tensor):
                    x = prepare_prompt(self.device, x,
                                       self.n_steps * self.time_hop,
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
                outputs = self.net.generate_step(inputs, t=t + prior_t)
                if not isinstance(outputs, tuple):
                    outputs = outputs,
                for interface, out in zip(outputs_itf, outputs):
                    # let the net return None when ignoring this step
                    if out is not None:
                        until = interface.set(t + prior_t, out) + t

            # wrap up
            final_outputs = tuple(x.data for x in inputs_itf[:len(outputs_itf)])
            self.net.after_generate(final_outputs, batch_idx)

            self.process_outputs(final_outputs, batch_idx)

        self.teardown()
