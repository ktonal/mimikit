from typing import Optional, Any, Callable, Tuple, Iterable, Dict, Union
from typing_extensions import Literal
import numpy as np
import torch
import h5mapper as h5m
from functools import partial
import dataclasses as dtc

from .callbacks import tqdm

__all__ = [
    'GenerateLoop',
    "GenerateLoopV2",
    'prepare_prompt',
    'generate_tqdm',
]

from .logger import AudioLogger
from ..utils import default_device

from ..config import Config
from ..features.item_spec import ItemSpec, Second, Frame
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
                leave=False, unit="step", mininterval=1.)


FillType = Union[Literal["blank", "data"], torch.Tensor]


def fill(
        x: Optional[torch.Tensor],
        prior_t: Tuple[FillType, int],
        n_steps: Tuple[FillType, int],
):
    if isinstance(x, np.ndarray):
        x = torch.from_numpy(x)
    if x is not None:
        to_cat = [x]
        dt, dev = x.dtype, x.device
        B, D = x.size(0), x.shape[2:]
    else:
        to_cat = []
        dt, dev = torch.float32, "cpu"
        B, D = 1, (1,)
    for fill_type, n in [prior_t, n_steps]:
        if isinstance(fill_type, torch.Tensor):
            assert fill_type.shape == (B,)
            to_cat += [fill_type.expand(B, n, 1)]
        elif fill_type == "blank":
            to_cat += [torch.zeros(B, n, *D, dtype=dt, device=dev)]
        elif fill_type == "data":
            pass
    return torch.cat(to_cat, dim=1)


class PromptIndices(h5m.Input):
    def __init__(self, n):
        self.getter = h5m.Getter()
        self.getter.n = n

    def __call__(self, item, file=None):
        return np.array([item], dtype=np.int32)


class GenerateLoopV2:
    @dtc.dataclass
    class Config(Config):
        output_duration_sec: float = 1.
        prompts_length_sec: float = 1.
        prompts_position_sec: Tuple[Optional[float], ...] = (None,)  # random if None
        parameters: Optional[Dict[str, Any]] = None
        batch_size: int = 1

        output_name_template: Optional[str] = None
        display_waveform: bool = True
        write_waveform: bool = False
        yield_inversed_outputs: bool = True
        callback: Optional[Callable[[Tuple[torch.Tensor, ...]], None]] = None

    @classmethod
    def get_n_steps(cls, config: Config, network: ARM):
        io_spec = network.config.io_spec
        sr = io_spec.sr
        unit = io_spec.unit
        output_n_samples = int(sr * config.output_duration_sec)
        if isinstance(unit, Frame):
            # TODO: DOESN'T TAKE INTO ACCOUNT THE HOP OF S2S
            hop_length = unit.hop_length
            return output_n_samples // hop_length
        else:
            return output_n_samples

    @classmethod
    def get_dataloader(cls, config, dataset: h5m.TypedFile, network: ARM):
        io_spec = network.config.io_spec
        sr = io_spec.sr
        prompt_n_samples = int(sr * config.prompts_length_sec)
        max_len = prompt_n_samples
        # TODO: get rid of '.signal' assumption
        max_i = dataset.signal.shape[0] - max_len
        # todo: fixture get n_prior + n_steps
        prompt_spec = ItemSpec(0, length=config.prompts_length_sec, unit=Second(sr))
        prompt_batch, _ = network.test_batch(prompt_spec)
        prompt_batch = (
            PromptIndices(n=max_i), *prompt_batch
        )
        indices = tuple(int(x * sr) if x is not None else x
                        for x in config.prompts_position_sec)
        return dataset.serve(
            prompt_batch,
            sampler=IndicesSampler(N=len(indices),
                                   indices=indices,
                                   max_i=max_i,
                                   redraw=True),
            shuffle=False,
            batch_size=config.batch_size
        )

    @classmethod
    def from_config(cls, config: Config, dataset: h5m.TypedFile, network: ARM):
        n_steps = cls.get_n_steps(config, network)
        dataloader = cls.get_dataloader(config, dataset, network)
        logger = AudioLogger(
            sr=network.config.io_spec.sr,
            file_template=config.output_name_template if config.write_waveform else None,
            title_template=config.output_name_template if config.display_waveform else None)

        return cls(config, network, n_steps, dataloader, logger)

    def __init__(
            self,
            config: "GenerateLoopV2.Config",
            network: ARM,
            n_steps: int,
            dataloader: torch.utils.data.DataLoader,
            logger: Optional[AudioLogger] = None,
    ):
        self.config = config
        self.network = network
        self.n_steps = n_steps
        self.dataloader = dataloader
        self.logger = logger
        self._initial_device = None
        self._was_training = False
        self.device = None
        self.template_vars = {}

    def setup(self):
        net = self.network
        self._initial_device = net.device
        self._was_training = net.training
        net.eval()
        self.device = default_device()
        net.to(self.device)
        torch.set_grad_enabled(False)

    def teardown(self):
        self.network.to(self._initial_device)
        self.network.train() if self._was_training else None
        torch.set_grad_enabled(True)

    def run(self):

        self.setup()

        for batch in self.dataloader:
            prompt_idx, batch = batch[0], batch[1:]
            # prepare
            batch = tuple((torch.from_numpy(x) if isinstance(x, np.ndarray) else x).to(self.device)
                          for x in batch)
            self.network.before_generate(batch, prompt_idx)

            rf, prior_t, n_steps = self.network.rf, batch[0].size(1), self.n_steps

            tensors = h5m.process_batch(
                batch, lambda x: isinstance(x, torch.Tensor),
                partial(fill, prior_t=("data", prior_t), n_steps=("blank", n_steps))
            )
            # todo: initialize targets & couple auto regressive features
            params = self.config.parameters
            params = {} if params is None else params
            params = {k: v for k, v in params.items() if k in self.network.generate_params}
            # generate
            until = 0
            for t in generate_tqdm(range(prior_t, prior_t + n_steps)):
                if t < until:
                    continue
                inputs = tuple(tensor[:, t - rf:t] for tensor in tensors)
                outputs = self.network.generate_step(inputs, t=t, **params)
                if not isinstance(outputs, tuple):
                    outputs = outputs,
                for tensor, out in zip(tensors, outputs):
                    # let the net return None when ignoring this step
                    if out is not None:
                        n_out = min(out.size(1), tensor.size(1)-t)
                        tensor.data[:, t:t + n_out] = out[:, :n_out]
                        until = t + n_out

            # wrap up
            final_outputs = tuple(x.data for x in tensors)
            self.network.after_generate(final_outputs, prompt_idx)

            final_outputs = self.process_outputs(final_outputs, prompt_idx, **self.template_vars)
            yield final_outputs
            if self.config.callback is not None:
                self.config.callback(final_outputs)
        self.teardown()

    def process_outputs(
            self,
            final_outputs: Tuple[torch.Tensor, ...],
            prompt_idx: torch.Tensor,
            **template_vars
    ):
        if (self.logger is None or (
                not self.config.write_waveform and
                not self.config.display_waveform)
        ) and not self.config.yield_inversed_outputs:
            return final_outputs
        features = self.network.config.io_spec.targets
        outputs = tuple(feature.inv(out) for feature, out in zip(features, final_outputs))
        for output in outputs:
            for example, idx in zip(output, prompt_idx):
                if self.config.write_waveform:
                    self.logger.write(example, prompt_idx=idx.item(), **template_vars)
                if self.config.display_waveform:
                    self.logger.display(example, prompt_idx=idx.item(), **template_vars)
        return outputs if self.config.yield_inversed_outputs else final_outputs


class GenerateLoop:
    """
    interfaces' length must equal the number of items in the batches of the DataLoader
    if an interfaces has None as setter, it is considered a parameter, not a "generable"
    """

    def __init__(self,
                 # config: "GenerateLoop.Config",
                 network: ARM = None,
                 dataloader: torch.utils.data.dataloader.DataLoader = None,
                 inputs: Iterable[h5m.Input] = tuple(),
                 n_batches: Optional[int] = None,
                 n_steps: int = 1,
                 time_hop: int = 1,
                 disable_grads: bool = True,
                 device: str = default_device(),
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
        net.to(self.device)
        if self.disable_grads:
            torch.set_grad_enabled(False)

    def teardown(self):
        self.net.to(self._initial_device)
        self.net.train() if self._was_training else None
        if self.disable_grads:
            torch.set_grad_enabled(True)

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
            final_outputs = tuple(x.dataset for x in inputs_itf[:len(outputs_itf)])
            self.net.after_generate(final_outputs, batch_idx)

            self.process_outputs(final_outputs, batch_idx)

        self.teardown()
