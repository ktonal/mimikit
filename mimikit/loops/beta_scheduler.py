import math
import weakref
from functools import wraps

import torch


class BetaScheduler(torch.optim.lr_scheduler.LRScheduler):

    def __init__(
            self,
            optimizer,
            max_beta,
            epochs=None,
            steps_per_epoch=None,
            pct_start=0.3,
            div_factor=25.,
            final_div_factor=1e4,
    ):
        self.optimizer = optimizer
        self.total_steps = epochs * steps_per_epoch
        self._schedule_phases = [
                {
                    'end_step': float(pct_start * self.total_steps) - 1,
                    'start_beta': 'initial_beta',
                    'end_beta': 'max_beta',
                },
                {
                    'end_step': self.total_steps - 1,
                    'start_beta': 'max_beta',
                    'end_beta': 'min_beta',
                },
            ]
        max_betas = [max_beta] * len(self.optimizer.param_groups)
        for idx, group in enumerate(self.optimizer.param_groups):
            group['initial_beta'] = max_betas[idx] / div_factor
            group['max_beta'] = max_betas[idx]
            group['min_beta'] = group['initial_beta'] / final_div_factor

        self.last_epoch = -1

        def with_counter(method):
            if getattr(method, '_with_counter', False):
                # `optimizer.step()` has already been replaced, return.
                return method

            # Keep a weak reference to the optimizer instance to prevent
            # cyclic references.
            instance_ref = weakref.ref(method.__self__)
            # Get the unbound method for the same purpose.
            func = method.__func__
            cls = instance_ref().__class__
            del method

            @wraps(func)
            def wrapper(*args, **kwargs):
                instance = instance_ref()
                instance._step_count += 1
                wrapped = func.__get__(instance, cls)
                return wrapped(*args, **kwargs)

            # Note that the returned function here is no longer a bound method,
            # so attributes like `__func__` and `__self__` no longer exist.
            wrapper._with_counter = True
            return wrapper

        self.optimizer.step = with_counter(self.optimizer.step)
        self.optimizer._step_count = 0
        self._step_count = 0

    def anneal_func(self, start, end, pct):
        "Cosine anneal from `start` to `end` as pct goes from 0.0 to 1.0."
        cos_out = math.cos(math.pi * pct) + 1
        return end + (start - end) / 2.0 * cos_out

    def get_beta(self):
        lrs = []
        step_num = self.last_epoch
        for group in self.optimizer.param_groups:
            start_step = 0
            for i, phase in enumerate(self._schedule_phases):
                end_step = phase['end_step']
                if step_num <= end_step or i == len(self._schedule_phases) - 1:
                    pct = (step_num - start_step) / (end_step - start_step)
                    computed_beta = self.anneal_func(group[phase['start_beta']], group[phase['end_beta']], pct)
                start_step = phase['end_step']

            lrs.append(computed_beta)
        return lrs

    def step(self, epoch=None):
        self._step_count += 1
        self.last_epoch += 1
        values = self.get_beta()
        for i, data in enumerate(zip(self.optimizer.param_groups, values)):
            param_group, beta = data
            param_group['betas'] = (beta, beta)

        self._last_beta = [group['betas'][0] for group in self.optimizer.param_groups]

    def state_dict(self):
        """Returns the state of the scheduler as a :class:`dict`.

        It contains an entry for every variable in self.__dict__ which
        is not the optimizer.
        """
        return {key: value for key, value in self.__dict__.items() if key != 'optimizer'}

    def load_state_dict(self, state_dict):
        """Loads the schedulers state.

        Args:
            state_dict (dict): scheduler state. Should be an object returned
                from a call to :meth:`state_dict`.
        """
        self.__dict__.update(state_dict)