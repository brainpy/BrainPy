# -*- coding: utf-8 -*-

import math

from brainpy import backend
from brainpy.backend import ops
from brainpy.simulation.utils import size2len


__all__ = [
    'ConstantDelay',
]


class ConstantDelay(object):
    """Constant delay variable for synapse computation.
    """

    def __init__(self, size, delay_time):
        if isinstance(size, int):
            size = (size,)
        self.size = tuple(size)
        self.delay_time = delay_time

        if isinstance(delay_time, (int, float)):
            self.uniform_delay = True
            self.delay_num_step = int(math.ceil(delay_time / backend.get_dt())) + 1
            self.delay_data = ops.zeros((self.delay_num_step,) + self.size)
        else:
            if not len(self.size) == 1:
                raise NotImplementedError(f'Currently, BrainPy only supports 1D heterogeneous delays, while does '
                                          f'not implement the heterogeneous delay with {len(self.size)}-dimensions.')
            self.num = size2len(size)
            if isinstance(delay_time, type(ops.as_tensor([1]))):
                assert ops.shape(delay_time) == self.size
            elif callable(delay_time):
                delay_time2 = ops.zeros(size)
                for i in range(size[0]):
                    delay_time2[i] = delay_time()
                delay_time = delay_time2
            else:
                raise NotImplementedError(f'Currently, BrainPy does not support delay type '
                                          f'of {type(delay_time)}: {delay_time}')
            self.uniform_delay = False
            delay = delay_time / backend.get_dt()
            dint = ops.as_tensor(delay_time / backend.get_dt(), dtype=int)
            ddiff = (delay - dint) >= 0.5
            self.delay_num_step = ops.as_tensor(delay + ddiff, dtype=int) + 1
            self.delay_data = ops.zeros((max(self.delay_num_step),) + size)
            self.diag = ops.arange(self.num)

        self.delay_in_idx = self.delay_num_step - 1
        if self.uniform_delay:
            self.delay_out_idx = 0
        else:
            self.delay_out_idx = ops.zeros(self.num, dtype=int)
        self.name = None  # will be set by the parent

    def pull(self, idx=None):
        if self.uniform_delay:
            if idx is None:
                return self.delay_data[self.delay_out_idx]
            else:
                return self.delay_data[self.delay_out_idx][idx]
        else:
            if idx is None:
                return self.delay_data[self.delay_out_idx, self.diag]
            else:
                didx = self.delay_out_idx[idx]
                return self.delay_data[didx, idx]

    def push(self, idx_or_val, value=None):
        if self.uniform_delay:
            if value is None:
                self.delay_data[self.delay_in_idx] = idx_or_val
            else:
                self.delay_data[self.delay_in_idx][idx_or_val] = value
        else:
            if value is None:
                self.delay_data[self.delay_in_idx, self.diag] = idx_or_val
            else:
                didx = self.delay_in_idx[idx_or_val]
                self.delay_data[didx, idx_or_val] = value

    def update(self):
        self.delay_in_idx = (self.delay_in_idx + 1) % self.delay_num_step
        self.delay_out_idx = (self.delay_out_idx + 1) % self.delay_num_step

    def reset(self):
        self.delay_data[:] = 0
