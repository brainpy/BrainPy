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
        size = tuple(size)
        self.num = size2len(size)

        if isinstance(delay_time, (int, float)):
            self.uniform_delay = True
            self.delay_time = delay_time
            self.delay_num_step = int(math.ceil(delay_time / backend.get_dt())) + 1
            self.delay_data = ops.zeros((self.delay_num_step,) + size)
        else:
            assert self.num == len(delay_time)
            self.uniform_delay = False
            self.delay_time = delay_time
            delay = delay_time / backend.get_dt()
            dint = ops.as_tensor(delay_time / backend.get_dt(), dtype=int)
            ddiff = (delay - dint) >= 0.5
            self.delay_num_step = ops.as_tensor(delay + ddiff, dtype=int) + 1
            self.delay_data = ops.zeros((max(self.delay_num_step),) + size)

        self.delay_in_idx = self.delay_num_step - 1
        self.delay_out_idx = 0

        if self.uniform_delay:
            self.push = self.uniform_push
            self.pull = self.uniform_pull

        else:
            if backend.get_backend_name().startswith('numba'):
                self.push = self.non_uniform_push_for_jit_bk
                self.pull = self.non_uniform_pull_for_jit_bk
            else:
                self.diag = ops.arange(self.num)
                self.push = self.non_uniform_push_for_tensor_bk
                self.pull = self.non_uniform_pull_for_tensor_bk

    def non_uniform_push_for_tensor_bk(self, idx_or_val, value=None):
        if value is None:
            self.delay_data[self.delay_in_idx, self.diag] = idx_or_val
        else:
            didx = self.delay_in_idx[idx_or_val]
            self.delay_data[didx, idx_or_val] = value

    def non_uniform_pull_for_tensor_bk(self, idx=None):
        if idx is None:
            return self.delay_data[self.delay_out_idx, self.diag]
        else:
            didx = self.delay_out_idx[idx]
            return self.delay_data[didx, idx]

    def non_uniform_push_for_jit_bk(self, idx_or_val, value=None):
        if value is None:
            for i in range(self.num):
                self.delay_data[self.delay_in_idx, i] = idx_or_val[i]
        else:
            didx = self.delay_in_idx[idx_or_val]
            self.delay_data[didx, idx_or_val] = value

    def non_uniform_pull_for_jit_bk(self, idx=None):
        if idx is None:
            g = ops.zeros(self.num)
            for i in range(self.num):
                didx = self.delay_out_idx[i]
                g[i] = self.delay_data[didx, i]
            return g
        else:
            didx = self.delay_out_idx[idx]
            return self.delay_data[didx, idx]

    def uniform_push(self, idx_or_val, value=None):
        if value is None:
            self.delay_data[self.delay_in_idx] = idx_or_val
        else:
            self.delay_data[self.delay_in_idx][idx_or_val] = value

    def uniform_pull(self, idx=None):
        if idx is None:
            return self.delay_data[self.delay_out_idx]
        else:
            return self.delay_data[self.delay_out_idx][idx]

    def update(self):
        self.delay_in_idx = (self.delay_in_idx + 1) % self.delay_num_step
        self.delay_out_idx = (self.delay_out_idx + 1) % self.delay_num_step

