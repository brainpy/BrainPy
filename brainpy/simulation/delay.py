# -*- coding: utf-8 -*-

import math

from brainpy import backend

__all__ = [
    'ConstantDelay',
    'push_type1',
    'push_type2',
    'pull_type0',
    'pull_type1',
]


class ConstantDelay(object):
    """Constant delay variable for synapse computation.

    """

    def __init__(self, size, delay_time):
        self.delay_time = delay_time
        self.delay_num_step = int(math.ceil(delay_time / backend.get_dt())) + 1
        self.delay_in_idx = self.delay_num_step - 1
        self.delay_out_idx = 0

        if isinstance(size, int):
            size = (size,)
        size = tuple(size)
        self.delay_data = backend.zeros((self.delay_num_step,) + size)

    def push(self, idx_or_val, value=None):
        if value is None:
            self.delay_data[self.delay_in_idx] = idx_or_val
        else:
            self.delay_data[self.delay_in_idx][idx_or_val] = value

    def pull(self, idx=None):
        if idx is None:
            return self.delay_data[self.delay_out_idx]
        else:
            return self.delay_data[self.delay_out_idx][idx]

    def update(self):
        self.delay_in_idx = (self.delay_in_idx + 1) % self.delay_num_step
        self.delay_out_idx = (self.delay_out_idx + 1) % self.delay_num_step


def push_type1(idx_or_val, delay_data, delay_in_idx):
    delay_data[delay_in_idx] = idx_or_val


def push_type2(idx_or_val, value, delay_data, delay_in_idx):
    delay_data[delay_in_idx][idx_or_val] = value


def pull_type0(delay_data, delay_out_idx):
    return delay_data[delay_out_idx]


def pull_type1(idx, delay_data, delay_out_idx):
    return delay_data[delay_out_idx][idx]
