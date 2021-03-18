# -*- coding: utf-8 -*-


import abc
import math

from brainpy import backend
from brainpy import profile

__all__ = [
    'AbstractDelay',
    'ConstantDelay',
    'VaryingDelay',
    'NeutralDelay',
]


class AbstractDelay(abc.ABC):
    def __setitem__(self, time, value):
        pass

    def __getitem__(self, time):
        pass


class ConstantDelay(AbstractDelay):
    def __init__(self, size, delay_len, before_t0):
        # check size
        if isinstance(size, int):
            size = (size,)
        if not isinstance(size, (tuple, list)):
            raise ValueError('"size" must be an int, or a list/tuple of int.')

        # check delay_len
        dt = profile.get_dt()
        num_delay = int(math.ceil(delay_len / dt))

        # delay data
        self.data = backend.zeros((num_delay,) + size)

        # check defore_t0
        if callable(before_t0):
            for i in range(num_delay):
                self.data[i] = before_t0((i - num_delay) * dt)
        else:
            self.data[:] = before_t0

        # other variables
        self._delay_in = 0
        self._delay_out = ...

    def __setitem__(self, time, value):
        pass

    def __getitem__(self, time):
        pass


class VaryingDelay(AbstractDelay):
    def __init__(self):
        pass


class NeutralDelay(AbstractDelay):
    def __init__(self):
        pass
