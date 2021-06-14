# -*- coding: utf-8 -*-


import abc
import math

from brainpy import backend
from brainpy.backend import ops

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
  def __init__(self, v0, delay_len, before_t0=0., t0=0., dt=None):
    # size
    self.size = ops.shape(v0)

    # delay_len
    self.delay_len = delay_len
    self.dt = backend.get_dt() if dt is None else dt
    self.num_delay = int(math.ceil(delay_len / self.dt))

    # other variables
    self._delay_in = self.num_delay - 1
    self._delay_out = 0
    self.current_time = t0

    # before_t0
    self.before_t0 = before_t0

    # delay data
    self.data = ops.zeros((self.num_delay + 1,) + self.size)
    if callable(before_t0):
      for i in range(self.num_delay):
        self.data[i] = before_t0(t0 + (i - self.num_delay) * self.dt)
    else:
      self.data[:-1] = before_t0
    self.data[-1] = v0

  def __setitem__(self, time, value):  # push
    self.data[self._delay_in] = value
    self.current_time = time

  def __getitem__(self, time):  # pull
    diff = self.current_time - time
    m = math.ceil(diff / self.dt)
    return self.data[self._delay_out]

  def update(self):
    self._delay_in = (self._delay_in + 1) % self.num_delay
    self._delay_out = (self._delay_out + 1) % self.num_delay


class VaryingDelay(AbstractDelay):
  def __init__(self):
    pass


class NeutralDelay(AbstractDelay):
  def __init__(self):
    pass
