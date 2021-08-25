# -*- coding: utf-8 -*-

import math as pmath

from brainpy import errors
from brainpy import math as bmath
from brainpy.simulation.brainobjects.base import DynamicSystem
from brainpy.simulation.utils import size2len

__all__ = [
  'Delay',
  'ConstantDelay',
]


class Delay(DynamicSystem):
  def __init__(self, steps=('update',), name=None):
    super(Delay, self).__init__(steps=steps, monitors=None, name=name)

  def update(self, _t, _i):
    raise NotImplementedError


class ConstantDelay(Delay):
  """Constant delay object.

  For examples:

  >>> ConstantDelay(size=10, delay=10.)
  >>>
  >>> import numpy as np
  >>> ConstantDelay(size=100, delay=lambda: np.random.randint(5, 10))
  >>> ConstantDelay(size=100, delay=np.random.random(100) * 4 + 10)

  Parameters
  ----------
  size : int, list of int, tuple of int
      The delay data size.
  delay : int, float, function, ndarray
      The delay time. With the unit of `brainpy.math.get_dt()`.
  name : str, optional
      The name.
  """

  def __init__(self, size, delay, name=None, dtype=None):
    # delay data size
    if isinstance(size, int):
      size = (size,)
    if not isinstance(size, (tuple, list)):
      raise errors.ModelDefError(f'"size" must a tuple/list of int, '
                                 f'but we got {type(size)}: {size}')
    self.size = tuple(size)

    # delay time length
    self.delay = delay

    # data and operations
    if isinstance(delay, (int, float)):  # uniform delay
      self.uniform_delay = True
      self.delay_num_step = bmath.Variable(bmath.array([int(pmath.ceil(delay / bmath.get_dt())) + 1]))
      self.delay_data = bmath.Variable(bmath.zeros((self.delay_num_step[0],) + self.size, dtype=dtype))
      self.delay_out_idx = bmath.Variable(bmath.array([0]))
      self.delay_in_idx = self.delay_num_step - 1

      self.push = self._push_for_uniform_delay
      self.pull = self._pull_for_uniform_delay
    else:  # non-uniform delay
      self.uniform_delay = False
      if not len(self.size) == 1:
        raise NotImplementedError(f'Currently, BrainPy only supports 1D '
                                  f'heterogeneous delays, while we got the '
                                  f'heterogeneous delay with {len(self.size)}'
                                  f'-dimensions.')
      self.num = size2len(size)
      if callable(delay):  # like: "delay=lambda: np.random.randint(5, 10)"
        temp = bmath.zeros(size)
        for i in range(size[0]):
          temp[i] = delay()
        delay = temp
      else:
        if bmath.shape(delay) != self.size:
          raise errors.ModelUseError(f"The shape of the delay time size must be "
                                     f"the same with the delay data size. But we "
                                     f"got {bmath.shape(delay)} != {self.size}")
      delay = bmath.around(delay / bmath.get_dt())
      self.diag = bmath.array(bmath.arange(self.num), dtype=bmath.int_)
      self.delay_num_step = bmath.Variable(bmath.array(delay, dtype=bmath.int_) + 1)
      self.delay_data = bmath.Variable(bmath.zeros((self.delay_num_step.max(),) + size, dtype=dtype))
      self.delay_in_idx = self.delay_num_step - 1
      self.delay_out_idx = bmath.Variable(bmath.zeros(self.num, dtype=bmath.int_))

      self.push = self._push_for_nonuniform_delay
      self.pull = self._pull_for_nonuniform_delay

    super(ConstantDelay, self).__init__(name=name)

  def _pull_for_uniform_delay(self, idx=None):
    """Pull delay data in the case of uniform delay time."""
    if idx is None:
      return self.delay_data[self.delay_out_idx[0]]
    else:
      return self.delay_data[self.delay_out_idx[0]][idx]

  def _pull_for_nonuniform_delay(self, idx=None):
    """Pull delay data in the case of non-uniform delay time."""
    if idx is None:
      return self.delay_data[self.delay_out_idx, self.diag]
    else:
      didx = self.delay_out_idx[idx]
      return self.delay_data[didx, idx]

  def _push_for_uniform_delay(self, idx_or_val, value=None):
    """Push external data onto the bottom of the delay,
    for the case of uniform delay time."""
    if value is None:
      self.delay_data[self.delay_in_idx[0]] = idx_or_val
    else:
      self.delay_data[self.delay_in_idx[0]][idx_or_val] = value

  def _push_for_nonuniform_delay(self, idx_or_val, value=None):
    """Push external data onto the bottom of the delay,
    for the case of non-uniform delay time."""
    if value is None:
      self.delay_data[self.delay_in_idx, self.diag] = idx_or_val
    else:
      didx = self.delay_in_idx[idx_or_val]
      self.delay_data[didx, idx_or_val] = value

  def update(self, _t, _i):
    self.delay_in_idx[:] = (self.delay_in_idx + 1) % self.delay_num_step
    self.delay_out_idx[:] = (self.delay_out_idx + 1) % self.delay_num_step

  def reset(self):
    self.delay_data[:] = 0
    self.delay_in_idx[:] = self.delay_num_step - 1
    self.delay_out_idx[:] = 0 if self.uniform_delay else bmath.zeros(self.num, dtype=bmath.int_)
