# -*- coding: utf-8 -*-

import math

from brainpy import errors
from brainpy import backend
from brainpy.backend import ops
from brainpy.simulation.utils import size2len
from .base import DynamicSystem

__all__ = [
  'ConstantDelay',
]

_Delay_NO = 0


class ConstantDelay(DynamicSystem):
  """Constant delay object.

  For examples:

  >>> ConstantDelay(size=10, delay_time=10.)
  >>>
  >>> import numpy as np
  >>> ConstantDelay(size=100, delay_time=lambda: np.random.randint(5, 10))
  >>> ConstantDelay(size=100, delay_time=np.random.random(100) * 4 + 10)

  Parameters
  ----------
  size : int, list of int, tuple of int
      The delay data size.
  delay_time : int, float, function, list of int, list of float, tuple of int, tuple of float, ndarray, tensor
      The delay time length. With the unit of [ms].
  name : str, optional
      The name.
  show_code : bool
      Whether show the formatted code.

  """

  def __init__(self, size, delay_time, name=None, show_code=False):
    if name is None:
      global _Delay_NO
      name = f'Delay{_Delay_NO}'
      _Delay_NO += 1

    # delay data size
    if isinstance(size, int):
      size = (size,)
    if not isinstance(size, (tuple, list)):
      raise errors.ModelDefError(f'"size" must a tuple/list of int, '
                                 f'but we got {type(size)}: {size}')
    self.size = tuple(size)

    # delay time length
    self.delay_time = delay_time
    if isinstance(delay_time, (int, float)):
      self.uniform_delay = True
      self.delay_num_step = int(math.ceil(delay_time / backend.get_dt())) + 1
      self.delay_data = ops.zeros((self.delay_num_step,) + self.size)
      self.delay_in_idx = self.delay_num_step - 1
      self.delay_out_idx = 0

      self.push = self._push_for_uniform_delay
      self.pull = self._pull_for_uniform_delay
    else:
      if not len(self.size) == 1:
        raise NotImplementedError(f'Currently, BrainPy only supports 1D '
                                  f'heterogeneous delays, while we got the '
                                  f'heterogeneous delay with {len(self.size)}'
                                  f'-dimensions.')
      self.num = size2len(size)
      if callable(delay_time):
        temp = ops.zeros(size)
        for i in range(size[0]):
          temp[i] = delay_time()
        delay_time = temp
      else:
        if ops.shape(delay_time) != self.size:
          raise errors.ModelUseError(f"The shape of the delay time size must be "
                                     f"the same with the delay data size. But we "
                                     f"got {ops.shape(delay_time)} != {self.size}")
      self.uniform_delay = False
      delay = delay_time / backend.get_dt()
      dint = ops.as_tensor(delay_time / backend.get_dt(), dtype=ops.int)  # floor
      ddiff = (delay - dint) >= 0.5
      self.delay_num_step = ops.as_tensor(dint + ddiff, dtype=ops.int) + 1
      self.delay_data = ops.zeros((max(self.delay_num_step),) + size)
      self.diag = ops.as_tensor(ops.arange(self.num), dtype=ops.int)
      self.delay_in_idx = self.delay_num_step - 1
      self.delay_out_idx = ops.zeros(self.num, dtype=int)

      self.push = self._push_for_nonuniform_delay
      self.pull = self._pull_for_nonuniform_delay

    super(ConstantDelay, self).__init__(steps={'update': self.update},
                                        monitors=None,
                                        name=name,  # will be set by the host
                                        show_code=show_code)

  def _pull_for_uniform_delay(self, idx=None):
    if idx is None:
      return self.delay_data[self.delay_out_idx]
    else:
      return self.delay_data[self.delay_out_idx][idx]

  def _pull_for_nonuniform_delay(self, idx=None):
    if idx is None:
      return self.delay_data[self.delay_out_idx, self.diag]
    else:
      didx = self.delay_out_idx[idx]
      return self.delay_data[didx, idx]

  def _push_for_uniform_delay(self, idx_or_val, value=None):
    if value is None:
      self.delay_data[self.delay_in_idx] = idx_or_val
    else:
      self.delay_data[self.delay_in_idx][idx_or_val] = value

  def _push_for_nonuniform_delay(self, idx_or_val, value=None):
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
    self.delay_in_idx = self.delay_num_step - 1
    self.delay_out_idx = 0 if self.uniform_delay else ops.zeros(self.num, dtype=int)
