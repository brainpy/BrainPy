# -*- coding: utf-8 -*-

import math as pm

from brainpy import math as bm
from brainpy.errors import ModelBuildError
from brainpy.building.brainobjects.base import DynamicalSystem
from brainpy.simulation.utils import size2len

__all__ = [
  'Delay',
  'ConstantDelay',
]


class Delay(DynamicalSystem):
  """Base class to model delay variables.

  Parameters
  ----------
  steps : tuple of str, tuple of function, dict of (str, function), optional
      The callable function, or a list of callable functions.
  name : str, optional
      The name of the dynamic system.
  """

  def __init__(self, steps=('update',), name=None):
    super(Delay, self).__init__(steps=steps, name=name)

  def update(self, _t, _dt, **kwargs):
    raise NotImplementedError


class ConstantDelay(Delay):
  """Class used to model constant delay variables.

  This class automatically supports batch size on the last axis. For example, if
  you run batch with the size of (10, 100), where `100` are batch size, then this
  class can automatically support your batched data.

  For examples:

  >>> import brainpy as bp
  >>>
  >>> bp.ConstantDelay(size=10, delay=10.)
  >>> bp.ConstantDelay(size=100, delay=bp.math.random.random(100) * 4 + 10)

  Parameters
  ----------
  size : int, list of int, tuple of int
    The delay data size.
  delay : int, float, function, ndarray
    The delay time. With the unit of `dt`.
  num_batch : optional, int
    The batch size.
  steps : optional, tuple of str, tuple of function, dict of (str, function)
    The callable function, or a list of callable functions.
  name : optional, str
    The name of the dynamic system.
  """

  def __init__(self, size, delay, dtype=None, dt=None, **kwargs):
    # dt
    self.dt = bm.get_dt() if dt is None else dt

    # data size
    if isinstance(size, int): size = (size,)
    if not isinstance(size, (tuple, list)):
      raise ModelBuildError(f'"size" must a tuple/list of int, but we got {type(size)}: {size}')
    self.size = tuple(size)

    # delay time length
    self.delay = delay

    # data and operations
    if isinstance(delay, (int, float)):  # uniform delay
      self.uniform_delay = True
      self.num_step = int(pm.ceil(delay / self.dt)) + 1
      self.out_idx = bm.Variable(bm.array([0], dtype=bm.uint32))
      self.in_idx = bm.Variable(bm.array([self.num_step - 1], dtype=bm.uint32))
      self.data = bm.Variable(bm.zeros((self.num_step,) + self.size, dtype=dtype))

    else:  # non-uniform delay
      self.uniform_delay = False
      if not len(self.size) == 1:
        raise NotImplementedError(f'Currently, BrainPy only supports 1D heterogeneous '
                                  f'delays, while we got the heterogeneous delay with '
                                  f'{len(self.size)}-dimensions.')
      self.num = size2len(size)
      if bm.ndim(delay) != 1:
        raise ModelBuildError(f'Only support a 1D non-uniform delay. '
                              f'But we got {delay.ndim}D: {delay}')
      if delay.shape[0] != self.size[0]:
        raise ModelBuildError(f"The first shape of the delay time size must "
                              f"be the same with the delay data size. But "
                              f"we got {delay.shape[0]} != {self.size[0]}")
      delay = bm.around(delay / self.dt)
      self.diag = bm.array(bm.arange(self.num), dtype=bm.int_)
      self.num_step = bm.array(delay, dtype=bm.uint32) + 1
      self.in_idx = bm.Variable(self.num_step - 1)
      self.out_idx = bm.Variable(bm.zeros(self.num, dtype=bm.uint32))
      self.data = bm.Variable(bm.zeros((self.num_step.max(),) + size, dtype=dtype))

    super(ConstantDelay, self).__init__(**kwargs)

  @property
  def oldest(self):
    return self.pull()

  @property
  def latest(self):
    if self.uniform_delay:
      return self.data[self.in_idx[0]]
    else:
      return self.data[self.in_idx, self.diag]

  def pull(self):
    if self.uniform_delay:
      return self.data[self.out_idx[0]]
    else:
      return self.data[self.out_idx, self.diag]

  def push(self, value):
    if self.uniform_delay:
      self.data[self.in_idx[0]] = value
    else:
      self.data[self.in_idx, self.diag] = value

  def update(self, _t, _dt, **kwargs):
    """Update the delay index."""
    self.in_idx[:] = (self.in_idx + 1) % self.num_step
    self.out_idx[:] = (self.out_idx + 1) % self.num_step

  def reset(self):
    """Reset the variables."""
    self.in_idx[:] = self.num_step - 1
    self.out_idx[:] = 0
    self.data[:] = 0
