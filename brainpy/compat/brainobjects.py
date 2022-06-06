# -*- coding: utf-8 -*-

import math as pm
import warnings

import brainpy.math as bm
from brainpy import dyn
from brainpy import tools
from brainpy.errors import ModelBuildError

__all__ = [
  'DynamicalSystem',
  'Container',
  'Network',
  'ConstantDelay',
  'NeuGroup',
  'TwoEndConn',
]


class DynamicalSystem(dyn.DynamicalSystem):
  """Dynamical System.

  .. deprecated:: 2.1.0
     Please use "brainpy.dyn.DynamicalSystem" instead.
  """
  def __init__(self, *args, **kwargs):
    warnings.warn('Please use "brainpy.dyn.DynamicalSystem" instead. '
                  '"brainpy.DynamicalSystem" is deprecated since '
                  'version 2.0.3', DeprecationWarning)
    super(DynamicalSystem, self).__init__(*args, **kwargs)


class Container(dyn.Container):
  """Container.

  .. deprecated:: 2.1.0
     Please use "brainpy.dyn.Container" instead.
  """
  def __init__(self, *args, **kwargs):
    warnings.warn('Please use "brainpy.dyn.Container" instead. '
                  '"brainpy.Container" is deprecated since '
                  'version 2.0.3', DeprecationWarning)
    super(Container, self).__init__(*args, **kwargs)


class Network(dyn.Network):
  """Network.

  .. deprecated:: 2.1.0
     Please use "brainpy.dyn.Network" instead.
  """
  def __init__(self, *args, **kwargs):
    warnings.warn('Please use "brainpy.dyn.Network" instead. '
                  '"brainpy.Network" is deprecated since '
                  'version 2.0.3', DeprecationWarning)
    super(Network, self).__init__(*args, **kwargs)


class ConstantDelay(dyn.DynamicalSystem):
  """Class used to model constant delay variables.

  This class automatically supports batch size on the last axis. For example, if
  you run batch with the size of (10, 100), where `100` are batch size, then this
  class can automatically support your batched data.
  For examples,

  >>> import brainpy as bp
  >>> bp.dyn.ConstantDelay(size=(10, 100), delay=10.)

  This class also support nonuniform delays.

  >>> bp.dyn.ConstantDelay(size=100, delay=bp.math.random.random(100) * 4 + 10)

  .. deprecated:: 2.1.0
     Please use "brainpy.dyn.ConstantDelay" instead.

  Parameters
  ----------
  size : int, list of int, tuple of int
    The delay data size.
  delay : int, float, function, ndarray
    The delay time. With the unit of `dt`.
  dt: float, optional
    The time precision.
  name : optional, str
    The name of the dynamic system.
  """

  def __init__(self, size, delay, dtype=None, dt=None, **kwargs):
    warnings.warn('Please use "brainpy.dyn.ConstantDelay" instead. '
                  '"brainpy.ConstantDelay" is deprecated since '
                  'version 2.0.3', DeprecationWarning)

    # dt
    self.dt = bm.get_dt() if dt is None else dt
    self.dtype = dtype

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
      self.num = 1

    else:  # non-uniform delay
      self.uniform_delay = False
      if not len(self.size) == 1:
        raise NotImplementedError(f'Currently, BrainPy only supports 1D heterogeneous '
                                  f'delays, while we got the heterogeneous delay with '
                                  f'{len(self.size)}-dimensions.')
      self.num = tools.size2num(size)
      if bm.ndim(delay) != 1:
        raise ModelBuildError(f'Only support a 1D non-uniform delay. '
                              f'But we got {delay.ndim}D: {delay}')
      if delay.shape[0] != self.size[0]:
        raise ModelBuildError(f"The first shape of the delay time size must "
                              f"be the same with the delay data size. But "
                              f"we got {delay.shape[0]} != {self.size[0]}")
      delay = bm.around(delay / self.dt)
      self.diag = bm.array(bm.arange(self.num))
      self.num_step = bm.array(delay, dtype=bm.uint32) + 1
      self.in_idx = bm.Variable(self.num_step - 1)
      self.out_idx = bm.Variable(bm.zeros(self.num, dtype=bm.uint32))
      self.data = bm.Variable(bm.zeros((self.num_step.max(),) + size, dtype=dtype))

    super(ConstantDelay, self).__init__(**kwargs)

  def reset(self):
    """Reset the variables."""
    self.in_idx[:] = self.num_step - 1
    self.out_idx[:] = 0
    self.data[:] = 0

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

  def update(self, t=None, dt=None, **kwargs):
      """Update the delay index."""
      self.in_idx[:] = (self.in_idx + 1) % self.num_step
      self.out_idx[:] = (self.out_idx + 1) % self.num_step


class NeuGroup(dyn.NeuGroup):
  """Neuron group.

  .. deprecated:: 2.1.0
     Please use "brainpy.dyn.NeuGroup" instead.
  """
  def __init__(self, *args, **kwargs):
    warnings.warn('Please use "brainpy.dyn.NeuGroup" instead. '
                  '"brainpy.NeuGroup" is deprecated since '
                  'version 2.0.3', DeprecationWarning)
    super(NeuGroup, self).__init__(*args, **kwargs)


class TwoEndConn(dyn.TwoEndConn):
  """Two-end synaptic connection.

  .. deprecated:: 2.1.0
     Please use "brainpy.dyn.TwoEndConn" instead.
  """
  def __init__(self, *args, **kwargs):
    warnings.warn('Please use "brainpy.dyn.TwoEndConn" instead. '
                  '"brainpy.TwoEndConn" is deprecated since '
                  'version 2.0.3', DeprecationWarning)
    super(TwoEndConn, self).__init__(*args, **kwargs)
