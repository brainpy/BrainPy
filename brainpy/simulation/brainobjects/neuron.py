# -*- coding: utf-8 -*-

from brainpy import errors, math
from brainpy.base.collector import Collector
from brainpy.simulation import utils
from brainpy.simulation.brainobjects.base import DynamicalSystem


__all__ = [
  'NeuGroup',
  'Channel',
  'CondNeuGroup',
  'Soma',
  'Dendrite',
]


class NeuGroup(DynamicalSystem):
  """Base class to model neuronal groups.

  Parameters
  ----------
  size : int, tuple of int, list of int
      The neuron group geometry.
  steps : tuple of str, tuple of function, dict of (str, function), optional
      The step functions.
  steps : tuple of str, tuple of function, dict of (str, function), optional
      The callable function, or a list of callable functions.
  monitors : None, list, tuple, datastructures.Monitor
      Variables to monitor.
  name : str, optional
      The name of the dynamic system.
  """

  def __init__(self, size, name=None, steps=('update',), **kwargs):
    # size
    if isinstance(size, (list, tuple)):
      if len(size) <= 0:
        raise errors.BrainPyError('size must be int, or a tuple/list of int.')
      if not isinstance(size[0], int):
        raise errors.BrainPyError('size must be int, or a tuple/list of int.')
      size = tuple(size)
    elif isinstance(size, int):
      size = (size,)
    else:
      raise errors.BrainPyError('size must be int, or a tuple/list of int.')
    self.size = size
    self.num = utils.size2len(size)

    # initialize
    super(NeuGroup, self).__init__(steps=steps, name=name, **kwargs)

  def update(self, _t, _dt, **kwargs):
    """The function to specify the updating rule.

    Parameters
    ----------
    _t : float
      The current time.
    _dt : float
      The time step.
    """
    raise NotImplementedError(f'Subclass of {self.__class__.__name__} must '
                              f'implement "update" function.')


# ----------------------------------------------------
#
#         Conductance-based Neuron Model
#
# ----------------------------------------------------


class Channel(DynamicalSystem):
  """Base class to model ion channels."""

  def __init__(self, **kwargs):
    super(Channel, self).__init__(**kwargs)

  def init(self, num_batch, host):
    """Initialize variables in the channel."""
    if not isinstance(host, CondNeuGroup): raise ValueError
    self.num_batch = num_batch
    self.host = host

  def update(self, _t, _dt, **kwargs):
    """The function to specify the updating rule."""
    raise NotImplementedError(f'Subclass of {self.__class__.__name__} '
                              f'must implement "update" function.')


class CondNeuGroup(NeuGroup):
  """Conductance neuron group.

  This model requires the channels implement


  """
  def __init__(self, C=1., A=1e-3, V_th=0., **channels):
    self.C = C
    self.A = 1e-3 / A
    self.V_th = V_th

    # children channels
    self.child_channels = Collector()
    for key, ch in channels.items():
      if not isinstance(ch, Channel):
        raise errors.BrainPyError(f'{self.__class__.__name__} only receives '
                                  f'{Channel.__name__} instance, while we '
                                  f'got {type(ch)}: {ch}.')
      self.child_channels[key] = ch

  def init(self, size, num_batch=None, monitors=None, name=None):
    super(CondNeuGroup, self).__init__(size, steps=('update',), monitors=monitors, name=name)

    # initialize variables
    self.V = math.Variable(math.zeros(self.num, dtype=math.float_))
    self.spike = math.Variable(math.zeros(self.num, dtype=math.bool_))
    self.input = math.Variable(math.zeros(self.num, dtype=math.float_))
    self.I_ion = math.Variable(math.zeros(self.num, dtype=math.float_))
    self.V_linear = math.Variable(math.zeros(self.num, dtype=math.float_))

    # initialize variables in channels
    for ch in self.child_channels.values(): ch.init(host=self, num_batch=num_batch)

    return self

  def update(self, _t, _dt, **kwargs):
    # update variables in channels
    for ch in self.child_channels.values(): ch.update(_t, _dt)

    # update V using Exponential Euler method
    dvdt = self.I_ion / self.C + self.input * (self.A / self.C)
    linear = self.V_linear / self.C
    V = self.V + dvdt * (math.exp(linear * _dt) - 1) / linear

    # update other variables
    self.spike[:] = math.logical_and(V >= self.V_th, self.V < self.V_th)
    self.V_linear[:] = 0.
    self.I_ion[:] = 0.
    self.input[:] = 0.
    self.V[:] = V

  def __getattr__(self, item):
    """Wrap the dot access ('self.'). """
    child_channels = super(CondNeuGroup, self).__getattribute__('child_channels')
    if item in child_channels:
      return child_channels[item]
    else:
      return super(CondNeuGroup, self).__getattribute__(item)


# ---------------------------------------------------------
#
#          Multi-Compartment Neuron Model
#
# ---------------------------------------------------------

class Dendrite(DynamicalSystem):
  """Base class to model dendrites.

  """

  def __init__(self, name, **kwargs):
    super(Dendrite, self).__init__(name=name, **kwargs)


class Soma(DynamicalSystem):
  """Base class to model soma in multi-compartment neuron models.

  """

  def __init__(self, name, **kwargs):
    super(Soma, self).__init__(name=name, **kwargs)
