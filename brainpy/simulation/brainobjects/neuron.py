# -*- coding: utf-8 -*-

from brainpy import errors, math
from brainpy.base.collector import Collector
from brainpy.simulation import utils
from brainpy.integrators.wrapper import odeint
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
  name : str, optional
      The group name.
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

  def update(self, _t, _dt):
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
#          Conductance Neuron Model
#
# ----------------------------------------------------

class Channel(DynamicalSystem):
  """Base class to model ion channels."""

  def __init__(self, **kwargs):
    super(Channel, self).__init__(**kwargs)

  def init(self, host):
    """Initialize variables in the channel."""
    if not isinstance(host, CondNeuGroup):
      raise ValueError
    self.host = host

  def update(self, _t, _dt):
    """The function to specify the updating rule."""
    raise NotImplementedError(f'Subclass of {self.__class__.__name__} must '
                              f'implement "update" function.')


class CondNeuGroup(NeuGroup):
  """Conductance neuron group."""
  def __init__(self, C=1., A=1e-3, Vth=0., **channels):
    self.C = C
    self.A = A
    self.Vth = Vth

    # children channels
    self.child_channels = Collector()
    for key, ch in channels.items():
      if not isinstance(ch, Channel):
        raise errors.BrainPyError(f'{self.__class__.__name__} only receives {Channel.__name__} '
                                  f'instance, while we got {type(ch)}: {ch}.')
      self.child_channels[key] = ch

  def init(self, size, monitors=None, name=None):
    super(CondNeuGroup, self).__init__(size, steps=('update',), monitors=monitors, name=name)

    # initialize variables
    self.V = math.Variable(math.zeros(self.num, dtype=math.float_))
    self.spike = math.Variable(math.zeros(self.num, dtype=math.bool_))
    self.input = math.Variable(math.zeros(self.num, dtype=math.float_))

    # initialize node variables
    for ch in self.child_channels.values():
      ch.init(host=self)

    # checking
    self._output_channels = []
    self._update_channels = []
    for ch in self.child_channels.values():
      if not hasattr(ch, 'I'):
        self._update_channels.append(ch)
      else:
        if not isinstance(getattr(ch, 'I'), math.Variable):
          raise errors.BrainPyError
        self._output_channels.append(ch)

    return self

  def update(self, _t, _dt):
    for ch in self._update_channels:
      ch.update(_t, _dt)
    for ch in self._output_channels:
      ch.update(_t, _dt)
      self.input += ch.I

    # update variables
    V = self.V + self.input / self.C * _dt
    # V = self.V + self.input / self.C / self.A * _dt
    self.spike[:] = math.logical_and(V >= self.Vth, self.V < self.Vth)
    self.V[:] = V
    self.input[:] = 0.

  def __getattr__(self, item):
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
