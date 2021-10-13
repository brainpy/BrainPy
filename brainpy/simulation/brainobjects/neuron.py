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
  num_batch : optional, int
    The batch size.
  steps : tuple of str, tuple of function, dict of (str, function), optional
    The step functions.
  steps : tuple of str, tuple of function, dict of (str, function), optional
    The callable function, or a list of callable functions.
  monitors : None, list, tuple, datastructures.Monitor
    Variables to monitor.
  name : str, optional
    The name of the dynamic system.
  """

  def __init__(self, size, num_batch=None, name=None, steps=('update',), **kwargs):
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
    self.num_batch = num_batch
    if num_batch is None:
      self.shape = (self.num, )
    else:
      assert isinstance(num_batch, int), 'Only support int for "num_batch"'
      self.shape = (self.num, num_batch)

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
  """Base class to model ion channels.

  Notes
  -----

  The ``__init__()`` function in :py:class:`Channel` is used to specify
  the parameters of the channel. The ``__call__()`` function
  is used to initialize the variables in this channel.
  """

  def __init__(self, **kwargs):
    super(Channel, self).__init__(**kwargs)

  def init(self, host, **kwargs):
    """Initialize variables in this channel."""
    if not isinstance(host, CondNeuGroup):
      raise errors.BrainPyError(f'Only support host with instance of {str(DynamicalSystem)}, while we got {host}')
    self.host = host

  def update(self, _t, _dt, **kwargs):
    """The function to specify the updating rule."""
    raise NotImplementedError(f'Subclass of {self.__class__.__name__} '
                              f'must implement "update" function.')


class CondNeuGroup(NeuGroup):
  """Base class to model conductance-based neuron group.

  The standard formulation for a conductance-based model is given as

  .. math::

      C_m {dV \over dt} = \sum_jg_j(E - V) + I_{ext}

  where :math:`g_j=\bar{g}_{j} M^x N^y` is the channel conductance, :math:`E` is the
  reversal potential, :math:`M` is the activation variable, and :math:`N` is the
  inactivation variable.

  :math:`M` and :math:`N` have the dynamics of

  .. math::

      {dx \over dt} = \phi_x {x_\infty (V) - x \over \tau_x(V)}

  where :math:`x \in [M, N]`, :math:`\phi_x` is a temperature-dependent factor,
  :math:`x_\infty` is the steady state, and :math:`\tau_x` is the time constant.
  Equivalently, the above equation can be written as:

  .. math::

      \frac{d x}{d t}=\phi_{x}\left(\alpha_{x}(1-x)-\beta_{x} x\right)

  where :math:`\alpha_{x}` and :math:`\beta_{x}` are rate constants.

  Parameters
  ----------
  size : int, tuple of int
    The network size of this neuron group.
  num_batch : optional, int
    The batch size.
  monitors : optional, list of str, tuple of str
    The neuron group monitor.
  name : optional, str
    The neuron group name.

  Notes
  -----

  The ``__init__()`` function in :py:class:`CondNeuGroup` is used to specify
  the parameters of channels and this neuron group. The ``__call__()`` function
  is used to initialize the variables in this neuron group.
  """
  def __init__(self, C=1., A=1e-3, V_th=0., **channels):
    # parameters for neuron
    self.C = C
    self.A = A
    self.V_th = V_th

    # children channels
    self.child_channels = Collector()
    for key, ch in channels.items():
      if not isinstance(ch, Channel):
        raise errors.BrainPyError(f'{self.__class__.__name__} only receives {Channel.__name__} '
                                  f'instance, while we got {type(ch)}: {ch}.')
      self.child_channels[key] = ch

  def init(self, size, num_batch=None, monitors=None, steps=('update',), name=None):
    super(CondNeuGroup, self).__init__(size, num_batch=num_batch, steps=steps, monitors=monitors, name=name)

    # initialize variables
    self.V = math.Variable(math.zeros(self.shape, dtype=math.float_))
    self.spike = math.Variable(math.zeros(self.shape, dtype=math.bool_))
    self.input = math.Variable(math.zeros(self.shape, dtype=math.float_))
    self.I_ion = math.Variable(math.zeros(self.shape, dtype=math.float_))
    self.V_linear = math.Variable(math.zeros(self.shape, dtype=math.float_))

    # initialize variables in channels
    for ch in self.child_channels.values(): ch.init(host=self)
    return self

  def update(self, _t, _dt, **kwargs):
    # update variables in channels
    for ch in self.child_channels.values():
      ch.update(_t, _dt)

    # update V using Exponential Euler method
    dvdt = self.I_ion / self.C + self.input * (1e-3 / self.A / self.C)
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
