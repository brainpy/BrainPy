# -*- coding: utf-8 -*-

from brainpy import errors
from brainpy.simulation.brainobjects.base import DynamicSystem
from brainpy.simulation.brainobjects.delays import ConstantDelay
from brainpy.simulation.brainobjects.neuron import NeuGroup

__all__ = [
  'TwoEndConn',
]


class TwoEndConn(DynamicSystem):
  """Two End Synaptic Connections.

  Parameters
  ----------
  steps : function, list of function, tuple of function, dict of (str, function), optional
      The step functions.
  pre : NeuGroup
      Pre-synaptic neuron group.
  post : NeuGroup
      Post-synaptic neuron group.
  monitors : list of str, tuple of str
      Variables to monitor.
  name : str
      The name of the neuron group.
  show_code : bool
      Whether show the formatted code.
  """

  def __init__(self, pre, post, name=None, steps=('update',), **kwargs):
    # pre or post neuron group
    # ------------------------
    if not isinstance(pre, NeuGroup):
      raise errors.BrainPyError('"pre" must be an instance of NeuGroup.')
    if not isinstance(post, NeuGroup):
      raise errors.BrainPyError('"post" must be an instance of NeuGroup.')
    self.pre = pre
    self.post = post

    # initialize
    # ----------
    super(TwoEndConn, self).__init__(steps=steps, name=name, **kwargs)

  def register_constant_delay(self, key, size, delay, dtype=None):
    """Register a constant delay.

    Parameters
    ----------
    key : str
        The delay name.
    size : int, list of int, tuple of int
        The delay data size.
    delay : int, float, ndarray
        The delay time, with the unit same with `brainpy.math.get_dt()`.

    Returns
    -------
    delay : ConstantDelay
        An instance of ConstantDelay.
    """

    if not hasattr(self, 'steps'):
      raise errors.BrainPyError('Please initialize the super class first before '
                                'registering constant_delay. \n\n'
                                'super(YourClassName, self).__init__(**kwargs)')
    if not key.isidentifier():
      raise ValueError(f'{key} is not a valid identifier.')

    cdelay = ConstantDelay(size, delay, name=f'{self.name}_delay_{key}', dtype=dtype)
    self.steps[f'{key}_update'] = cdelay.update

    return cdelay

  def update(self, _t, _i):
    raise NotImplementedError
