# -*- coding: utf-8 -*-

import warnings

from brainpy import dyn

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


class ConstantDelay(dyn.ConstantDelay):
  """Constant Delay.

  .. deprecated:: 2.1.0
     Please use "brainpy.dyn.ConstantDelay" instead.
  """
  def __init__(self, *args, **kwargs):
    warnings.warn('Please use "brainpy.dyn.ConstantDelay" instead. '
                  '"brainpy.ConstantDelay" is deprecated since '
                  'version 2.0.3', DeprecationWarning)
    super(ConstantDelay, self).__init__(*args, **kwargs)


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
