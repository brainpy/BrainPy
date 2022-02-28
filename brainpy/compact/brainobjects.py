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
  def __init__(self, *args, **kwargs):
    warnings.warn('Please use "brainpy.dyn.DynamicalSystem" instead. '
                  '"brainpy.DynamicalSystem" is deprecated since '
                  'version 2.0.3', DeprecationWarning)
    super(DynamicalSystem, self).__init__(*args, **kwargs)


class Container(dyn.Container):
  def __init__(self, *args, **kwargs):
    warnings.warn('Please use "brainpy.dyn.Container" instead. '
                  '"brainpy.Container" is deprecated since '
                  'version 2.0.3', DeprecationWarning)
    super(Container, self).__init__(*args, **kwargs)


class Network(dyn.Network):
  def __init__(self, *args, **kwargs):
    warnings.warn('Please use "brainpy.dyn.Network" instead. '
                  '"brainpy.Network" is deprecated since '
                  'version 2.0.3', DeprecationWarning)
    super(Network, self).__init__(*args, **kwargs)


class ConstantDelay(dyn.ConstantDelay):
  def __init__(self, *args, **kwargs):
    warnings.warn('Please use "brainpy.dyn.ConstantDelay" instead. '
                  '"brainpy.ConstantDelay" is deprecated since '
                  'version 2.0.3', DeprecationWarning)
    super(ConstantDelay, self).__init__(*args, **kwargs)


class NeuGroup(dyn.NeuGroup):
  def __init__(self, *args, **kwargs):
    warnings.warn('Please use "brainpy.dyn.NeuGroup" instead. '
                  '"brainpy.NeuGroup" is deprecated since '
                  'version 2.0.3', DeprecationWarning)
    super(NeuGroup, self).__init__(*args, **kwargs)


class TwoEndConn(dyn.TwoEndConn):
  def __init__(self, *args, **kwargs):
    warnings.warn('Please use "brainpy.dyn.TwoEndConn" instead. '
                  '"brainpy.TwoEndConn" is deprecated since '
                  'version 2.0.3', DeprecationWarning)
    super(TwoEndConn, self).__init__(*args, **kwargs)
