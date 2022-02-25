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
    warnings.warn('Please use "bp.dyn.DynamicalSystem" instead. '
                  '"bp.DynamicalSystem" is deprecated since '
                  'version 2.0.3', DeprecationWarning)
    super(DynamicalSystem, self).__init__(*args, **kwargs)


class Container(dyn.Container):
  def __init__(self, *args, **kwargs):
    warnings.warn('Please use "bp.dyn.Container" instead. '
                  '"bp.Container" is deprecated since '
                  'version 2.0.3', DeprecationWarning)
    super(Container, self).__init__(*args, **kwargs)


class Network(dyn.Network):
  def __init__(self, *args, **kwargs):
    warnings.warn('Please use "bp.dyn.Network" instead. '
                  '"bp.Network" is deprecated since '
                  'version 2.0.3', DeprecationWarning)
    super(Network, self).__init__(*args, **kwargs)


class ConstantDelay(dyn.ConstantDelay):
  def __init__(self, *args, **kwargs):
    warnings.warn('Please use "bp.dyn.ConstantDelay" instead. '
                  '"bp.ConstantDelay" is deprecated since '
                  'version 2.0.3', DeprecationWarning)
    super(ConstantDelay, self).__init__(*args, **kwargs)


class NeuGroup(dyn.NeuGroup):
  def __init__(self, *args, **kwargs):
    warnings.warn('Please use "bp.dyn.NeuGroup" instead. '
                  '"bp.NeuGroup" is deprecated since '
                  'version 2.0.3', DeprecationWarning)
    super(NeuGroup, self).__init__(*args, **kwargs)


class TwoEndConn(dyn.TwoEndConn):
  def __init__(self, *args, **kwargs):
    warnings.warn('Please use "bp.dyn.TwoEndConn" instead. '
                  '"bp.TwoEndConn" is deprecated since '
                  'version 2.0.3', DeprecationWarning)
    super(TwoEndConn, self).__init__(*args, **kwargs)
