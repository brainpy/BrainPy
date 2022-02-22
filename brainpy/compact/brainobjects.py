# -*- coding: utf-8 -*-

import warnings

from brainpy import dynsim

__all__ = [
  'DynamicalSystem',
  'Container',
  'Network',
  'ConstantDelay',
  'NeuGroup',
  'TwoEndConn',
]


class DynamicalSystem(dynsim.DynamicalSystem):
  def __init__(self, *args, **kwargs):
    warnings.warn('Please use "bp.dynsim.DynamicalSystem" instead. '
                  '"bp.DynamicalSystem" is deprecated since '
                  'version 2.0.3', DeprecationWarning)
    super(DynamicalSystem, self).__init__(*args, **kwargs)


class Container(dynsim.Container):
  def __init__(self, *args, **kwargs):
    warnings.warn('Please use "bp.dynsim.Container" instead. '
                  '"bp.Container" is deprecated since '
                  'version 2.0.3', DeprecationWarning)
    super(Container, self).__init__(*args, **kwargs)


class Network(dynsim.Network):
  def __init__(self, *args, **kwargs):
    warnings.warn('Please use "bp.dynsim.Network" instead. '
                  '"bp.Network" is deprecated since '
                  'version 2.0.3', DeprecationWarning)
    super(Network, self).__init__(*args, **kwargs)


class ConstantDelay(dynsim.ConstantDelay):
  def __init__(self, *args, **kwargs):
    warnings.warn('Please use "bp.dynsim.ConstantDelay" instead. '
                  '"bp.ConstantDelay" is deprecated since '
                  'version 2.0.3', DeprecationWarning)
    super(ConstantDelay, self).__init__(*args, **kwargs)


class NeuGroup(dynsim.NeuGroup):
  def __init__(self, *args, **kwargs):
    warnings.warn('Please use "bp.dynsim.NeuGroup" instead. '
                  '"bp.NeuGroup" is deprecated since '
                  'version 2.0.3', DeprecationWarning)
    super(NeuGroup, self).__init__(*args, **kwargs)


class TwoEndConn(dynsim.TwoEndConn):
  def __init__(self, *args, **kwargs):
    warnings.warn('Please use "bp.dynsim.TwoEndConn" instead. '
                  '"bp.TwoEndConn" is deprecated since '
                  'version 2.0.3', DeprecationWarning)
    super(TwoEndConn, self).__init__(*args, **kwargs)
