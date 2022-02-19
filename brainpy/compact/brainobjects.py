# -*- coding: utf-8 -*-

import warnings

from .. import sim

__all__ = [
  'DynamicalSystem',
  'Container',
  'Network',
  'ConstantDelay',
  'NeuGroup',
  'TwoEndConn',
]


class DynamicalSystem(sim.DynamicalSystem):
  def __init__(self, *args, **kwargs):
    warnings.warn('Please use "bp.sim.DynamicalSystem" instead. '
                  '"bp.DynamicalSystem" will be removed since '
                  'version 2.1.0', DeprecationWarning)
    super(DynamicalSystem, self).__init__(*args, **kwargs)


class Container(sim.Container):
  def __init__(self, *args, **kwargs):
    warnings.warn('Please use "bp.sim.Container" instead. '
                  '"bp.Container" will be removed since '
                  'version 2.1.0', DeprecationWarning)
    super(Container, self).__init__(*args, **kwargs)


class Network(sim.Network):
  def __init__(self, *args, **kwargs):
    warnings.warn('Please use "bp.sim.Network" instead. '
                  '"bp.Network" will be removed since '
                  'version 2.1.0', DeprecationWarning)
    super(Network, self).__init__(*args, **kwargs)


class ConstantDelay(sim.ConstantDelay):
  def __init__(self, *args, **kwargs):
    warnings.warn('Please use "bp.sim.ConstantDelay" instead. '
                  '"bp.ConstantDelay" will be removed since '
                  'version 2.1.0', DeprecationWarning)
    super(ConstantDelay, self).__init__(*args, **kwargs)


class NeuGroup(sim.NeuGroup):
  def __init__(self, *args, **kwargs):
    warnings.warn('Please use "bp.sim.NeuGroup" instead. '
                  '"bp.NeuGroup" will be removed since '
                  'version 2.1.0', DeprecationWarning)
    super(NeuGroup, self).__init__(*args, **kwargs)


class TwoEndConn(sim.TwoEndConn):
  def __init__(self, *args, **kwargs):
    warnings.warn('Please use "bp.sim.TwoEndConn" instead. '
                  '"bp.TwoEndConn" will be removed since '
                  'version 2.1.0', DeprecationWarning)
    super(TwoEndConn, self).__init__(*args, **kwargs)
