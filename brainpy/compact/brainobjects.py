# -*- coding: utf-8 -*-

import warnings

from brainpy import sim

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
                  '"bp.DynamicalSystem" is deprecated since '
                  'version 2.0.3', DeprecationWarning)
    super(DynamicalSystem, self).__init__(*args, **kwargs)


class Container(sim.Container):
  def __init__(self, *args, **kwargs):
    warnings.warn('Please use "bp.sim.Container" instead. '
                  '"bp.Container" is deprecated since '
                  'version 2.0.3', DeprecationWarning)
    super(Container, self).__init__(*args, **kwargs)


class Network(sim.Network):
  def __init__(self, *args, **kwargs):
    warnings.warn('Please use "bp.sim.Network" instead. '
                  '"bp.Network" is deprecated since '
                  'version 2.0.3', DeprecationWarning)
    super(Network, self).__init__(*args, **kwargs)


class ConstantDelay(sim.ConstantDelay):
  def __init__(self, *args, **kwargs):
    warnings.warn('Please use "bp.sim.ConstantDelay" instead. '
                  '"bp.ConstantDelay" is deprecated since '
                  'version 2.0.3', DeprecationWarning)
    super(ConstantDelay, self).__init__(*args, **kwargs)


class NeuGroup(sim.NeuGroup):
  def __init__(self, *args, **kwargs):
    warnings.warn('Please use "bp.sim.NeuGroup" instead. '
                  '"bp.NeuGroup" is deprecated since '
                  'version 2.0.3', DeprecationWarning)
    super(NeuGroup, self).__init__(*args, **kwargs)


class TwoEndConn(sim.TwoEndConn):
  def __init__(self, *args, **kwargs):
    warnings.warn('Please use "bp.sim.TwoEndConn" instead. '
                  '"bp.TwoEndConn" is deprecated since '
                  'version 2.0.3', DeprecationWarning)
    super(TwoEndConn, self).__init__(*args, **kwargs)
