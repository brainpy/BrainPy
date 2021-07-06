# -*- coding: utf-8 -*-

from brainpy.simulation.brainobjects.container import Container


__all__ = [
  'Network'
]


class Network(Container):
  """Network object, an alias of Container.

  Network instantiates a network, which is aimed to load
  neurons, synapses, and other brain objects.

  Parameters
  ----------
  name : str
    The network name.

  """

  def __init__(self, monitors=None, name=None, **dynamic_systems):
    super(Network, self).__init__(name=name, monitors=monitors, **dynamic_systems)
