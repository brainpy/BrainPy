# -*- coding: utf-8 -*-

from brainpy.simulation.brainobjects.container import Container


__all__ = [
  'Network'
]

_Network_NO = 0


class Network(Container):
  """Network object, an alias of Container.

  Network instantiates a network, which is aimed to load
  neurons, synapses, and other brain objects.

  Parameters
  ----------
  name : str
    The network name.

  """

  def __init__(self, monitors=None, name=None, **kwargs):
    if name is None:
      global _Network_NO
      name = f'Net{_Network_NO}'
      _Network_NO += 1
    super(Network, self).__init__(name=name, monitors=monitors, **kwargs)
