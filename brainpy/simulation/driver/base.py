# -*- coding: utf-8 -*-

import abc

__all__ = [
  'DSDriver',
]


class DSDriver(abc.ABC):
  """Dynamical System Driver.
  """

  def __init__(self, target):
    from brainpy.simulation.brainobjects.base import DynamicSystem
    assert isinstance(target, DynamicSystem)
    self.target = target

  @abc.abstractmethod
  def build(self, rebuild=False, inputs=()):
    pass

  def upload(self, name, data_or_func):
    """Upload the data or function to the node or the network.

    Establish the connection between the host and the driver. The
    driver can upload its specific data of functions to the host.
    Then, at the frontend of the host, users can call such functions
    or data by "host.func_name" or "host.some_data".
    """
    setattr(self.target, name, data_or_func)

  @abc.abstractmethod
  def get_input_func(self, *args, **kwargs):
    pass

  @abc.abstractmethod
  def get_monitor_func(self, *args, **kwargs):
    pass
