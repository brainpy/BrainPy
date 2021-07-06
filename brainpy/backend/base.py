# -*- coding: utf-8 -*-

import abc

__all__ = [
  'AbstractDriver',
  'BaseDSDriver',
  'BaseDiffIntDriver',
]


class AbstractDriver(abc.ABC):
  """
  Abstract base class for backend driver.
  """

  @abc.abstractmethod
  def build(self, *args, **kwargs):
    """Build the node or the network running function.
    """
    pass


class BaseDSDriver(AbstractDriver):
  """Base Dynamical System Driver.
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


class BaseDiffIntDriver(AbstractDriver):
  """Base Integration Driver for Differential Equations.
  """

  def __init__(self, code_scope, code_lines, func_name, show_code):
    self.code_scope = code_scope
    self.code_lines = code_lines
    self.func_name = func_name
    self.show_code = show_code
    self.code = None
