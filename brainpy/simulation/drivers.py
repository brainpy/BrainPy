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

  @abc.abstractmethod
  def upload(self, *args, **kwargs):
    """Upload the data or function to the node or the network.

    Establish the connection between the host and the driver. The
    driver can upload its specific data of functions to the host.
    Then, at the frontend of the host, users can call such functions
    or data by "host.func_name" or "host.some_data".
    """
    pass


class BaseDSDriver(AbstractDriver):
  """Base Dynamical System Driver.
  """

  def __init__(self, target):
    self.target = target

  def upload(self, name, data_or_func):
    setattr(self.target, name, data_or_func)

  @abc.abstractmethod
  def get_input_func(self, *args, **kwargs):
    pass

  @abc.abstractmethod
  def get_monitor_func(self, *args, **kwargs):
    pass

  @abc.abstractmethod
  def get_steps_func(self, *args, **kwargs):
    pass


class BaseDiffIntDriver(AbstractDriver):
  """Base Integration Driver for Differential Equations.
  """

  def __init__(self, code_scope, code_lines, func_name, uploads, show_code):
    self.code_scope = code_scope
    self.code_lines = code_lines
    self.func_name = func_name
    self.uploads = uploads
    self.show_code = show_code

  def upload(self, host, key, value):
    setattr(host, key, value)
