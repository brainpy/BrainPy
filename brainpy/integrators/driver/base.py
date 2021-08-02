# -*- coding: utf-8 -*-

import abc


class DiffIntDriver(abc.ABC):
  """Driver for Integration of Differential Equations.
  """

  def __init__(self, code_scope, code_lines, func_name, show_code):
    self.code_scope = code_scope
    self.code_lines = code_lines
    self.func_name = func_name
    self.show_code = show_code
    self.code = None

  @abc.abstractmethod
  def build(self, *args, **kwargs):
    """Build the node or the network running function.
    """
    pass
