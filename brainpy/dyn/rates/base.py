# -*- coding: utf-8 -*-

from brainpy.dyn.base import DynamicalSystem
from brainpy.tools.others import to_size, size2num
from brainpy.types import Shape

__all__ = [
  'RateModel',
]


class RateModel(DynamicalSystem):
  """Base class of rate models."""

  def __init__(self,
               size: Shape,
               name: str = None):
    super(RateModel, self).__init__(name=name)

    self.size = to_size(size)
    self.num = size2num(self.size)

  def update(self, _t, _dt):
    """The function to specify the updating rule.

    Parameters
    ----------
    _t : float
      The current time.
    _dt : float
      The time step.
    """
    raise NotImplementedError(f'Subclass of {self.__class__.__name__} must '
                              f'implement "update" function.')
