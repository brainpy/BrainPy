# -*- coding: utf-8 -*-


from .base import Data

__all__ = [
  'Par',
]


class Par(Data):
  """`Par` is used to contain data for parameters.

  If `train=True`, this parameter is trainable.

  """

  def __init__(self, value, train=False):
    super(Par, self).__init__(value=value, train=train)
