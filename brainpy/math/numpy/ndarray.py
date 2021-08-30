# -*- coding: utf-8 -*-

import numpy as np

__all__ = [
  'ndarray',
  'Variable',
  'TrainVar',
]

ndarray = np.ndarray


class Variable(np.ndarray):
  def __init__(self, value, type):
    self.value = value
    self.type = type

    super(Variable, self).__init__()

  def issametype(self, other):
    if self.type:
      return not isinstance(other, Variable)
    else:
      if not isinstance(other, Variable):
        return False
      else:
        return other.type == self.type


class TrainVar(Variable):
  def __init__(self, value):
    super(TrainVar, self).__init__(value, 'train')

  def issametype(self, other):
    if not isinstance(other, Variable):
      return False
    else:
      return other.type == self.type
