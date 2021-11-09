# -*- coding: utf-8 -*-

from typing import cast

import numpy as np
try:
  import numba as nb
except ModuleNotFoundError:
  nb = None


__all__ = [
  'ndarray',
  'Variable',
  'TrainVar',
  'Parameter',
]

ndarray = np.ndarray


class Variable(np.ndarray):
  """Variable.

  """
  def __new__(cls, value, type='', replicate=None):
    if not isinstance(value, ndarray):
      arr_value = np.asarray(value)
    else:
      arr_value = value
    obj = arr_value.view(cls)
    if arr_value.dtype == np.dtype('O'):
      if isinstance(value, (list, tuple)):
        if len(value) > 0 and isinstance(value[0], ndarray) and nb is not None:
          value2 = nb.typed.List()
          for v in value:
            value2.append(v)
          value = value2
      obj.value = value
    else:
      obj.value = arr_value
    obj.type = type
    obj.replicate = replicate
    return obj

  def __array_finalize__(self, obj):
    if obj is None: return
    self.replicate = getattr(obj, 'replicate', None)
    self.value = getattr(obj, 'value', None)
    self.type = getattr(obj, 'type', None)

  def issametype(self, other):
    if self.type:
      return not isinstance(other, Variable)
    else:
      if not isinstance(other, Variable):
        return False
      else:
        return other.type == self.type


class TrainVar(Variable):
  """Trainable Variable.

  """
  __slots__ = ()

  def __new__(cls, value, replicate=None):
    return cast(TrainVar, super().__new__(cls, value=value, type='train', replicate=replicate))


class Parameter(Variable):
  """Parameter.

  """
  __slots__ = ()

  def __new__(cls, value, replicate=None):
    return cast(TrainVar, super().__new__(cls, value=value, type='param', replicate=replicate))
