# -*- coding: utf-8 -*-

import numpy as np

try:
  import numba as nb
except ModuleNotFoundError:
  nb = None

__all__ = [
  'Variable',
]


class Variable(np.ndarray):
  """Variable for numpy ndarray.

  This is useful when applying JIT onto the class objects.
  """

  def __new__(cls, value):
    if not isinstance(value, np.ndarray):
      arr_value = np.asarray(value)
    else:
      arr_value = value
    obj = arr_value.view(cls)
    if arr_value.dtype == np.dtype('O'):
      if isinstance(value, (list, tuple)):
        if len(value) > 0 and isinstance(value[0], np.ndarray) and nb is not None:
          value2 = nb.typed.List()
          for v in value:
            value2.append(v)
          value = value2
      obj.value = value
    else:
      obj.value = arr_value
    return obj

  def __array_finalize__(self, obj):
    if obj is None: return
    self.value = getattr(obj, 'value', None)
