# -*- coding: utf-8 -*-


from brainpy.backend import ops

__all__ = [
  'Data'
]


class Data(object):
  def __init__(self, value, train=False):
    self._value = value
    self._train = train

  @property
  def trainable(self):
    return self._train

  @property
  def value(self):
    return self._value

  @value.setter
  def value(self, value):
    self._value = value

  def assign(self, value):
    """Assign a new value with the type and shape checking.
    """
    if isinstance(self._value, (int, float)):
      if not isinstance(value, type(self._value)):
        raise TypeError(f'"self._value" ({type(self._value)}) is not '
                        f'the same type as "value" ({type(value)}).')
      else:
        self._value = value
    else:
      if isinstance(value, (int, float)):
        self._value = ops.ones_like(self._value) * value
      else:
        if ops.shape(self._value) != ops.shape(value):
          raise TypeError(f'"self._value" has the shape of ({ops.shape(self._value)}), '
                          f'while the same type as "value" ({type(value)}).')
        self._value = value

  @property
  def dtype(self):
    if hasattr(self._value, 'dtype'):
      return self._value.dtype
    return type(self._value)

  @property
  def shape(self):
    if hasattr(self._value, 'shape'):
      return self._value.shape
    return ()

  @property
  def ndim(self):
    if hasattr(self._value, 'ndim'):
      return self._value.ndim
    return 0

  def __repr__(self):
    return f'{self.__class__.__name__}({repr(self._value)})'

  def __getitem__(self, idx):
    return self._value.__getitem__(idx)

  def __array__(self, dtype=None):
    return self._value.__array__(dtype)

  def __getattr__(self, name):
    return getattr(self._value, name)

  def __neg__(self):
    return self._value.__neg__()

  def __pos__(self):
    return self._value.__pos__()

  def __abs__(self):
    return self._value.__abs__()

  def __invert__(self):
    return self._value.__invert__()

  def __eq__(self, oc):
    return self._value.__eq__(oc.value if isinstance(oc, Data) else oc)

  def __ne__(self, oc):
    return self._value.__ne__(oc.value if isinstance(oc, Data) else oc)

  def __lt__(self, oc):
    return self._value.__lt__(oc.value if isinstance(oc, Data) else oc)

  def __le__(self, oc):
    return self._value.__le__(oc.value if isinstance(oc, Data) else oc)

  def __gt__(self, oc):
    return self._value.__gt__(oc.value if isinstance(oc, Data) else oc)

  def __ge__(self, oc):
    return self._value.__ge__(oc.value if isinstance(oc, Data) else oc)

  def __add__(self, oc):
    return self._value.__add__(oc.value if isinstance(oc, Data) else oc)

  def __radd__(self, oc):
    return self._value.__radd__(oc.value if isinstance(oc, Data) else oc)

  def __sub__(self, oc):
    return self._value.__sub__(oc.value if isinstance(oc, Data) else oc)

  def __rsub__(self, oc):
    return self._value.__rsub__(oc.value if isinstance(oc, Data) else oc)

  def __mul__(self, oc):
    return self._value.__mul__(oc.value if isinstance(oc, Data) else oc)

  def __rmul__(self, oc):
    return self._value.__rmul__(oc.value if isinstance(oc, Data) else oc)

  def __div__(self, oc):
    return self._value.__div__(oc.value if isinstance(oc, Data) else oc)

  def __rdiv__(self, oc):
    return self._value.__rdiv__(oc.value if isinstance(oc, Data) else oc)

  def __truediv__(self, oc):
    return self._value.__truediv__(oc.value if isinstance(oc, Data) else oc)

  def __rtruediv__(self, oc):
    return self._value.__rtruediv__(oc.value if isinstance(oc, Data) else oc)

  def __floordiv__(self, oc):
    return self._value.__floordiv__(oc.value if isinstance(oc, Data) else oc)

  def __rfloordiv__(self, oc):
    return self._value.__rfloordiv__(oc.value if isinstance(oc, Data) else oc)

  def __divmod__(self, oc):
    return self._value.__divmod__(oc.value if isinstance(oc, Data) else oc)

  def __rdivmod__(self, oc):
    return self._value.__rdivmod__(oc.value if isinstance(oc, Data) else oc)

  def __mod__(self, oc):
    return self._value.__mod__(oc.value if isinstance(oc, Data) else oc)

  def __rmod__(self, oc):
    return self._value.__rmod__(oc.value if isinstance(oc, Data) else oc)

  def __pow__(self, oc):
    return self._value.__pow__(oc.value if isinstance(oc, Data) else oc)

  def __rpow__(self, oc):
    return self._value.__rpow__(oc.value if isinstance(oc, Data) else oc)

  def __matmul__(self, oc):
    return self._value.__matmul__(oc.value if isinstance(oc, Data) else oc)

  def __rmatmul__(self, oc):
    return self._value.__rmatmul__(oc.value if isinstance(oc, Data) else oc)

  def __and__(self, oc):
    return self._value.__and__(oc.value if isinstance(oc, Data) else oc)

  def __rand__(self, oc):
    return self._value.__rand__(oc.value if isinstance(oc, Data) else oc)

  def __or__(self, oc):
    return self._value.__or__(oc.value if isinstance(oc, Data) else oc)

  def __ror__(self, oc):
    return self._value.__ror__(oc.value if isinstance(oc, Data) else oc)

  def __xor__(self, oc):
    return self._value.__xor__(oc.value if isinstance(oc, Data) else oc)

  def __rxor__(self, oc):
    return self._value.__rxor__(oc.value if isinstance(oc, Data) else oc)

  def __lshift__(self, oc):
    return self._value.__lshift__(oc.value if isinstance(oc, Data) else oc)

  def __rlshift__(self, oc):
    return self._value.__rlshift__(oc.value if isinstance(oc, Data) else oc)

  def __rshift__(self, oc):
    return self._value.__rshift__(oc.value if isinstance(oc, Data) else oc)

  def __rrshift__(self, oc):
    return self._value.__rrshift__(oc.value if isinstance(oc, Data) else oc)

  def __round__(self, ndigits=None):
    return self._value.__round__(ndigits)
