# -*- coding: utf-8 -*-
import jax.ops
from jax import numpy as jnp
from jax.tree_util import register_pytree_node


__all__ = [
  'ndarray',
]


class ndarray(object):
  __slots__ = "_value"

  _registered = False
  key = None

  def __new__(cls, *args, **kwargs):
    if not cls._registered:
      def flatten(t):
        return ((t.value,), None)

      def unflatten(aux_data, children):
        return cls(*children)

      register_pytree_node(cls, flatten, unflatten)
      cls._registered = True
    return super().__new__(cls)

  def __init__(self, value):
    self._value = value

  def __repr__(self) -> str:
    lines = repr(self.value).split("\n")
    prefix = self.__class__.__name__ + "("
    lines[0] = prefix + lines[0]
    prefix = " " * len(prefix)
    for i in range(1, len(lines)):
      lines[i] = prefix + lines[i]
    lines[-1] = lines[-1] + ")"
    return "\n".join(lines)

  def __format__(self, format_spec: str) -> str:
    return format(self.value, format_spec)

  @property
  def value(self):
    return self._value

  @value.setter
  def value(self, value):
    self._value = value

  @property
  def dtype(self):
    return self._value.dtype

  @property
  def shape(self):
    return self._value.shape

  @property
  def ndim(self):
    return self._value.ndim

  def __getitem__(self, index):
    if isinstance(index, tuple):
      index = tuple(x.value if isinstance(x, ndarray) else x for x in index)
    elif isinstance(index, ndarray):
      index = index.value
    return ndarray(self.value[index])

  def __setitem__(self, index, value):
    if isinstance(index, tuple):
      index = tuple(x.value if isinstance(x, ndarray) else x for x in index)
    elif isinstance(index, ndarray):
      index = index.value
    if isinstance(value, ndarray):
      value = value.value
    self.value = jax.ops.index_update(self.value,
                                      jax.ops.index[index],
                                      value)

  # operations

  def __bool__(self) -> bool:
    return bool(self.value)

  def __len__(self) -> int:
    return len(self.value)

  def __neg__(self):
    return ndarray(self._value.__neg__())

  def __pos__(self):
    return ndarray(self._value.__pos__())

  def __abs__(self):
    return ndarray(self._value.__abs__())

  def __invert__(self):
    return ndarray(self._value.__invert__())

  def __eq__(self, oc):
    return ndarray(self._value.__eq__(oc._value if isinstance(oc, ndarray) else oc))

  def __ne__(self, oc):
    return ndarray(self._value.__ne__(oc._value if isinstance(oc, ndarray) else oc))

  def __lt__(self, oc):
    return ndarray(self._value.__lt__(oc._value if isinstance(oc, ndarray) else oc))

  def __le__(self, oc):
    return ndarray(self._value.__le__(oc._value if isinstance(oc, ndarray) else oc))

  def __gt__(self, oc):
    return ndarray(self._value.__gt__(oc._value if isinstance(oc, ndarray) else oc))

  def __ge__(self, oc):
    return ndarray(self._value.__ge__(oc._value if isinstance(oc, ndarray) else oc))

  def __add__(self, oc):
    return ndarray(self._value.__add__(oc._value if isinstance(oc, ndarray) else oc))

  def __radd__(self, oc):
    return ndarray(self._value.__radd__(oc._value if isinstance(oc, ndarray) else oc))

  def __sub__(self, oc):
    return ndarray(self._value.__sub__(oc._value if isinstance(oc, ndarray) else oc))

  def __rsub__(self, oc):
    return ndarray(self._value.__rsub__(oc._value if isinstance(oc, ndarray) else oc))

  def __mul__(self, oc):
    return ndarray(self._value.__mul__(oc._value if isinstance(oc, ndarray) else oc))

  def __rmul__(self, oc):
    return ndarray(self._value.__rmul__(oc._value if isinstance(oc, ndarray) else oc))

  def __div__(self, oc):
    return ndarray(self._value.__div__(oc._value if isinstance(oc, ndarray) else oc))

  def __rdiv__(self, oc):
    return ndarray(self._value.__rdiv__(oc._value if isinstance(oc, ndarray) else oc))

  def __truediv__(self, oc):
    return ndarray(self._value.__truediv__(oc._value if isinstance(oc, ndarray) else oc))

  def __rtruediv__(self, oc):
    return ndarray(self._value.__rtruediv__(oc._value if isinstance(oc, ndarray) else oc))

  def __floordiv__(self, oc):
    return ndarray(self._value.__floordiv__(oc._value if isinstance(oc, ndarray) else oc))

  def __rfloordiv__(self, oc):
    return ndarray(self._value.__rfloordiv__(oc._value if isinstance(oc, ndarray) else oc))

  def __divmod__(self, oc):
    return ndarray(self._value.__divmod__(oc._value if isinstance(oc, ndarray) else oc))

  def __rdivmod__(self, oc):
    return ndarray(self._value.__rdivmod__(oc._value if isinstance(oc, ndarray) else oc))

  def __mod__(self, oc):
    return ndarray(self._value.__mod__(oc._value if isinstance(oc, ndarray) else oc))

  def __rmod__(self, oc):
    return ndarray(self._value.__rmod__(oc._value if isinstance(oc, ndarray) else oc))

  def __pow__(self, oc):
    return ndarray(self._value.__pow__(oc._value if isinstance(oc, ndarray) else oc))

  def __rpow__(self, oc):
    return ndarray(self._value.__rpow__(oc._value if isinstance(oc, ndarray) else oc))

  def __matmul__(self, oc):
    return ndarray(self._value.__matmul__(oc._value if isinstance(oc, ndarray) else oc))

  def __rmatmul__(self, oc):
    return ndarray(self._value.__rmatmul__(oc._value if isinstance(oc, ndarray) else oc))

  def __and__(self, oc):
    return ndarray(self._value.__and__(oc._value if isinstance(oc, ndarray) else oc))

  def __rand__(self, oc):
    return ndarray(self._value.__rand__(oc._value if isinstance(oc, ndarray) else oc))

  def __or__(self, oc):
    return ndarray(self._value.__or__(oc._value if isinstance(oc, ndarray) else oc))

  def __ror__(self, oc):
    return ndarray(self._value.__ror__(oc._value if isinstance(oc, ndarray) else oc))

  def __xor__(self, oc):
    return ndarray(self._value.__xor__(oc._value if isinstance(oc, ndarray) else oc))

  def __rxor__(self, oc):
    return ndarray(self._value.__rxor__(oc._value if isinstance(oc, ndarray) else oc))

  def __lshift__(self, oc):
    return ndarray(self._value.__lshift__(oc._value if isinstance(oc, ndarray) else oc))

  def __rlshift__(self, oc):
    return ndarray(self._value.__rlshift__(oc._value if isinstance(oc, ndarray) else oc))

  def __rshift__(self, oc):
    return ndarray(self._value.__rshift__(oc._value if isinstance(oc, ndarray) else oc))

  def __rrshift__(self, oc):
    return ndarray(self._value.__rrshift__(oc._value if isinstance(oc, ndarray) else oc))

  def __round__(self, ndigits=None):
    return ndarray(self._value.__round__(ndigits))


def _wrap(f):
  def func(*args, **kwargs):
    args = [a.value if isinstance(a, ndarray) else a
            for a in args]
    kwargs = {k: v.value if isinstance(v, ndarray) else v
              for k, v in kwargs.items()}
    result = f(*args, **kwargs)
    if result is None:
      return
    elif isinstance(result, (tuple, list)):
      return type(result)([ndarray(res) if isinstance(res, jnp.ndarray) else res for res in result])
    else:
      return ndarray(result) if isinstance(result, jnp.ndarray) else result

  return func
