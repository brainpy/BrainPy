# -*- coding: utf-8 -*-


import jax.ops
import numpy as np
from jax import numpy as jnp
from jax.tree_util import register_pytree_node

__all__ = [
  'ndarray',
]


# Ways to change values in a zero-dimensional array
# -----
# Reference: https://stackoverflow.com/questions/56954714/how-do-i-assign-to-a-zero-dimensional-numpy-array
#
#   >>> x = np.array(10)
# 1. index the original array with ellipsis or an empty tuple
#    >>> x[...] = 2
#    >>> x[()] = 2


class ndarray(object):
  """ndarray for JAX backend.

  Limitations
  -----------

  1. Do not support "out" argument in all methods.
  """
  __slots__ = "_value"
  _registered = False

  def __new__(cls, *args, **kwargs):
    if not cls._registered:
      flatten = lambda t: ((t.value,), None)
      unflatten = lambda aux_data, children: ndarray(*children)
      register_pytree_node(ndarray, flatten, unflatten)
      cls._registered = True
    return super().__new__(cls)

  def __init__(self, value):
    self._value = value

  @property
  def value(self):
    return self._value

  @value.setter
  def value(self, value):
    self._value = value.value if isinstance(value, ndarray) else value

  @property
  def dtype(self):
    return self._value.dtype

  @property
  def shape(self):
    return self._value.shape

  @property
  def ndim(self):
    return self._value.ndim

  def __repr__(self) -> str:
    lines = repr(self.value).split("\n")
    prefix = self.__class__.__name__ + "("
    lines[0] = prefix + lines[0]
    prefix = " " * len(prefix)
    for i in range(1, len(lines)):
      lines[i] = prefix + lines[i]
    lines[-1] = lines[-1] + ")"
    return "\n".join(lines)

  def __iter__(self):
    for v in self._value:
      yield v

  def __format__(self, format_spec: str) -> str:
    return format(self.value, format_spec)

  def __getitem__(self, index):
    if isinstance(index, tuple):
      index = tuple(x.value if isinstance(x, ndarray) else x for x in index)
    elif isinstance(index, ndarray):
      index = index.value
    return self.value[index]

  def __setitem__(self, index, value):
    if isinstance(index, tuple):
      index = tuple(x.value if isinstance(x, ndarray) else x for x in index)
    elif isinstance(index, ndarray):
      index = index.value
    if isinstance(value, ndarray):
      value = value.value
    self._value = jax.ops.index_update(self._value, jax.ops.index[index], value)

  # ---------- #
  # operations #
  # ---------- #

  def __bool__(self) -> bool:
    return bool(self._value)

  def __len__(self) -> int:
    return len(self._value)

  def __neg__(self):
    return type(self)(self._value.__neg__())

  def __pos__(self):
    return type(self)(self._value.__pos__())

  def __abs__(self):
    return type(self)(self._value.__abs__())

  def __invert__(self):
    return type(self)(self._value.__invert__())

  def __eq__(self, oc):
    return type(self)(self._value.__eq__(oc._value if isinstance(oc, ndarray) else oc))

  def __ne__(self, oc):
    return type(self)(self._value.__ne__(oc._value if isinstance(oc, ndarray) else oc))

  def __lt__(self, oc):
    return type(self)(self._value.__lt__(oc._value if isinstance(oc, ndarray) else oc))

  def __le__(self, oc):
    return type(self)(self._value.__le__(oc._value if isinstance(oc, ndarray) else oc))

  def __gt__(self, oc):
    return type(self)(self._value.__gt__(oc._value if isinstance(oc, ndarray) else oc))

  def __ge__(self, oc):
    return type(self)(self._value.__ge__(oc._value if isinstance(oc, ndarray) else oc))

  def __add__(self, oc):
    return type(self)(self._value.__add__(oc._value if isinstance(oc, ndarray) else oc))

  def __radd__(self, oc):
    return type(self)(self._value.__radd__(oc._value if isinstance(oc, ndarray) else oc))

  def __iadd__(self, oc):
    # a += b
    self._value += (oc._value if isinstance(oc, ndarray) else oc)
    return self

  def __sub__(self, oc):
    return type(self)(self._value.__sub__(oc._value if isinstance(oc, ndarray) else oc))

  def __rsub__(self, oc):
    return type(self)(self._value.__rsub__(oc._value if isinstance(oc, ndarray) else oc))

  def __isub__(self, oc):
    # a -= b
    self._value = self._value.__sub__(oc._value if isinstance(oc, ndarray) else oc)
    return self

  def __mul__(self, oc):
    return type(self)(self._value.__mul__(oc._value if isinstance(oc, ndarray) else oc))

  def __rmul__(self, oc):
    return type(self)(self._value.__rmul__(oc._value if isinstance(oc, ndarray) else oc))

  def __imul__(self, oc):
    # a *= b
    self._value = self._value.__mul__(oc._value if isinstance(oc, ndarray) else oc)
    return self

  def __div__(self, oc):
    return type(self)(self._value.__div__(oc._value if isinstance(oc, ndarray) else oc))

  def __rdiv__(self, oc):
    return type(self)(self._value.__rdiv__(oc._value if isinstance(oc, ndarray) else oc))

  def __truediv__(self, oc):
    return type(self)(self._value.__truediv__(oc._value if isinstance(oc, ndarray) else oc))

  def __rtruediv__(self, oc):
    return type(self)(self._value.__rtruediv__(oc._value if isinstance(oc, ndarray) else oc))

  def __itruediv__(self, oc):
    # a /= b
    self._value = self._value.__truediv__(oc._value if isinstance(oc, ndarray) else oc)
    return self

  def __floordiv__(self, oc):
    return type(self)(self._value.__floordiv__(oc._value if isinstance(oc, ndarray) else oc))

  def __rfloordiv__(self, oc):
    return type(self)(self._value.__rfloordiv__(oc._value if isinstance(oc, ndarray) else oc))

  def __ifloordiv__(self, oc):
    # a //= b
    self._value = self._value.__floordiv__(oc._value if isinstance(oc, ndarray) else oc)
    return self

  def __divmod__(self, oc):
    return type(self)(self._value.__divmod__(oc._value if isinstance(oc, ndarray) else oc))

  def __rdivmod__(self, oc):
    return type(self)(self._value.__rdivmod__(oc._value if isinstance(oc, ndarray) else oc))

  def __mod__(self, oc):
    return type(self)(self._value.__mod__(oc._value if isinstance(oc, ndarray) else oc))

  def __rmod__(self, oc):
    return type(self)(self._value.__rmod__(oc._value if isinstance(oc, ndarray) else oc))

  def __imod__(self, oc):
    # a %= b
    self._value = self._value.__mod__(oc._value if isinstance(oc, ndarray) else oc)
    return self

  def __pow__(self, oc):
    return type(self)(self._value.__pow__(oc._value if isinstance(oc, ndarray) else oc))

  def __rpow__(self, oc):
    return type(self)(self._value.__rpow__(oc._value if isinstance(oc, ndarray) else oc))

  def __ipow__(self, oc):
    # a **= b
    self._value = self._value.__pow__(oc._value if isinstance(oc, ndarray) else oc)
    return self

  def __matmul__(self, oc):
    return type(self)(self._value.__matmul__(oc._value if isinstance(oc, ndarray) else oc))

  def __rmatmul__(self, oc):
    return type(self)(self._value.__rmatmul__(oc._value if isinstance(oc, ndarray) else oc))

  def __imatmul__(self, oc):
    # a @= b
    self._value = self._value.__matmul__(oc._value if isinstance(oc, ndarray) else oc)
    return self

  def __and__(self, oc):
    return type(self)(self._value.__and__(oc._value if isinstance(oc, ndarray) else oc))

  def __rand__(self, oc):
    return type(self)(self._value.__rand__(oc._value if isinstance(oc, ndarray) else oc))

  def __iand__(self, oc):
    # a &= b
    self._value = self._value.__and__(oc._value if isinstance(oc, ndarray) else oc)
    return self

  def __or__(self, oc):
    return type(self)(self._value.__or__(oc._value if isinstance(oc, ndarray) else oc))

  def __ror__(self, oc):
    return type(self)(self._value.__ror__(oc._value if isinstance(oc, ndarray) else oc))

  def __ior__(self, oc):
    # a |= b
    self._value = self._value.__or__(oc._value if isinstance(oc, ndarray) else oc)
    return self

  def __xor__(self, oc):
    return type(self)(self._value.__xor__(oc._value if isinstance(oc, ndarray) else oc))

  def __rxor__(self, oc):
    return type(self)(self._value.__rxor__(oc._value if isinstance(oc, ndarray) else oc))

  def __ixor__(self, oc):
    # a ^= b
    self._value = self._value.__xor__(oc._value if isinstance(oc, ndarray) else oc)
    return self

  def __lshift__(self, oc):
    return type(self)(self._value.__lshift__(oc._value if isinstance(oc, ndarray) else oc))

  def __rlshift__(self, oc):
    return type(self)(self._value.__rlshift__(oc._value if isinstance(oc, ndarray) else oc))

  def __ilshift__(self, oc):
    # a <<= b
    self._value = self._value.__lshift__(oc._value if isinstance(oc, ndarray) else oc)
    return self

  def __rshift__(self, oc):
    return type(self)(self._value.__rshift__(oc._value if isinstance(oc, ndarray) else oc))

  def __rrshift__(self, oc):
    return type(self)(self._value.__rrshift__(oc._value if isinstance(oc, ndarray) else oc))

  def __irshift__(self, oc):
    # a >>= b
    self._value = self._value.__rshift__(oc._value if isinstance(oc, ndarray) else oc)
    return self

  def __round__(self, ndigits=None):
    return type(self)(self._value.__round__(ndigits))

  def at(self, *args):
    raise NotImplementedError

  def aval(self, *args):
    raise NotImplementedError

  def block_host_until_ready(self, *args):
    raise NotImplementedError

  def block_until_ready(self, *args):
    raise NotImplementedError

  def broadcast(self, *args):
    raise NotImplementedError

  def client(self, *args):
    raise NotImplementedError

  def clone(self, *args):
    raise NotImplementedError

  def copy_to_device(self, *args):
    raise NotImplementedError

  def copy_to_host_async(self, *args):
    raise NotImplementedError

  def device(self, *args):
    raise NotImplementedError

  def device_buffer(self, *args):
    raise NotImplementedError

  def imag(self):
    return self._value.image

  def all(self, axis=None, keepdims=False, *args, **kwargs):
    return self.value.all(axis=axis, keepdims=keepdims, *args, **kwargs)

  def any(self, axis=None, keepdims=False, *args, **kwargs):
    return self.value.any(axis=axis, keepdims=keepdims, *args, **kwargs)

  def argmax(self, axis=None):
    return type(self)(self.value.argmax(axis=axis))

  def argmin(self, axis=None):
    return type(self)(self.value.argmin(axis=axis))

  def argpartition(self, kth, axis=-1, kind='introselect', order=None):
    res = self.value.argpartition(kth=kth, axis=axis, kind=kind, order=order)
    return type(self)(res) if isinstance(res, jnp.ndarray) else res

  def argsort(self, axis=-1, kind=None, order=None):
    return type(self)(self.value.argsort(axis=axis, kind=kind, order=order))

  def astype(self, dtype, order='K', casting='unsafe', subok=True, copy=True):
    return type(self)(self.value.astype(dtype=dtype, order=order, casting=casting, subok=subok, copy=copy))

  def byteswap(self, inplace=False):
    return type(self)(self.value.byteswap(inplace=inplace))

  def choose(self, choices, mode='raise'):
    choices = choices.value if isinstance(choices, ndarray) else choices
    return type(self)(self.value.choose(choices=choices, mode=mode))

  def clip(self, min=None, max=None, **kwargs):
    return type(self)(self.value.clip(min=min, max=max, **kwargs))

  def compress(self, condition, axis=None):
    condition = condition.value if isinstance(condition, ndarray) else condition
    return type(self)(self.value.compress(condition=condition, axis=axis))

  def conj(self):
    return type(self)(self.value.conj())

  def conjugate(self):
    return type(self)(self.value.conjugate())

  def copy(self, order='C'):
    return type(self)(self.value.copy(order=order))

  def cumprod(self, axis=None, dtype=None):
    return type(self)(self.value.cumprod(axis=axis, dtype=dtype))

  def cumsum(self, axis=None, dtype=None):
    return type(self)(self.value.cumsum(axis=axis, dtype=dtype))

  def diagonal(self, offset=0, axis1=0, axis2=1):
    return type(self)(self.value.diagonal(offset=offset, axis1=axis1, axis2=axis2))

  def dot(self, b):
    return type(self)(self.value.dot(b))

  def dump(self, file):
    self.value.dump(file=file)

  def dumps(self):
    return self.value.dumps()

  def fill(self, value):
    self.value.fill(value)

  def flatten(self, order='C'):
    return type(self)(self.value.flatten(order=order))

  def getfield(self, dtype, offset=0):
    return type(self)(self.value.getfield(dtype=dtype, offset=offset))

  def item(self, *args):
    return self.value.item(*args)

  def itemset(self, *args):
    self.value.itemset(*args)

  def max(self, axis=None, keepdims=False, *args, **kwargs):
    res = self.value.max(axis=axis, keepdims=keepdims, *args, **kwargs)
    return type(self)(res) if isinstance(res, jnp.ndarray) else res

  def mean(self, axis=None, dtype=None, keepdims=False, *args, **kwargs):
    res = self.value.mean(axis=axis, dtype=dtype, keepdims=keepdims, *args, **kwargs)
    return type(self)(res) if isinstance(res, jnp.ndarray) else res

  def min(self, axis=None, keepdims=False, *args, **kwargs):
    res = self.value.min(axis=axis, keepdims=keepdims, *args, **kwargs)
    return type(self)(res) if isinstance(res, jnp.ndarray) else res

  def newbyteorder(self, new_order='S', *args, **kwargs):
    return type(self)(self.value.newbyteorder(new_order=new_order, *args, **kwargs))

  def nonzero(self):
    return tuple(ndarray(a) for a in self.value.nonzero())

  def partition(self, kth, axis=-1, kind='introselect', order=None):
    return type(self)(self.value.partition(kth=kth, axis=axis, kind=kind, order=order))

  def prod(self, axis=None, dtype=None, keepdims=False, initial=1, where=True):
    res = self.value.prod(axis=axis, dtype=dtype, keepdims=keepdims, initial=initial, where=where)
    return type(self)(res) if isinstance(res, jnp.ndarray) else res

  def ptp(self, axis=None, keepdims=False):
    return type(self)(self.value.ptp(axis=axis, keepdims=keepdims))

  def ravel(self, order=None):
    return type(self)(self.value.ravel(order=order))

  def real(self):
    raise NotImplementedError

  def repeat(self, repeats, axis=None):
    return type(self)(self.value.repeat(repeats=repeats, axis=axis))

  def reshape(self, shape, order='C'):
    return type(self)(self.value.reshape(*shape, order=order))

  def round(self, decimals=0):
    return type(self)(self.value.round(decimals=decimals))

  def searchsorted(self, v, side='left', sorter=None):
    v = v.value if isinstance(v, ndarray) else v
    res = self.value.searchsorted(v=v, side=side, sorter=sorter)
    return type(self)(res) if isinstance(res, jnp.ndarray) else res

  def sort(self, axis=-1, kind=None, order=None):
    return type(self)(self.value.sort(axis=axis, kind=kind, order=order))

  def size(self):
    return self.value.size()

  def split(self, indices_or_sections, axis=0):
    return [type(self)(a) for a in self.value.split(indices_or_sections, axis=axis)]

  def squeeze(self, axis=None):
    return type(self)(self.value.squeeze(axis=axis))

  def std(self, axis=None, dtype=None, ddof=0, keepdims=False, *args, **kwargs):
    return type(self)(self.value.std(axis=axis, dtype=dtype, ddof=ddof, keepdims=keepdims,
                                     *args, **kwargs))

  def sum(self, axis=None, dtype=None, keepdims=False, initial=0, where=True):
    res = self.value.sum(axis=axis, dtype=dtype, keepdims=keepdims, initial=initial, where=where)
    return type(self)(res) if isinstance(res, jnp.ndarray) else res

  def swapaxes(self, axis1, axis2):
    return type(self)(self.value.swapaxes(axis1, axis2))

  def take(self, indices, axis=None, mode='raise'):
    indices = indices.value if isinstance(indices, ndarray) else indices
    return type(self)(self.value.take(indices=indices, axis=axis, mode=mode))

  def tobytes(self, order='C'):
    return type(self)(self.value.tobytes(order=order))

  def tile(self, reps):
    reps = reps.value if isinstance(reps, ndarray) else reps
    return type(self)(self.value.tile(reps))

  def tolist(self):
    return self.value.tolist()

  def trace(self, offset=0, axis1=0, axis2=1, dtype=None):
    return type(self)(self.value.trace(offset=offset, axis1=axis1, axis2=axis2, dtype=dtype))

  def transpose(self, *axes):
    return type(self)(self.value.transpose(*axes))

  def var(self, axis=None, dtype=None, ddof=0, keepdims=False, *args, **kwargs):
    return type(self)(self.value.var(axis=axis, dtype=dtype, ddof=ddof, keepdims=keepdims, *args, **kwargs))

  def view(self, dtype=None, *args, **kwargs):
    res = self.value.view(dtype=dtype, *args, **kwargs)
    return type(self)(res) if isinstance(res, jnp.ndarray) else res

  def numpy(self):
    return np.asarray(self.value)


def _wrap(f):
  def func(*args, **kwargs):
    args = [a.value if isinstance(a, ndarray) else a for a in args]
    kwargs = {k: v.value if isinstance(v, ndarray) else v for k, v in kwargs.items()}
    result = f(*args, **kwargs)
    if result is None:
      return
    elif isinstance(result, (tuple, list)):
      return type(result)([ndarray(res) if isinstance(res, jnp.ndarray) else res for res in result])
    else:
      return ndarray(result) if isinstance(result, jnp.ndarray) else result

  return func
