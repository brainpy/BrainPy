# -*- coding: utf-8 -*-
import typing
from typing import Union, Optional, NoReturn, Sequence, Any, Tuple as TupleType
import warnings
import operator

import jax
import numpy as np
from jax import numpy as jnp
from jax.dtypes import canonicalize_dtype
from jax.tree_util import register_pytree_node

import brainpy.math
from brainpy.errors import MathError

__all__ = [
  'Array', 'ndarray', 'JaxArray',  # alias of Array
  'Variable',
  'TrainVar',
  'Parameter',
  'VariableView',
]

# Ways to change values in a zero-dimensional array
# -----
# Reference: https://stackoverflow.com/questions/56954714/how-do-i-assign-to-a-zero-dimensional-numpy-array
#
#   >>> x = np.array(10)
# 1. index the original array with ellipsis or an empty tuple
#    >>> x[...] = 2
#    >>> x[()] = 2

_all_slice = slice(None, None, None)

msg = ('ArrayType created outside of the jit function '
       'cannot be updated in JIT mode. You should '
       'mark it as brainpy.math.Variable instead.')

_jax_transformation_context_ = []


def add_context(name) -> None:
  _jax_transformation_context_.append(name)


def del_context(name=None) -> None:
  try:
    context = _jax_transformation_context_.pop(-1)
    if name is not None:
      if context != name:
        raise MathError('Transformation context is different!')
  except IndexError:
    raise MathError('No transformation context!')


def get_context():
  if len(_jax_transformation_context_) > 0:
    return _jax_transformation_context_[-1]
  else:
    return None


def _check_input_array(array):
  if isinstance(array, Array):
    return array.value
  elif isinstance(array, np.ndarray):
    return jnp.asarray(array)
  else:
    return array


def _return(a):
  if isinstance(a, jax.Array) and a.ndim > 0:
    return Array(a)
  return a


def _as_jax_array_(obj):
  return obj.value if isinstance(obj, Array) else obj


BmArray = Union['Array', jax.Array, np.ndarray]
OptionalBmArray = Optional[Union['Array', jax.Array, np.ndarray]]
ExceptionOrArray = Union['Array', NoReturn]

class Array(object):
  """Multiple-dimensional array in BrainPy.
  """

  is_brainpy_array = True
  _need_check_context = True
  __slots__ = ("_value", "_transform_context")

  def __init__(self, value, dtype=None):
    # array value
    if isinstance(value, Array):
      value = value._value
    elif isinstance(value, (tuple, list, np.ndarray)):
      value = jnp.asarray(value)
    if dtype is not None:
      value = jnp.asarray(value, dtype=dtype)
    self._value = value
    # jit mode
    self._transform_context = get_context()

  def __check_context(self) -> None:
    # raise error when in-place updating a
    if self._need_check_context:
      if self._transform_context is None:
        if len(_jax_transformation_context_) > 0:
          raise MathError(f'Array created outside of the transformation functions '
                          f'({_jax_transformation_context_[-1]}) cannot be updated. '
                          f'You should mark it as a brainpy.math.Variable instead.')
      else:
        if len(_jax_transformation_context_) > 0:
          if self._transform_context != _jax_transformation_context_[-1]:
            raise MathError(f'Array context "{self._transform_context}" differs from the JAX '
                            f'transformation context "{_jax_transformation_context_[-1]}"'
                            '\n\n'
                            'Array created in one transformation function '
                            'cannot be updated another transformation function. '
                            'You should mark it as a brainpy.math.Variable instead.')

  @property
  def value(self):
    return self._value

  @value.setter
  def value(self, value):
    self.update(value)

  def update(self, value):
    """Update the value of this Array.
    """
    if isinstance(value, Array):
      value = value.value
    elif isinstance(value, np.ndarray):
      value = jnp.asarray(value)
    elif isinstance(value, jax.Array):
      pass
    else:
      value = jnp.asarray(value)
    # check
    if value.shape != self.value.shape:
      raise MathError(f"The shape of the original data is {self.value.shape}, "
                      f"while we got {value.shape}.")
    if value.dtype != self.value.dtype:
      raise MathError(f"The dtype of the original data is {self.value.dtype}, "
                      f"while we got {value.dtype}.")
    self._value = value.value if isinstance(value, Array) else value

  @property
  def dtype(self):
    """Variable dtype."""
    return self._value.dtype

  @property
  def shape(self):
    """Variable shape."""
    return self.value.shape

  @property
  def ndim(self):
    return self.value.ndim

  @property
  def imag(self):
    return _return(self.value.image)

  @property
  def real(self):
    return _return(self.value.real)

  @property
  def size(self):
    return self.value.size

  @property
  def T(self):
    return _return(self.value.T)

  # ----------------------- #
  # Python inherent methods #
  # ----------------------- #

  def __repr__(self) -> str:
    print_code = repr(self.value)
    if ', dtype' in print_code:
      print_code = print_code.split(', dtype')[0] + ')'
    prefix = f'{self.__class__.__name__}'
    prefix2 = f'{self.__class__.__name__}(value='
    if '\n' in print_code:
      lines = print_code.split("\n")
      blank1 = " " * len(prefix2)
      lines[0] = prefix2 + lines[0]
      for i in range(1, len(lines)):
        lines[i] = blank1 + lines[i]
      lines[-1] += ","
      blank2 = " " * (len(prefix) + 1)
      lines.append(f'{blank2}dtype={self.dtype})')
      print_code = "\n".join(lines)
    else:
      print_code = prefix2 + print_code + f', dtype={self.dtype})'
    return print_code

  def __format__(self, format_spec: str) -> str:
    return format(self.value)

  def __iter__(self):
    """Solve the issue of DeviceArray.__iter__.

    Details please see JAX issues:

    - https://github.com/google/jax/issues/7713
    - https://github.com/google/jax/pull/3821
    """
    for v in self.value:
      yield v

  def __getitem__(self, index):
    if isinstance(index, slice) and (index == _all_slice):
      return self.value
    elif isinstance(index, tuple):
      index = tuple((x.value if isinstance(x, Array) else x) for x in index)
    elif isinstance(index, Array):
      index = index.value
    return self.value[index]

  def __setitem__(self, index, value):
    # value is Array
    if isinstance(value, Array):
      value = value.value
    # value is numpy.ndarray
    elif isinstance(value, np.ndarray):
      value = jnp.asarray(value)

    # index is a tuple
    if isinstance(index, tuple):
      index = tuple(_check_input_array(x) for x in index)
    # index is Array
    elif isinstance(index, Array):
      index = index.value
    # index is numpy.ndarray
    elif isinstance(index, np.ndarray):
      index = jnp.asarray(index)

    # update
    self.value = self.value.at[index].set(value)

  # ---------- #
  # operations #
  # ---------- #

  def __len__(self) -> int:
    return len(self.value)

  def __neg__(self):
    return _return(self.value.__neg__())

  def __pos__(self):
    return _return(self.value.__pos__())

  def __abs__(self):
    return _return(self.value.__abs__())

  def __invert__(self):
    return _return(self.value.__invert__())

  def __eq__(self, oc):
    return _return(self.value == _check_input_array(oc))

  def __ne__(self, oc):
    return _return(self.value != _check_input_array(oc))

  def __lt__(self, oc):
    return _return(self.value < _check_input_array(oc))

  def __le__(self, oc):
    return _return(self.value <= _check_input_array(oc))

  def __gt__(self, oc):
    return _return(self.value > _check_input_array(oc))

  def __ge__(self, oc):
    return _return(self.value >= _check_input_array(oc))

  def __add__(self, oc):
    return _return(self.value + _check_input_array(oc))

  def __radd__(self, oc):
    return _return(self.value + _check_input_array(oc))

  def __iadd__(self, oc):
    # a += b
    self.value = self.value + _check_input_array(oc)
    return self

  def __sub__(self, oc):
    return _return(self.value - _check_input_array(oc))

  def __rsub__(self, oc):
    return _return(_check_input_array(oc) - self.value)

  def __isub__(self, oc):
    # a -= b
    self.value = self.value - _check_input_array(oc)
    return self

  def __mul__(self, oc):
    return _return(self.value * _check_input_array(oc))

  def __rmul__(self, oc):
    return _return(_check_input_array(oc) * self.value)

  def __imul__(self, oc):
    # a *= b
    self.value = self.value * _check_input_array(oc)
    return self

  def __rdiv__(self, oc):
    return _return(_check_input_array(oc) / self.value)

  def __truediv__(self, oc):
    return _return(self.value / _check_input_array(oc))

  def __rtruediv__(self, oc):
    return _return(_check_input_array(oc) / self.value)

  def __itruediv__(self, oc):
    # a /= b
    self.value = self.value / _check_input_array(oc)
    return self

  def __floordiv__(self, oc):
    return _return(self.value // _check_input_array(oc))

  def __rfloordiv__(self, oc):
    return _return(_check_input_array(oc) // self.value)

  def __ifloordiv__(self, oc):
    # a //= b
    self.value = self.value // _check_input_array(oc)
    return self

  def __divmod__(self, oc):
    return _return(self.value.__divmod__(_check_input_array(oc)))

  def __rdivmod__(self, oc):
    return _return(self.value.__rdivmod__(_check_input_array(oc)))

  def __mod__(self, oc):
    return _return(self.value % _check_input_array(oc))

  def __rmod__(self, oc):
    return _return(_check_input_array(oc) % self.value)

  def __imod__(self, oc):
    # a %= b
    self.value = self.value % _check_input_array(oc)
    return self

  def __pow__(self, oc):
    return _return(self.value ** _check_input_array(oc))

  def __rpow__(self, oc):
    return _return(_check_input_array(oc) ** self.value)

  def __ipow__(self, oc):
    # a **= b
    self.value = self.value ** _check_input_array(oc)
    return self

  def __matmul__(self, oc):
    return _return(self.value @ _check_input_array(oc))

  def __rmatmul__(self, oc):
    return _return(_check_input_array(oc) @ self.value)

  def __imatmul__(self, oc):
    # a @= b
    self.value = self.value @ _check_input_array(oc)
    return self

  def __and__(self, oc):
    return _return(self.value & _check_input_array(oc))

  def __rand__(self, oc):
    return _return(_check_input_array(oc) & self.value)

  def __iand__(self, oc):
    # a &= b
    self.value = self.value & _check_input_array(oc)
    return self

  def __or__(self, oc):
    return _return(self.value | _check_input_array(oc))

  def __ror__(self, oc):
    return _return(_check_input_array(oc) | self.value)

  def __ior__(self, oc):
    # a |= b
    self.value = self.value | _check_input_array(oc)
    return self

  def __xor__(self, oc):
    return _return(self.value ^ _check_input_array(oc))

  def __rxor__(self, oc):
    return _return(_check_input_array(oc) ^ self.value)

  def __ixor__(self, oc):
    # a ^= b
    self.value = self.value ^ _check_input_array(oc)
    return self

  def __lshift__(self, oc):
    return _return(self.value << _check_input_array(oc))

  def __rlshift__(self, oc):
    return _return(_check_input_array(oc) << self.value)

  def __ilshift__(self, oc):
    # a <<= b
    self.value = self.value << _check_input_array(oc)
    return self

  def __rshift__(self, oc):
    return _return(self.value >> _check_input_array(oc))

  def __rrshift__(self, oc):
    return _return(_check_input_array(oc) >> self.value)

  def __irshift__(self, oc):
    # a >>= b
    self.value = self.value >> _check_input_array(oc)
    return self

  def __round__(self, ndigits=None):
    return _return(self.value.__round__(ndigits))

  # ----------------------- #
  #       JAX methods       #
  # ----------------------- #

  @property
  def at(self):
    return self.value.at

  def block_host_until_ready(self, *args):
    return self.value.block_host_until_ready(*args)

  def block_until_ready(self, *args):
    return self.value.block_until_ready(*args)

  def device(self):
    return self.value.device()

  @property
  def device_buffer(self):
    return self.value.device_buffer

  # ----------------------- #
  #      NumPy methods      #
  # ----------------------- #

  def all(self, axis=None, keepdims=False):
    """Returns True if all elements evaluate to True."""
    r = self.value.all(axis=axis, keepdims=keepdims)
    return _return(r)

  def any(self, axis=None, keepdims=False):
    """Returns True if any of the elements of a evaluate to True."""
    r = self.value.any(axis=axis, keepdims=keepdims)
    return _return(r)

  def argmax(self, axis=None):
    """Return indices of the maximum values along the given axis."""
    return _return(self.value.argmax(axis=axis))

  def argmin(self, axis=None):
    """Return indices of the minimum values along the given axis."""
    return _return(self.value.argmin(axis=axis))

  def argpartition(self, kth, axis=-1, kind='introselect', order=None):
    """Returns the indices that would partition this array."""
    return _return(self.value.argpartition(kth=kth, axis=axis, kind=kind, order=order))

  def argsort(self, axis=-1, kind=None, order=None):
    """Returns the indices that would sort this array."""
    return _return(self.value.argsort(axis=axis, kind=kind, order=order))

  def astype(self, dtype):
    """Copy of the array, cast to a specified type.

    Parameters
    ----------
    dtype: str, dtype
      Typecode or data-type to which the array is cast.
    """
    if dtype is None:
      return _return(self.value)
    else:
      return _return(self.value.astype(dtype))

  def byteswap(self, inplace=False):
    """Swap the bytes of the array elements

    Toggle between low-endian and big-endian data representation by
    returning a byteswapped array, optionally swapped in-place.
    Arrays of byte-strings are not swapped. The real and imaginary
    parts of a complex number are swapped individually."""
    return _return(self.value.byteswap(inplace=inplace))

  def choose(self, choices, mode='raise'):
    """Use an index array to construct a new array from a set of choices."""
    return _return(self.value.choose(choices=_as_jax_array_(choices), mode=mode))

  def clip(self, min=None, max=None):
    """Return an array whose values are limited to [min, max]. One of max or min must be given."""
    return _return(self.value.clip(min=min, max=max))

  def compress(self, condition, axis=None):
    """Return selected slices of this array along given axis."""
    return _return(self.value.compress(condition=_as_jax_array_(condition), axis=axis))

  def conj(self):
    """Complex-conjugate all elements."""
    return _return(self.value.conj())

  def conjugate(self):
    """Return the complex conjugate, element-wise."""
    return _return(self.value.conjugate())

  def copy(self):
    """Return a copy of the array."""
    return _return(self.value.copy())

  def cumprod(self, axis=None, dtype=None):
    """Return the cumulative product of the elements along the given axis."""
    return _return(self.value.cumprod(axis=axis, dtype=dtype))

  def cumsum(self, axis=None, dtype=None):
    """Return the cumulative sum of the elements along the given axis."""
    return _return(self.value.cumsum(axis=axis, dtype=dtype))

  def diagonal(self, offset=0, axis1=0, axis2=1):
    """Return specified diagonals."""
    return _return(self.value.diagonal(offset=offset, axis1=axis1, axis2=axis2))

  def dot(self, b):
    """Dot product of two arrays."""
    return _return(self.value.dot(_as_jax_array_(b)))

  def fill(self, value):
    """Fill the array with a scalar value."""
    self.value = jnp.ones_like(self.value) * value

  def flatten(self):
    return _return(self.value.flatten())

  def item(self, *args):
    """Copy an element of an array to a standard Python scalar and return it."""
    return self.value.item(*args)

  def max(self, axis=None, keepdims=False, *args, **kwargs):
    """Return the maximum along a given axis."""
    res = self.value.max(axis=axis, keepdims=keepdims, *args, **kwargs)
    return _return(res)

  def mean(self, axis=None, dtype=None, keepdims=False, *args, **kwargs):
    """Returns the average of the array elements along given axis."""
    res = self.value.mean(axis=axis, dtype=dtype, keepdims=keepdims, *args, **kwargs)
    return _return(res)

  def min(self, axis=None, keepdims=False, *args, **kwargs):
    """Return the minimum along a given axis."""
    res = self.value.min(axis=axis, keepdims=keepdims, *args, **kwargs)
    return _return(res)

  def nonzero(self):
    """Return the indices of the elements that are non-zero."""
    return tuple(_return(a) for a in self.value.nonzero())

  def prod(self, axis=None, dtype=None, keepdims=False, initial=1, where=True):
    """Return the product of the array elements over the given axis."""
    res = self.value.prod(axis=axis, dtype=dtype, keepdims=keepdims, initial=initial, where=where)
    return _return(res)

  def ptp(self, axis=None, keepdims=False):
    """Peak to peak (maximum - minimum) value along a given axis."""
    r = self.value.ptp(axis=axis, keepdims=keepdims)
    return _return(r)

  def put(self, indices, values):
    """Replaces specified elements of an array with given values.

    Parameters
    ----------
    indices: array_like
      Target indices, interpreted as integers.
    values: array_like
      Values to place in the array at target indices.
    """
    self.__setitem__(indices, values)

  def ravel(self, order=None):
    """Return a flattened array."""
    return _return(self.value.ravel(order=order))

  def repeat(self, repeats, axis=None):
    """Repeat elements of an array."""
    return _return(self.value.repeat(repeats=repeats, axis=axis))

  def reshape(self, *shape, order='C'):
    """Returns an array containing the same data with a new shape."""
    return _return(self.value.reshape(*shape, order=order))

  def resize(self, new_shape):
    """Change shape and size of array in-place."""
    self.value = self.value.reshape(new_shape)

  def round(self, decimals=0):
    """Return ``a`` with each element rounded to the given number of decimals."""
    return _return(self.value.round(decimals=decimals))

  def searchsorted(self, v, side='left', sorter=None):
    """Find indices where elements should be inserted to maintain order.

    Find the indices into a sorted array `a` such that, if the
    corresponding elements in `v` were inserted before the indices, the
    order of `a` would be preserved.

    Assuming that `a` is sorted:

    ======  ============================
    `side`  returned index `i` satisfies
    ======  ============================
    left    ``a[i-1] < v <= a[i]``
    right   ``a[i-1] <= v < a[i]``
    ======  ============================

    Parameters
    ----------
    v : array_like
        Values to insert into `a`.
    side : {'left', 'right'}, optional
        If 'left', the index of the first suitable location found is given.
        If 'right', return the last such index.  If there is no suitable
        index, return either 0 or N (where N is the length of `a`).
    sorter : 1-D array_like, optional
        Optional array of integer indices that sort array a into ascending
        order. They are typically the result of argsort.

    Returns
    -------
    indices : array of ints
        Array of insertion points with the same shape as `v`.
    """
    return _return(self.value.searchsorted(v=_as_jax_array_(v), side=side, sorter=sorter))

  def sort(self, axis=-1, kind='quicksort', order=None):
    """Sort an array in-place.

    Parameters
    ----------
    axis : int, optional
        Axis along which to sort. Default is -1, which means sort along the
        last axis.
    kind : {'quicksort', 'mergesort', 'heapsort', 'stable'}
        Sorting algorithm. The default is 'quicksort'. Note that both 'stable'
        and 'mergesort' use timsort under the covers and, in general, the
        actual implementation will vary with datatype. The 'mergesort' option
        is retained for backwards compatibility.
    order : str or list of str, optional
        When `a` is an array with fields defined, this argument specifies
        which fields to compare first, second, etc.  A single field can
        be specified as a string, and not all fields need be specified,
        but unspecified fields will still be used, in the order in which
        they come up in the dtype, to break ties.
    """
    self.value = self.value.sort(axis=axis, kind=kind, order=order)

  def squeeze(self, axis=None):
    """Remove axes of length one from ``a``."""
    return _return(self.value.squeeze(axis=axis))

  def std(self, axis=None, dtype=None, ddof=0, keepdims=False):
    """Compute the standard deviation along the specified axis.

    Returns the standard deviation, a measure of the spread of a distribution,
    of the array elements. The standard deviation is computed for the
    flattened array by default, otherwise over the specified axis.

    Parameters
    ----------
    axis : None or int or tuple of ints, optional
        Axis or axes along which the standard deviation is computed. The
        default is to compute the standard deviation of the flattened array.
        If this is a tuple of ints, a standard deviation is performed over
        multiple axes, instead of a single axis or all the axes as before.
    dtype : dtype, optional
        Type to use in computing the standard deviation. For arrays of
        integer type the default is float64, for arrays of float types it is
        the same as the array type.
    ddof : int, optional
        Means Delta Degrees of Freedom.  The divisor used in calculations
        is ``N - ddof``, where ``N`` represents the number of elements.
        By default `ddof` is zero.
    keepdims : bool, optional
        If this is set to True, the axes which are reduced are left
        in the result as dimensions with size one. With this option,
        the result will broadcast correctly against the input array.

        If the default value is passed, then `keepdims` will not be
        passed through to the `std` method of sub-classes of
        `ndarray`, however any non-default value will be.  If the
        sub-class' method does not implement `keepdims` any
        exceptions will be raised.

    Returns
    -------
    standard_deviation : ndarray, see dtype parameter above.
        If `out` is None, return a new array containing the standard deviation,
        otherwise return a reference to the output array.
    """
    r = self.value.std(axis=axis, dtype=dtype, ddof=ddof, keepdims=keepdims)
    return _return(r)

  def sum(self, axis=None, dtype=None, keepdims=False, initial=0, where=True):
    """Return the sum of the array elements over the given axis."""
    res = self.value.sum(axis=axis, dtype=dtype, keepdims=keepdims, initial=initial, where=where)
    return _return(res)

  def swapaxes(self, axis1, axis2):
    """Return a view of the array with `axis1` and `axis2` interchanged."""
    return _return(self.value.swapaxes(axis1, axis2))

  def split(self, indices_or_sections, axis=0):
    """Split an array into multiple sub-arrays as views into ``ary``.

    Parameters
    ----------
    indices_or_sections : int, 1-D array
      If `indices_or_sections` is an integer, N, the array will be divided
      into N equal arrays along `axis`.  If such a split is not possible,
      an error is raised.

      If `indices_or_sections` is a 1-D array of sorted integers, the entries
      indicate where along `axis` the array is split.  For example,
      ``[2, 3]`` would, for ``axis=0``, result in

        - ary[:2]
        - ary[2:3]
        - ary[3:]

      If an index exceeds the dimension of the array along `axis`,
      an empty sub-array is returned correspondingly.
    axis : int, optional
      The axis along which to split, default is 0.

    Returns
    -------
    sub-arrays : list of ndarrays
      A list of sub-arrays as views into `ary`.
    """
    return [_return(a) for a in self.value.split(indices_or_sections, axis=axis)]

  def take(self, indices, axis=None, mode=None):
    """Return an array formed from the elements of a at the given indices."""
    return _return(self.value.take(indices=_as_jax_array_(indices), axis=axis, mode=mode))

  def tobytes(self):
    """Construct Python bytes containing the raw data bytes in the array.

    Constructs Python bytes showing a copy of the raw contents of data memory.
    The bytes object is produced in C-order by default. This behavior is
    controlled by the ``order`` parameter."""
    return self.value.tobytes()

  def tolist(self):
    """Return the array as an ``a.ndim``-levels deep nested list of Python scalars.

    Return a copy of the array data as a (nested) Python list.
    Data items are converted to the nearest compatible builtin Python type, via
    the `~numpy.ndarray.item` function.

    If ``a.ndim`` is 0, then since the depth of the nested list is 0, it will
    not be a list at all, but a simple Python scalar.
    """
    return self.value.tolist()

  def trace(self, offset=0, axis1=0, axis2=1, dtype=None):
    """Return the sum along diagonals of the array."""
    return _return(self.value.trace(offset=offset, axis1=axis1, axis2=axis2, dtype=dtype))

  def transpose(self, *axes):
    """Returns a view of the array with axes transposed.

    For a 1-D array this has no effect, as a transposed vector is simply the
    same vector. To convert a 1-D array into a 2D column vector, an additional
    dimension must be added. `np.atleast2d(a).T` achieves this, as does
    `a[:, np.newaxis]`.
    For a 2-D array, this is a standard matrix transpose.
    For an n-D array, if axes are given, their order indicates how the
    axes are permuted (see Examples). If axes are not provided and
    ``a.shape = (i[0], i[1], ... i[n-2], i[n-1])``, then
    ``a.transpose().shape = (i[n-1], i[n-2], ... i[1], i[0])``.

    Parameters
    ----------
    axes : None, tuple of ints, or `n` ints

     * None or no argument: reverses the order of the axes.

     * tuple of ints: `i` in the `j`-th place in the tuple means `a`'s
       `i`-th axis becomes `a.transpose()`'s `j`-th axis.

     * `n` ints: same as an n-tuple of the same ints (this form is
       intended simply as a "convenience" alternative to the tuple form)

    Returns
    -------
    out : ndarray
        View of `a`, with axes suitably permuted.
    """
    return _return(self.value.transpose(*axes))

  def tile(self, reps):
    """Construct an array by repeating A the number of times given by reps.

    If `reps` has length ``d``, the result will have dimension of
    ``max(d, A.ndim)``.

    If ``A.ndim < d``, `A` is promoted to be d-dimensional by prepending new
    axes. So a shape (3,) array is promoted to (1, 3) for 2-D replication,
    or shape (1, 1, 3) for 3-D replication. If this is not the desired
    behavior, promote `A` to d-dimensions manually before calling this
    function.

    If ``A.ndim > d``, `reps` is promoted to `A`.ndim by pre-pending 1's to it.
    Thus for an `A` of shape (2, 3, 4, 5), a `reps` of (2, 2) is treated as
    (1, 1, 2, 2).

    Note : Although tile may be used for broadcasting, it is strongly
    recommended to use numpy's broadcasting operations and functions.

    Parameters
    ----------
    reps : array_like
        The number of repetitions of `A` along each axis.

    Returns
    -------
    c : ndarray
        The tiled output array.
    """
    return _return(self.value.tile(_as_jax_array_(reps)))

  def var(self, axis=None, dtype=None, ddof=0, keepdims=False):
    """Returns the variance of the array elements, along given axis."""
    r = self.value.var(axis=axis, dtype=dtype, ddof=ddof, keepdims=keepdims)
    return _return(r)

  def view(self, dtype=None, *args, **kwargs):
    """New view of array with the same data."""
    return _return(self.value.view(dtype=dtype, *args, **kwargs))

  # ------------------
  # NumPy support
  # ------------------

  def numpy(self, dtype=None):
    """Convert to numpy.ndarray."""
    # warnings.warn('Deprecated since 2.1.12. Please use ".to_numpy()" instead.', DeprecationWarning)
    return np.asarray(self.value, dtype=dtype)

  def to_numpy(self, dtype=None):
    """Convert to numpy.ndarray."""
    return np.asarray(self.value, dtype=dtype)

  def to_jax(self, dtype=None):
    """Convert to jax.numpy.ndarray."""
    if dtype is None:
      return self.value
    else:
      return jnp.asarray(self.value, dtype=dtype)

  def __array__(self, dtype=None):
    """Support ``numpy.array()`` and ``numpy.asarray()`` functions."""
    return np.asarray(self.value, dtype=dtype)

  def __jax_array__(self):
    return self.value

  def as_variable(self):
    """As an instance of Variable."""
    return Variable(self)

  def __format__(self, specification):
    return self.value.__format__(specification)

  def __bool__(self) -> bool:
    return self.value.__bool__()

  def __float__(self):
    return self.value.__float__()

  def __int__(self):
    return self.value.__int__()

  def __complex__(self):
    return self.value.__complex__()

  def __hex__(self):
    assert self.ndim == 0, 'hex only works on scalar values'
    return hex(self.value)  # type: ignore

  def __oct__(self):
    assert self.ndim == 0, 'oct only works on scalar values'
    return oct(self.value)  # type: ignore

  def __index__(self):
    return operator.index(self.value)

  def __dlpack__(self):
    from jax.dlpack import to_dlpack  # pylint: disable=g-import-not-at-top
    return to_dlpack(self.value)

  # **********************************
  # For Pytorch compitable
  # **********************************

  def unsqueeze(self, dim: int) -> 'Array':
    """
    Array.unsqueeze(dim) -> Array, or so called Tensor
    equals
    Array.expand_dims(dim)

    See :func:`brainpy.math.unsqueeze`
    """
    return Array(jnp.expand_dims(self.value, dim))

  def expand_dims(self, axis: Union[int, Sequence[int]]) -> 'Array':
    """
    self.expand_dims(axis: int|Sequence[int])

    1. 如果axis类型为int：
    返回一个在self基础上的第axis维度前插入一个维度Array，
    axis<0表示倒数第|axis|维度，
    令n=len(self._value.shape)，则axis的范围为[-(n+1),n]

    2. 如果axis类型为Sequence[int]：
    则返回依次扩展axis[i]的结果，
    即self.expand_dims(axis)==self.expand_dims(axis[0]).expand_dims(axis[1])...expand_dims(axis[len(axis)-1])


    1. If the type of axis is int:

    Returns an Array of dimensions inserted before the axis dimension based on self,

    The first | axis < 0 indicates the bottom axis | dimensions,

    Set n=len(self._value.shape), then axis has the range [-(n+1),n]


    2. If the type of axis is Sequence[int] :

    Returns the result of extending axis[i] in sequence,

    self.expand_dims(axis)==self.expand_dims(axis[0]).expand_dims(axis[1])... expand_dims(axis[len(axis)-1])

    """
    return Array(jnp.expand_dims(self.value, axis))

  def expand(self, *shape: Union[int, Sequence[int]]) -> 'Array':
    """
    Expand an array to a new shape.

    Parameters
    ----------
    shape : tuple or int
        The shape of the desired array. A single integer ``i`` is interpreted
        as ``(i,)``.

    Returns
    -------
    expanded : Array
        A readonly view on the original array with the given shape. It is
        typically not contiguous. Furthermore, more than one element of a
        expanded array may refer to a single memory location.
    """
    return Array(jnp.broadcast_to(self._value, shape))

  def expand_as(self, array: BmArray) -> 'Array':
    """
    Expand an array to a shape of another array.

    Parameters
    ----------
    array : Array

    Returns
    -------
    expanded : Array
        A readonly view on the original array with the given shape of array. It is
        typically not contiguous. Furthermore, more than one element of a
        expanded array may refer to a single memory location.
    """
    if not isinstance(array, Array):
      array = Array(array)
    return Array(jnp.broadcast_to(self.value, array.value.shape))

  def squeeze(self,
              axis: Optional[Union[int, Sequence]] = None) -> 'Array':
    return Array(self.squeeze(axis))

  # def item(self, *args) -> Any:
  #   return self.value.item(*args)

  def pow(self, index: int):
    return self._value ** index

  def addr(self,
           vec1: BmArray,
           vec2: BmArray,
           *,
           beta: float = 1.0,
           alpha: float = 1.0,
           out: OptionalBmArray = None) -> Union[None, NoReturn]:
    if not isinstance(beta, int) and not isinstance(beta, float):
      raise Exception('Wrong beta param of addr')
    if not isinstance(alpha, int) and not isinstance(alpha, float):
      raise Exception('Wrong alpha param of addr')
    if not isinstance(vec1, Array):
      vec1 = Array(vec1)
    if not isinstance(vec2, Array):
      vec2 = Array(vec2)
    if not isinstance(out, Array):
      out = Array(out)
    return _return(jnp.outer(vec1, vec2, out=out))

  def addr_(self,
            vec1: 'Array',
            vec2: 'Array',
            *,
            beta: float = 1.0,
            alpha: float = 1.0) -> Union['Array', NoReturn]:
    if not isinstance(beta, (int, float)):
      raise Exception('Wrong beta param of addr')
    if not isinstance(alpha, (int, float)):
      raise Exception('Wrong alpha param of addr')
    return _return(jnp.outer(vec1, vec2, out=self))

  def outer(self, other: BmArray) -> Union[NoReturn, None]:
    # if other is None:
    #   raise Exception('Array can not make outer product with None')
    if not isinstance(other, Array, jax.Array, np.ndarray):
      raise TypeError("other must be brainpy Array")
    return _return(jnp.outer(self.value, other.value))

  def sum(self) -> 'Array':
    return _return(self.value.sum())

  def abs(self, *, out: OptionalBmArray = None) -> 'Array':
    abs_value = jnp.abs(self.value)
    if isinstance(out, (Array, jax.Array, np.ndarray)):
      out.value = abs_value
    return _return(abs_value)

  def abs_(self) -> 'Array':
    """
    in-place version of Array.abs()
    """
    self.value = jnp.abs(self.value)

  def absolute(self, *, out: OptionalBmArray = None) -> 'Array':
    """
    alias of Array.abs
    """
    if not isinstance(out, Array):
      out = Array(out)
    return self.abs(out=out)

  def absolute_(self) -> 'Array':
    """
    alias of Array.abs_()
    """
    return self.abs_()

  def sin(self, *, out: OptionalBmArray = None) -> 'Array':
    '''
    sin(self,out=None),
    return the sin value of self,
    and save the result to out if out is not None
    '''
    value = jnp.sin(self.value)
    if isinstance(out, (Array, jax.Array, np.ndarray)):
      out.value = value
    return _return(value)

  def sin_(self) -> None:
    '''
    sin_(self),
    in-place version of sin
    no return
    out is the sin value of self,
    and save the result to self
    '''
    self.value = jnp.sin(self.value)

  def sinh(self, *, out: OptionalBmArray = None) -> 'Array':
    '''
    sinh(self,out=None),
    return the sinh value of self,
    and save the result to out if out is not None
    '''
    value = jnp.sinh(self.value)
    if isinstance(out, (Array, jax.Array, np.ndarray)):
      out.value = value
    return _return(value)

  def sinh_(self) -> None:
    '''
    sinh_(self),
    in-place version of sinh
    no return
    out is the sinh value of self,
    and save the result to self
    '''
    self.value = jnp.sinh(self.value)

  def arcsin(self, *, out: OptionalBmArray = None) -> 'Array':
    '''
    arcsin(self,out=None),
    return the arcsin value of self,
    and save the result to out if out is not None
    '''
    value = jnp.arcsin(self.value)
    if isinstance(out, (Array, jax.Array, np.ndarray)):
      out.value = value
    return _return(value)

  def arcsin_(self) -> None:
    '''
    arcsin_(self),
    in-place version of arcsin
    no return
    out is the arcsin value of self,
    and save the result to self
    '''
    self.value = jnp.arcsin(self.value)

  def arcsinh(self, *, out: OptionalBmArray = None) -> 'Array':
    '''
    arcsinh(self,out=None),
    return the arcsinh value of self,
    and save the result to out if out is not None
    '''
    value = jnp.arcsinh(self.value)
    if isinstance(out, (Array, jax.Array, np.ndarray)):
      out.value = value
    return _return(value)

  def arcsinh_(self) -> None:
    '''
    arcsinh_(self),
    in-place version of arcsinh
    no return
    out is the arcsinh value of self,
    and save the result to self
    '''
    self.value = jnp.arcsinh(self.value)

  def cos(self, *, out: OptionalBmArray = None) -> 'Array':
    '''
    cos(self,out=None),
    return the cos value of self,
    and save the result to out if out is not None
    '''
    value = jnp.cos(self.value)
    if isinstance(out, (Array, jax.Array, np.ndarray)):
      out.value = value
    return _return(value)

  def cos_(self) -> None:
    '''
    cos_(self),
    in-place version of cos
    no return
    out is the cos value of self,
    and save the result to self
    '''
    self.value = jnp.cos(self.value)

  def cosh(self, *, out: OptionalBmArray = None) -> 'Array':
    '''
    cosh(self,out=None),
    return the cosh value of self,
    and save the result to out if out is not None
    '''
    value = jnp.cosh(self.value)
    if isinstance(out, (Array, jax.Array, np.ndarray)):
      out.value = value
    return _return(value)

  def cosh_(self) -> None:
    '''
    cosh_(self),
    in-place version of cosh
    no return
    out is the cosh value of self,
    and save the result to self
    '''
    self.value = jnp.cosh(self.value)

  def arccos(self, *, out: OptionalBmArray = None) -> 'Array':
    '''
    arccos(self,out=None),
    return the arccos value of self,
    and save the result to out if out is not None
    '''
    value = jnp.arccos(self.value)
    if isinstance(out, (Array, jax.Array, np.ndarray)):
      out.value = value
    return _return(value)

  def arccos_(self) -> None:
    '''
    arccos_(self),
    in-place version of arccos
    no return
    out is the arccos value of self,
    and save the result to self
    '''
    self.value = jnp.arccos(self.value)

  def arccosh(self, *, out: OptionalBmArray = None) -> 'Array':
    '''
    arccosh(self,out=None),
    return the arccosh value of self,
    and save the result to out if out is not None
    '''
    value = jnp.arccosh(self.value)
    if isinstance(out, (Array, jax.Array, np.ndarray)):
      out.value = value
    return _return(value)

  def arccosh_(self) -> None:
    '''
    arccosh_(self),
    in-place version of arccosh
    no return
    out is the arccosh value of self,
    and save the result to self
    '''
    self.value = jnp.arccosh(self.value)

  def tan(self, *, out: OptionalBmArray = None) -> 'Array':
    '''
    tan(self,out=None),
    return the tan value of self,
    and save the result to out if out is not None
    '''
    value = jnp.tan(self.value)
    if isinstance(out, (Array, jax.Array, np.ndarray)):
      out.value = value
    return _return(value)

  def tan_(self) -> None:
    '''
    tan_(self),
    in-place version of tan
    no return
    out is the tan value of self,
    and save the result to self
    '''
    self.value = jnp.tan(self.value)

  def tanh(self, *, out: OptionalBmArray = None) -> 'Array':
    '''
    tanh(self,out=None),
    return the tanh value of self,
    and save the result to out if out is not None
    '''
    value = jnp.tanh(self.value)
    if isinstance(out, (Array, jax.Array, np.ndarray)):
      out.value = value
    return _return(value)

  def tanh_(self) -> None:
    '''
    tanh_(self),
    in-place version of tanh
    no return
    out is the tanh value of self,
    and save the result to self
    '''
    self.value = jnp.tanh(self.value)

  def arctan(self, *, out: OptionalBmArray = None) -> 'Array':
    '''
    arctan(self,out=None),
    return the arctan value of self,
    and save the result to out if out is not None
    '''
    value = jnp.arctan(self.value)
    if isinstance(out, (Array, jax.Array, np.ndarray)):
      out.value = value
    return _return(value)

  def arctan_(self) -> None:
    '''
    arctan_(self),
    in-place version of arctan
    no return
    out is the arctan value of self,
    and save the result to self
    '''
    self.value = jnp.arctan(self.value)

  def arctanh(self, *, out: OptionalBmArray = None) -> 'Array':
    '''
    arctanh(self,out=None),
    return the arctanh value of self,
    and save the result to out if out is not None
    '''
    value = jnp.arctanh(self.value)
    if isinstance(out, (Array, jax.Array, np.ndarray)):
      out.value = value
    return _return(value)

  def arctanh_(self) -> None:
    '''
    arctanh_(self),
    in-place version of arctanh
    no return
    out is the arctanh value of self,
    and save the result to self
    '''
    self.value = jnp.arctanh(self.value)

  def clamp(self,
            min_value: OptionalBmArray = None,
            max_value: OptionalBmArray = None,
            *,
            out: OptionalBmArray = None) -> 'Array':
    """
    return the value between min_value and max_value,
    if min_value is None, then no lower bound,
    if max_value is None, then no upper bound.
    """

    return _return(self.value.clip(min_value, max_value))

  def clamp_(self,
             min_value: OptionalBmArray = None,
             max_value: OptionalBmArray = None) -> 'Array':
    """
    return the value between min_value and max_value,
    if min_value is None, then no lower bound,
    if max_value is None, then no upper bound.
    """
    return jnp.clip(self.value, min_value, max_value, out=self)

  def clip_(self,
            min_value: OptionalBmArray = None,
            max_value: OptionalBmArray = None) -> 'Array':
    """
    alias for clamp_
    """
    return Array(jnp.clip(self.value, min_value, max_value, out=self))

  def clip(self,
           min_value: OptionalBmArray = None,
           max_value: OptionalBmArray = None,
           *,
           out: OptionalBmArray = None) -> 'Array':
    """
    alias for clamp
    """
    # if out is not None:
    #   if not isinstance(out, (Array, jax.Array, np.ndarray)):
    #     raise Exception('Unexcepted param out')
    value = None
    if out is not None:
      if isinstance(out, Array, jax.Array, np.ndarray):
        value = jnp.clip(self.value, out=out)
    else:
      value = jnp.clip(self.value, min_value, max_value)
    return _return(value)

  def clone(self) -> 'Array':
    return Array(jnp.copy(self.value))

  def copy_(self, src: BmArray) -> None:
    if isinstance(src, Array, jax.Array, np.ndarray):
      self.value = jnp.copyto(self.value, src)

  # def conj(self) -> 'Array':
  #   return Array(jnp.conj(self.value))

  def cov_with(self,
               y: OptionalBmArray = None,
               rowvar: bool = True,
               bias: bool = False,
               ddof: Optional[int] = None,
               fweights: OptionalBmArray = None,
               aweights: OptionalBmArray = None) -> 'Array':
    return Array(jnp.cov(self.value, y, rowvar, bias, fweights, aweights))

  def cov(self,
          *,
          correction: int = 1,
          fweights: OptionalBmArray = None,
          aweights: OptionalBmArray = None) -> ExceptionOrArray:
    try:
      x = [e[0] for e in self.value]
      y = [e[1] for e in self.value]
      return Array(jnp.cov(x, y, ddof=correction, fweights=fweights, aweights=aweights))
    except Exception as e:
      raise Exception('Wrong format, need to be [[x1,y1],[x2,y2],[x3,y3]]')






JaxArray = Array
ndarray = Array


class Variable(Array):
  """The pointer to specify the dynamical variable.

  Initializing an instance of ``Variable`` by two ways:

  >>> import brainpy.math as bm
  >>> # 1. init a Variable by the concreate data
  >>> v1 = bm.Variable(bm.zeros(10))
  >>> # 2. init a Variable by the data shape
  >>> v2 = bm.Variable(10)

  Note that when initializing a `Variable` by the data shape,
  all values in this `Variable` will be initialized as zeros.

  Parameters
  ----------
  value_or_size: Shape, Array, int
    The value or the size of the value.
  dtype:
    The type of the data.
  batch_axis: optional, int
    The batch axis.
  """

  _need_check_context = False
  __slots__ = ('_value', '_batch_axis')

  def __init__(
          self,
          value_or_size,
          dtype: type = None,
          batch_axis: int = None,
  ):
    if isinstance(value_or_size, int):
      value = jnp.zeros(value_or_size, dtype=dtype)
    elif isinstance(value_or_size, (tuple, list)) and all([isinstance(s, int) for s in value_or_size]):
      value = jnp.zeros(value_or_size, dtype=dtype)
    else:
      value = value_or_size

    super(Variable, self).__init__(value, dtype=dtype)

    # check batch axis
    if isinstance(value, Variable):
      if value.batch_axis is not None and batch_axis is not None:
        if batch_axis != value.batch_axis:
          raise ValueError(f'"batch_axis" is not consistent. Got batch_axis in the given value '
                           f'is {value.batch_axis}, but the specified batch_axis is {batch_axis}')
      batch_axis = value.batch_axis

    # assign batch axis
    self._batch_axis = batch_axis
    if batch_axis is not None:
      if batch_axis >= self.ndim:
        raise MathError(f'This variables has {self.ndim} dimension, '
                        f'but the batch axis is set to be {batch_axis}.')

  @property
  def nobatch_shape(self) -> TupleType[int, ...]:
    """Shape without batch axis."""
    if self.batch_axis is not None:
      shape = list(self.value.shape)
      shape.pop(self.batch_axis)
      return tuple(shape)
    else:
      return self.shape

  @property
  def batch_axis(self) -> Optional[int]:
    return self._batch_axis

  @batch_axis.setter
  def batch_axis(self, val):
    raise ValueError(f'Cannot set "batch_axis" after creating a {self.__class__.__name__} instance.')

  @property
  def batch_size(self) -> Optional[int]:
    if self.batch_axis is None:
      return None
    else:
      return self.shape[self.batch_axis]

  @batch_size.setter
  def batch_size(self, val):
    raise ValueError(f'Cannot set "batch_size" manually.')

  def update(self, value):
    """Update the value of this Array.
    """
    if self._batch_axis is None:
      ext_shape = jnp.shape(value)
      int_shape = self.shape
    else:
      ext_shape = value.shape[:self._batch_axis] + value.shape[self._batch_axis + 1:]
      int_shape = self.shape[:self._batch_axis] + self.shape[self._batch_axis + 1:]
    if ext_shape != int_shape:
      error = f"The shape of the original data is {self.shape}, while we got {value.shape}"
      error += f' with batch_axis={self._batch_axis}.'
      raise MathError(error)
    if hasattr(value, 'dtype'):
      dtype = value.dtype
    else:
      dtype = canonicalize_dtype(type(value))
    if dtype != self.dtype:
      raise MathError(f"The dtype of the original data is {self.dtype}, "
                      f"while we got {dtype}.")
    self._value = value.value if isinstance(value, Array) else value


class TrainVar(Variable):
  """The pointer to specify the trainable variable.
  """
  __slots__ = ('_value', '_batch_axis')

  def __init__(self,
               value_or_size,
               dtype: type = None,
               batch_axis: int = None):
    super(TrainVar, self).__init__(value_or_size,
                                   dtype=dtype,
                                   batch_axis=batch_axis)


class Parameter(Variable):
  """The pointer to specify the parameter.
  """
  __slots__ = ('_value', '_batch_axis')

  def __init__(self,
               value_or_size,
               dtype: type = None,
               batch_axis: int = None):
    super(Parameter, self).__init__(value_or_size,
                                    dtype=dtype,
                                    batch_axis=batch_axis)


class ParallelVariable(Variable):
  pass


class BatchVariable(Variable):
  pass


class VariableView(Variable):
  """A view of a Variable instance.

  This class is used to create a subset view of ``brainpy.math.Variable``.

  >>> import brainpy.math as bm
  >>> bm.random.seed(123)
  >>> origin = bm.Variable(bm.random.random(5))
  >>> view = bm.VariableView(origin, slice(None, 2, None))  # origin[:2]
  VariableView([0.02920651, 0.19066381], dtype=float32)

  ``VariableView`` can be used to update the subset of the original
  Variable instance, and make operations on this subset of the Variable.

  >>> view[:] = 1.
  >>> view
  VariableView([1., 1.], dtype=float32)
  >>> origin
  Variable([1.       , 1.       , 0.5482849, 0.6564884, 0.8446237], dtype=float32)
  >>> view + 10
  Array([11., 11.], dtype=float32)
  >>> view *= 10
  VariableView([10., 10.], dtype=float32)

  The above example demonstrates that the updating of an ``VariableView`` instance
  is actually made in the original ``Variable`` instance.

  Moreover, it's worthy to note that ``VariableView`` is not a PyTree.
  """

  def __init__(self, value: Variable, index):
    self.index = jax.tree_util.tree_map(_as_jax_array_, index, is_leaf=lambda a: isinstance(a, Array))
    if not isinstance(value, Variable):
      raise ValueError('Must be instance of Variable.')
    super(VariableView, self).__init__(value.value, batch_axis=value.batch_axis)
    self._value = value

  def __repr__(self) -> str:
    print_code = repr(self._value)
    prefix = f'{self.__class__.__name__}'
    blank = " " * (len(prefix) + 1)
    lines = print_code.split("\n")
    lines[0] = prefix + "(" + lines[0]
    for i in range(1, len(lines)):
      lines[i] = blank + lines[i]
    lines[-1] += ","
    lines.append(blank + f'index={self.index})')
    print_code = "\n".join(lines)
    return print_code

  @property
  def value(self):
    return self._value[self.index]

  @value.setter
  def value(self, v):
    self.update(v)

  def update(self, value):
    int_shape = self.shape
    if self.batch_axis is None:
      ext_shape = value.shape
    else:
      ext_shape = value.shape[:self.batch_axis] + value.shape[self.batch_axis + 1:]
      int_shape = int_shape[:self.batch_axis] + int_shape[self.batch_axis + 1:]
    if ext_shape != int_shape:
      error = f"The shape of the original data is {self.shape}, while we got {value.shape}"
      if self.batch_axis is None:
        error += '. Do you forget to set "batch_axis" when initialize this variable?'
      else:
        error += f' with batch_axis={self.batch_axis}.'
      raise MathError(error)
    if value.dtype != self._value.dtype:
      raise MathError(f"The dtype of the original data is {self._value.dtype}, "
                      f"while we got {value.dtype}.")
    self._value[self.index] = value.value if isinstance(value, Array) else value


def _jaxarray_unflatten(aux_data, flat_contents):
  r = Array(*flat_contents)
  r._transform_context = aux_data[0]
  return r


register_pytree_node(Array,
                     lambda t: ((t.value,), (t._transform_context,)),
                     _jaxarray_unflatten)

register_pytree_node(Variable,
                     lambda t: ((t.value,), None),
                     lambda aux_data, flat_contents: Variable(*flat_contents))

register_pytree_node(TrainVar,
                     lambda t: ((t.value,), None),
                     lambda aux_data, flat_contents: TrainVar(*flat_contents))

register_pytree_node(Parameter,
                     lambda t: ((t.value,), None),
                     lambda aux_data, flat_contents: Parameter(*flat_contents))
