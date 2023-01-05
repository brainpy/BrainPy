# -*- coding: utf-8 -*-

from typing import Optional

import jax.numpy as jnp
import numpy as np
from jax.tree_util import tree_map, tree_flatten, tree_unflatten

from ._utils import wraps
from .arraycreation import *
from .arrayinterporate import *
from .ndarray import Array

__all__ = [
  'full', 'full_like', 'eye', 'identity', 'diag', 'tri', 'tril', 'triu',

  # math funcs
  'real', 'imag', 'conj', 'conjugate', 'ndim', 'isreal', 'isscalar',
  'add', 'reciprocal', 'negative', 'positive', 'multiply', 'divide',
  'power', 'subtract', 'true_divide', 'floor_divide', 'float_power',
  'fmod', 'mod', 'modf', 'divmod', 'remainder', 'abs', 'exp', 'exp2',
  'expm1', 'log', 'log10', 'log1p', 'log2', 'logaddexp', 'logaddexp2',
  'lcm', 'gcd', 'arccos', 'arccosh', 'arcsin', 'arcsinh', 'arctan',
  'arctan2', 'arctanh', 'cos', 'cosh', 'sin', 'sinc', 'sinh', 'tan',
  'tanh', 'deg2rad', 'hypot', 'rad2deg', 'degrees', 'radians', 'round',
  'around', 'round_', 'rint', 'floor', 'ceil', 'trunc', 'fix', 'prod',
  'sum', 'diff', 'median', 'nancumprod', 'nancumsum', 'nanprod', 'nansum',
  'cumprod', 'cumsum', 'ediff1d', 'cross', 'trapz', 'isfinite', 'isinf',
  'isnan', 'signbit', 'copysign', 'nextafter', 'ldexp', 'frexp', 'convolve',
  'sqrt', 'cbrt', 'square', 'absolute', 'fabs', 'sign', 'heaviside',
  'maximum', 'minimum', 'fmax', 'fmin', 'interp', 'clip', 'angle',

  # Elementwise bit operations
  'bitwise_and', 'bitwise_not', 'bitwise_or', 'bitwise_xor',
  'invert', 'left_shift', 'right_shift',

  # logic funcs
  'equal', 'not_equal', 'greater', 'greater_equal', 'less', 'less_equal',
  'array_equal', 'isclose', 'allclose', 'logical_and', 'logical_not',
  'logical_or', 'logical_xor', 'all', 'any', "alltrue", 'sometrue',

  # array manipulation
  'shape', 'size', 'reshape', 'ravel', 'moveaxis', 'transpose', 'swapaxes',
  'concatenate', 'stack', 'vstack', 'hstack', 'dstack', 'column_stack',
  'split', 'dsplit', 'hsplit', 'vsplit', 'tile', 'repeat', 'unique',
  'append', 'flip', 'fliplr', 'flipud', 'roll', 'atleast_1d', 'atleast_2d',
  'atleast_3d', 'expand_dims', 'squeeze', 'sort', 'argsort', 'argmax', 'argmin',
  'argwhere', 'nonzero', 'flatnonzero', 'where', 'searchsorted', 'extract',
  'count_nonzero', 'max', 'min', 'amax', 'amin',

  # array creation
  'array_split', 'meshgrid', 'vander',

  # indexing funcs
  'nonzero', 'where', 'tril_indices', 'tril_indices_from', 'triu_indices',
  'triu_indices_from', 'take', 'select',

  # statistic funcs
  'nanmin', 'nanmax', 'ptp', 'percentile', 'nanpercentile', 'quantile', 'nanquantile',
  'median', 'average', 'mean', 'std', 'var', 'nanmedian', 'nanmean', 'nanstd', 'nanvar',
  'corrcoef', 'correlate', 'cov', 'histogram', 'bincount', 'digitize',

  # window funcs
  'bartlett', 'blackman', 'hamming', 'hanning', 'kaiser',

  # constants
  'e', 'pi', 'inf',

  # linear algebra
  'dot', 'vdot', 'inner', 'outer', 'kron', 'matmul', 'trace',

  # data types
  'dtype', 'finfo', 'iinfo', 'uint8', 'uint16', 'uint32', 'uint64',
  'int8', 'int16', 'int32', 'int64', 'float16', 'float32',
  'float64', 'complex64', 'complex128',

  # more
  'product', 'row_stack', 'apply_over_axes', 'apply_along_axis', 'array_equiv',
  'array_repr', 'array_str', 'block', 'broadcast_arrays', 'broadcast_shapes',
  'broadcast_to', 'compress', 'cumproduct', 'diag_indices', 'diag_indices_from',
  'diagflat', 'diagonal', 'einsum', 'einsum_path', 'geomspace', 'gradient',
  'histogram2d', 'histogram_bin_edges', 'histogramdd', 'i0', 'in1d', 'indices',
  'insert', 'intersect1d', 'iscomplex', 'isin', 'ix_', 'lexsort', 'load',
  'save', 'savez', 'mask_indices', 'msort', 'nan_to_num', 'nanargmax', 'setdiff1d',
  'nanargmin', 'pad', 'poly', 'polyadd', 'polyder', 'polyfit', 'polyint',
  'polymul', 'polysub', 'polyval', 'resize', 'rollaxis', 'roots', 'rot90',
  'setxor1d', 'tensordot', 'trim_zeros', 'union1d', 'unravel_index', 'unwrap',
  'take_along_axis', 'can_cast', 'choose', 'copy', 'frombuffer', 'fromfile',
  'fromfunction', 'fromiter', 'fromstring', 'get_printoptions', 'iscomplexobj',
  'isneginf', 'isposinf', 'isrealobj', 'issubdtype', 'issubsctype', 'iterable',
  'packbits', 'piecewise', 'printoptions', 'set_printoptions', 'promote_types',
  'ravel_multi_index', 'result_type', 'sort_complex', 'unpackbits', 'delete',

  # unique
  'add_docstring', 'add_newdoc', 'add_newdoc_ufunc', 'array2string', 'asanyarray',
  'ascontiguousarray', 'asfarray', 'asscalar', 'common_type', 'disp', 'genfromtxt',
  'loadtxt', 'info', 'issubclass_', 'place', 'polydiv', 'put', 'putmask', 'safe_eval',
  'savetxt', 'savez_compressed', 'show_config', 'typename', 'copyto', 'matrix', 'asmatrix', 'mat',
]

_min = min
_max = max


# others
# ------


def _as_jax_array_(obj):
  return obj.value if isinstance(obj, Array) else obj


@wraps(jnp.full)
def full(shape, fill_value, dtype=None):
  return Array(jnp.full(shape, fill_value, dtype=dtype))


@wraps(jnp.full_like)
def full_like(a, fill_value, dtype=None, shape=None):
  a = _as_jax_array_(a)
  return Array(jnp.full_like(a, fill_value, dtype=dtype, shape=shape))


@wraps(jnp.eye)
def eye(N, M=None, k=0, dtype=None):
  return Array(jnp.eye(N, M=M, k=k, dtype=dtype))


@wraps(jnp.identity)
def identity(n, dtype=None):
  return Array(jnp.identity(n, dtype=dtype))


@wraps(jnp.diag)
def diag(a, k=0):
  a = _as_jax_array_(a)
  return Array(jnp.diag(a, k))


@wraps(jnp.tri)
def tri(N, M=None, k=0, dtype=None):
  return Array(jnp.tri(N, M=M, k=k, dtype=dtype))


@wraps(jnp.tril)
def tril(a, k=0):
  a = _as_jax_array_(a)
  return Array(jnp.tril(a, k))


@wraps(jnp.triu)
def triu(a, k=0):
  a = _as_jax_array_(a)
  return Array(jnp.triu(a, k))


@wraps(jnp.delete)
def delete(arr, obj, axis=None):
  arr = _as_jax_array_(arr)
  obj = _as_jax_array_(obj)
  return Array(jnp.delete(arr, obj, axis=axis))


@wraps(jnp.take_along_axis)
def take_along_axis(a, indices, axis, mode=None):
  a = _as_jax_array_(a)
  indices = _as_jax_array_(indices)
  return Array(jnp.take_along_axis(a, indices, axis, mode))


@wraps(jnp.block)
def block(arrays):
  leaves, tree = tree_flatten(arrays, is_leaf=lambda a: isinstance(a, Array))
  leaves = [_as_jax_array_(l) for l in leaves]
  arrays = tree_unflatten(tree, leaves)
  return Array(jnp.block(arrays))


@wraps(jnp.broadcast_arrays)
def broadcast_arrays(*args):
  args = [(_as_jax_array_(a)) for a in args]
  return jnp.broadcast_arrays(args)


broadcast_shapes = wraps(jnp.broadcast_shapes)(jnp.broadcast_shapes)


@wraps(jnp.broadcast_to)
def broadcast_to(arr, shape):
  arr = _as_jax_array_(arr)
  return Array(jnp.broadcast_to(arr, shape))


@wraps(jnp.compress)
def compress(condition, a, axis=None, out=None):
  condition = _as_jax_array_(condition)
  a = _as_jax_array_(a)
  return Array(jnp.compress(condition, a, axis, out))


@wraps(jnp.diag_indices)
def diag_indices(n, ndim=2):
  res = jnp.diag_indices(n, ndim)
  if isinstance(res, tuple):
    return tuple(Array(r) for r in res)
  else:
    return Array(res)


@wraps(jnp.diag_indices_from)
def diag_indices_from(arr):
  arr = _as_jax_array_(arr)
  res = jnp.diag_indices_from(arr)
  if isinstance(res, tuple):
    return tuple(Array(r) for r in res)
  else:
    return Array(res)


@wraps(jnp.diagflat)
def diagflat(v, k=0):
  v = _as_jax_array_(v)
  return Array(jnp.diagflat(v, k))


@wraps(jnp.diagonal)
def diagonal(a, offset=0, axis1: int = 0, axis2: int = 1):
  a = _as_jax_array_(a)
  return Array(jnp.diagonal(a, offset, axis1, axis2))


@wraps(jnp.einsum)
def einsum(*operands, out=None, optimize='optimal', precision=None, _use_xeinsum=False):
  operands = tuple((_as_jax_array_(a)) for a in operands)
  return Array(jnp.einsum(*operands, out=out, optimize=optimize, precision=precision, _use_xeinsum=_use_xeinsum))


@wraps(jnp.einsum_path)
def einsum_path(subscripts, *operands, optimize='greedy'):
  operands = tuple((_as_jax_array_(a)) for a in operands)
  return jnp.einsum_path(subscripts, *operands, optimize=optimize)


@wraps(jnp.geomspace)
def geomspace(start, stop, num=50, endpoint=True, dtype=None, axis: int = 0):
  return Array(jnp.geomspace(start, stop, num, endpoint, dtype, axis))


@wraps(jnp.gradient)
def gradient(f, *varargs, axis=None, edge_order=None):
  f = _as_jax_array_(f)
  res = jnp.gradient(f, *varargs, axis=axis, edge_order=edge_order)
  if isinstance(res, (list, tuple)):
    return list(Array(r) for r in res)
  else:
    return Array(res)


@wraps(jnp.histogram2d)
def histogram2d(x, y, bins=10, range=None, weights=None, density=None):
  x = _as_jax_array_(x)
  y = _as_jax_array_(y)
  H, xedges, yedges = jnp.histogram2d(x, y, bins, range, weights, density)
  return Array(H), Array(xedges), Array(yedges)


@wraps(jnp.histogram_bin_edges)
def histogram_bin_edges(a, bins=10, range=None, weights=None):
  a = _as_jax_array_(a)
  return Array(jnp.histogram_bin_edges(a, bins, range, weights))


@wraps(jnp.histogramdd)
def histogramdd(sample, bins=10, range=None, weights=None, density=None):
  sample = _as_jax_array_(sample)
  r = jnp.histogramdd(sample, bins, range, weights, density)
  return Array(r[0]), r[1]


@wraps(jnp.i0)
def i0(x):
  x = _as_jax_array_(x)
  return Array(jnp.i0(x))


@wraps(jnp.in1d)
def in1d(ar1, ar2, assume_unique=False, invert=False):
  ar1 = _as_jax_array_(ar1)
  ar2 = _as_jax_array_(ar2)
  return Array(jnp.in1d(ar1, ar2, assume_unique, invert))


@wraps(jnp.indices)
def indices(dimensions, dtype=None, sparse=False):
  dtype = jnp.int32 if dtype is None else dtype
  res = jnp.indices(dimensions, dtype, sparse)
  if isinstance(res, (tuple, list)):
    return tuple(Array(r) for r in res)
  else:
    return Array(res)


@wraps(jnp.insert)
def insert(arr, obj, values, axis=None):
  arr = _as_jax_array_(arr)
  values = _as_jax_array_(values)
  return Array(jnp.insert(arr, obj, values, axis))


@wraps(jnp.intersect1d)
def intersect1d(ar1, ar2, assume_unique=False, return_indices=False):
  ar1 = _as_jax_array_(ar1)
  ar2 = _as_jax_array_(ar2)
  res = jnp.intersect1d(ar1, ar2, assume_unique, return_indices)
  if return_indices:
    return tuple([Array(r) for r in res])
  else:
    return Array(res)


@wraps(jnp.iscomplex)
def iscomplex(x):
  x = _as_jax_array_(x)
  return jnp.iscomplex(x)


@wraps(jnp.isin)
def isin(element, test_elements, assume_unique=False, invert=False):
  element = _as_jax_array_(element)
  test_elements = _as_jax_array_(test_elements)
  return Array(jnp.isin(element, test_elements, assume_unique, invert))


@wraps(jnp.ix_)
def ix_(*args):
  args = [_as_jax_array_(a) for a in args]
  return jnp.ix_(*args)


@wraps(jnp.lexsort)
def lexsort(keys, axis=-1):
  leaves, tree = tree_flatten(keys, is_leaf=lambda x: isinstance(x, Array))
  leaves = [_as_jax_array_(l) for l in leaves]
  keys = tree_unflatten(tree, leaves)
  return Array(jnp.lexsort(keys, axis))



def load(file, mmap_mode=None, allow_pickle=False, fix_imports=True,
         encoding='ASCII'):
  return np.load(file,
                 mmap_mode=mmap_mode,
                 allow_pickle=allow_pickle,
                 fix_imports=fix_imports,
                 encoding=encoding)


@wraps(np.save)
def save(file, arr, allow_pickle=True, fix_imports=True):
  arr = _as_jax_array_(arr)
  np.save(file, arr, allow_pickle, fix_imports)


@wraps(np.savez)
def savez(file, *args, **kwds):
  args = [_as_jax_array_(a) for a in args]
  kwds = {k: _as_jax_array_(v) for k, v in kwds.items()}
  np.savez(file, *args, **kwds)


mask_indices = wraps(jnp.mask_indices)(jnp.mask_indices)


def msort(a):
  return Array(jnp.sort(_as_jax_array_(a), axis=0))


@wraps(jnp.nan_to_num)
def nan_to_num(x, copy=True, nan=0.0, posinf=None, neginf=None):
  x = _as_jax_array_(x)
  return Array(jnp.nan_to_num(x, copy, nan=nan, posinf=posinf, neginf=neginf))


@wraps(jnp.nanargmax)
def nanargmax(a, axis=None, out=None, keepdims=None):
  return Array(jnp.nanargmax(_as_jax_array_(a), axis=axis, out=out, keepdims=keepdims))


@wraps(jnp.nanargmin)
def nanargmin(a, axis=None, out=None, keepdims=None):
  return Array(jnp.nanargmin(_as_jax_array_(a), axis=axis, out=out, keepdims=keepdims))


@wraps(jnp.pad)
def pad(array, pad_width, mode="constant", **kwargs):
  array = _as_jax_array_(array)
  pad_width = _as_jax_array_(pad_width)
  kwargs = {k: _as_jax_array_(v) for k, v in kwargs.items()}
  return Array(jnp.pad(array, pad_width, mode, **kwargs))


@wraps(jnp.poly)
def poly(seq_of_zeros):
  seq_of_zeros = _as_jax_array_(seq_of_zeros)
  return Array(jnp.poly(seq_of_zeros))


@wraps(jnp.polyadd)
def polyadd(a1, a2):
  a1 = _as_jax_array_(a1)
  a2 = _as_jax_array_(a2)
  return Array(jnp.polyadd(a1, a2))


@wraps(jnp.polyder)
def polyder(p, m=1):
  p = _as_jax_array_(p)
  return Array(jnp.polyder(p, m))


@wraps(jnp.polyfit)
def polyfit(x, y, deg, rcond=None, full=False, w=None, cov=False):
  x = _as_jax_array_(x)
  y = _as_jax_array_(y)
  res = jnp.polyfit(x, y, deg, rcond=rcond, full=full, w=w, cov=cov)
  if isinstance(res, (tuple, list)):
    return tuple(Array(r) for r in res)
  else:
    return Array(res)


@wraps(jnp.polyint)
def polyint(p, m=1, k=None):
  p = _as_jax_array_(p)
  return Array(jnp.polyint(p, m, k))


@wraps(jnp.polymul)
def polymul(a1, a2, **kwargs):
  a1 = _as_jax_array_(a1)
  a2 = _as_jax_array_(a2)
  return Array(jnp.polymul(a1, a2, **kwargs))


@wraps(jnp.polysub)
def polysub(a1, a2):
  a1 = _as_jax_array_(a1)
  a2 = _as_jax_array_(a2)
  return Array(jnp.polysub(a1, a2))


@wraps(jnp.polyval)
def polyval(p, x):
  p = _as_jax_array_(p)
  x = _as_jax_array_(x)
  return Array(jnp.polyval(p, x))


@wraps(jnp.resize)
def resize(a, new_shape):
  a = _as_jax_array_(a)
  return Array(jnp.resize(a, new_shape))


@wraps(jnp.rollaxis)
def rollaxis(a, axis: int, start=0):
  a = _as_jax_array_(a)
  return Array(jnp.rollaxis(a, axis, start))


@wraps(jnp.roots)
def roots(p):
  p = _as_jax_array_(p)
  return Array(jnp.roots(p))


@wraps(jnp.rot90)
def rot90(m, k=1, axes=(0, 1)):
  m = _as_jax_array_(m)
  return Array(jnp.rot90(m, k, axes))


@wraps(jnp.setdiff1d)
def setdiff1d(ar1, ar2, assume_unique=False, **kwargs):
  return Array(jnp.setdiff1d(_as_jax_array_(ar1),
                             _as_jax_array_(ar2),
                             assume_unique=assume_unique,
                             **kwargs))


@wraps(jnp.setxor1d)
def setxor1d(ar1, ar2, assume_unique=False):
  return Array(jnp.setxor1d(_as_jax_array_(ar1),
                            _as_jax_array_(ar2),
                            assume_unique=assume_unique))


@wraps(jnp.tensordot)
def tensordot(a, b, axes=2, **kwargs):
  a = _as_jax_array_(a)
  b = _as_jax_array_(b)
  return Array(jnp.tensordot(a, b, axes, **kwargs))


@wraps(jnp.trim_zeros)
def trim_zeros(filt, trim='fb'):
  return Array(jnp.trim_zeros(_as_jax_array_(filt), trim))


@wraps(jnp.union1d)
def union1d(ar1, ar2, **kwargs):
  ar1 = _as_jax_array_(ar1)
  ar2 = _as_jax_array_(ar2)
  return Array(jnp.union1d(ar1, ar2, **kwargs))


@wraps(jnp.unravel_index)
def unravel_index(indices, shape):
  indices = _as_jax_array_(indices)
  shape = _as_jax_array_(shape)
  return jnp.unravel_index(indices, shape)


@wraps(jnp.unwrap)
def unwrap(p, discont=jnp.pi, axis: int = -1, period: float = 2 * jnp.pi):
  p = _as_jax_array_(p)
  return Array(jnp.unwrap(p, discont, axis, period))


# math funcs
# ----------

# 1. Basics
@wraps(jnp.isreal)
def isreal(x):
  x = _as_jax_array_(x)
  return jnp.isreal(x)


@wraps(jnp.isscalar)
def isscalar(x):
  x = _as_jax_array_(x)
  return jnp.isscalar(x)


@wraps(jnp.real)
def real(x):
  return jnp.real(_as_jax_array_(x))


@wraps(jnp.imag)
def imag(x):
  return jnp.imag(_as_jax_array_(x))


@wraps(jnp.conj)
def conj(x):
  return jnp.conj(_as_jax_array_(x))


@wraps(jnp.conjugate)
def conjugate(x):
  return jnp.conjugate(_as_jax_array_(x))


@wraps(jnp.ndim)
def ndim(x):
  return jnp.ndim(_as_jax_array_(x))


# 2. Arithmetic operations
@wraps(jnp.add)
def add(x, y):
  return x + y


@wraps(jnp.reciprocal)
def reciprocal(x):
  x = _as_jax_array_(x)
  return Array(jnp.reciprocal(x))


@wraps(jnp.negative)
def negative(x):
  x = _as_jax_array_(x)
  return Array(jnp.negative(x))


@wraps(jnp.positive)
def positive(x):
  x = _as_jax_array_(x)
  return Array(jnp.positive(x))


@wraps(jnp.multiply)
def multiply(x1, x2):
  x1 = _as_jax_array_(x1)
  x2 = _as_jax_array_(x2)
  return Array(jnp.multiply(x1, x2))


@wraps(jnp.divide)
def divide(x1, x2):
  x1 = _as_jax_array_(x1)
  x2 = _as_jax_array_(x2)
  return Array(jnp.divide(x1, x2))


@wraps(jnp.power)
def power(x1, x2):
  x1 = _as_jax_array_(x1)
  x2 = _as_jax_array_(x2)
  return Array(jnp.power(x1, x2))


@wraps(jnp.subtract)
def subtract(x1, x2):
  x1 = _as_jax_array_(x1)
  x2 = _as_jax_array_(x2)
  return Array(jnp.subtract(x1, x2))


@wraps(jnp.true_divide)
def true_divide(x1, x2):
  x1 = _as_jax_array_(x1)
  x2 = _as_jax_array_(x2)
  return Array(jnp.true_divide(x1, x2))


@wraps(jnp.floor_divide)
def floor_divide(x1, x2):
  x1 = _as_jax_array_(x1)
  x2 = _as_jax_array_(x2)
  return Array(jnp.floor_divide(x1, x2))


@wraps(jnp.float_power)
def float_power(x1, x2):
  x1 = _as_jax_array_(x1)
  x2 = _as_jax_array_(x2)
  return Array(jnp.float_power(x1, x2))


@wraps(jnp.fmod)
def fmod(x1, x2):
  x1 = _as_jax_array_(x1)
  x2 = _as_jax_array_(x2)
  return Array(jnp.fmod(x1, x2))


@wraps(jnp.mod)
def mod(x1, x2):
  if isinstance(x1, Array):  x1 = x1.value
  x2 = _as_jax_array_(x2)
  return Array(jnp.mod(x1, x2))


@wraps(jnp.divmod)
def divmod(x1, x2):
  x1 = _as_jax_array_(x1)
  x2 = _as_jax_array_(x2)
  r = jnp.divmod(x1, x2)
  return Array(r[0]), Array(r[1])


@wraps(jnp.remainder)
def remainder(x1, x2):
  x1 = _as_jax_array_(x1)
  x2 = _as_jax_array_(x2)
  return Array(jnp.remainder(x1, x2))


@wraps(jnp.modf)
def modf(x):
  x = _as_jax_array_(x)
  r = jnp.modf(x)
  return Array(r[0]), Array(r[1])


@wraps(jnp.abs)
def abs(x):
  x = _as_jax_array_(x)
  return Array(jnp.absolute(x))


@wraps(jnp.absolute)
def absolute(x):
  x = _as_jax_array_(x)
  return Array(jnp.absolute(x))


# 3. Exponents and logarithms
@wraps(jnp.exp)
def exp(x):
  x = _as_jax_array_(x)
  return Array(jnp.exp(x))


@wraps(jnp.exp2)
def exp2(x):
  x = _as_jax_array_(x)
  return Array(jnp.exp2(x))


@wraps(jnp.expm1)
def expm1(x):
  x = _as_jax_array_(x)
  return Array(jnp.expm1(x))


@wraps(jnp.log)
def log(x):
  x = _as_jax_array_(x)
  return Array(jnp.log(x))


@wraps(jnp.log10)
def log10(x):
  x = _as_jax_array_(x)
  return Array(jnp.log10(x))


@wraps(jnp.log1p)
def log1p(x):
  x = _as_jax_array_(x)
  return Array(jnp.log1p(x))


@wraps(jnp.log2)
def log2(x):
  x = _as_jax_array_(x)
  return Array(jnp.log2(x))


@wraps(jnp.logaddexp)
def logaddexp(x1, x2):
  x1 = _as_jax_array_(x1)
  x2 = _as_jax_array_(x2)
  return Array(jnp.logaddexp(x1, x2))


@wraps(jnp.logaddexp2)
def logaddexp2(x1, x2):
  x1 = _as_jax_array_(x1)
  x2 = _as_jax_array_(x2)
  return Array(jnp.logaddexp2(x1, x2))


# 4. Rational routines
@wraps(jnp.lcm)
def lcm(x1, x2):
  x1 = _as_jax_array_(x1)
  x2 = _as_jax_array_(x2)
  return Array(jnp.lcm(x1, x2))


@wraps(jnp.gcd)
def gcd(x1, x2):
  x1 = _as_jax_array_(x1)
  x2 = _as_jax_array_(x2)
  return Array(jnp.gcd(x1, x2))


# 5. trigonometric functions
@wraps(jnp.arccos)
def arccos(x):
  x = _as_jax_array_(x)
  return Array(jnp.arccos(x))


@wraps(jnp.arccosh)
def arccosh(x):
  x = _as_jax_array_(x)
  return Array(jnp.arccosh(x))


@wraps(jnp.arcsin)
def arcsin(x):
  x = _as_jax_array_(x)
  return Array(jnp.arcsin(x))


@wraps(jnp.arcsinh)
def arcsinh(x):
  x = _as_jax_array_(x)
  return Array(jnp.arcsinh(x))


@wraps(jnp.arctan)
def arctan(x):
  x = _as_jax_array_(x)
  return Array(jnp.arctan(x))


@wraps(jnp.arctan2)
def arctan2(x, y):
  x = _as_jax_array_(x)
  y = _as_jax_array_(y)
  return Array(jnp.arctan2(x, y))


@wraps(jnp.arctanh)
def arctanh(x):
  x = _as_jax_array_(x)
  return Array(jnp.arctanh(x))


@wraps(jnp.cos)
def cos(x):
  x = _as_jax_array_(x)
  return Array(jnp.cos(x))


@wraps(jnp.cosh)
def cosh(x):
  x = _as_jax_array_(x)
  return Array(jnp.cosh(x))


@wraps(jnp.sin)
def sin(x):
  x = _as_jax_array_(x)
  return Array(jnp.sin(x))


@wraps(jnp.sinc)
def sinc(x):
  x = _as_jax_array_(x)
  return Array(jnp.sinc(x))


@wraps(jnp.sinh)
def sinh(x):
  x = _as_jax_array_(x)
  return Array(jnp.sinh(x))


@wraps(jnp.tan)
def tan(x):
  x = _as_jax_array_(x)
  return Array(jnp.tan(x))


@wraps(jnp.tanh)
def tanh(x):
  x = _as_jax_array_(x)
  return Array(jnp.tanh(x))


@wraps(jnp.deg2rad)
def deg2rad(x):
  x = _as_jax_array_(x)
  return Array(jnp.deg2rad(x))


@wraps(jnp.rad2deg)
def rad2deg(x):
  x = _as_jax_array_(x)
  return Array(jnp.rad2deg(x))


@wraps(jnp.degrees)
def degrees(x):
  x = _as_jax_array_(x)
  return Array(jnp.degrees(x))


@wraps(jnp.radians)
def radians(x):
  x = _as_jax_array_(x)
  return Array(jnp.radians(x))


@wraps(jnp.hypot)
def hypot(x1, x2):
  x1 = _as_jax_array_(x1)
  x2 = _as_jax_array_(x2)
  return Array(jnp.hypot(x1, x2))


# 6. Rounding
@wraps(jnp.round)
def round(a, decimals=0):
  a = _as_jax_array_(a)
  return Array(jnp.round(a, decimals=decimals))


around = round
round_ = round


@wraps(jnp.rint)
def rint(x):
  x = _as_jax_array_(x)
  return Array(jnp.rint(x))


@wraps(jnp.floor)
def floor(x):
  x = _as_jax_array_(x)
  return Array(jnp.floor(x))


@wraps(jnp.ceil)
def ceil(x):
  x = _as_jax_array_(x)
  return Array(jnp.ceil(x))


@wraps(jnp.trunc)
def trunc(x):
  x = _as_jax_array_(x)
  return Array(jnp.trunc(x))


@wraps(jnp.fix)
def fix(x):
  x = _as_jax_array_(x)
  return Array(jnp.fix(x))


# 7. Sums, products, differences, Reductions


@wraps(jnp.prod)
def prod(a, axis=None, dtype=None, keepdims=None, initial=None, where=None, **kwargs):
  a = _as_jax_array_(a)
  r = jnp.prod(a, axis=axis, dtype=dtype, keepdims=keepdims, initial=initial, where=where, **kwargs)
  return r if axis is None else Array(r)


product = prod


@wraps(jnp.sum)
def sum(a, axis=None, dtype=None, keepdims=None, initial=None, where=None, **kwargs):
  a = _as_jax_array_(a)
  r = jnp.sum(a, axis=axis, dtype=dtype, keepdims=keepdims, initial=initial, where=where, **kwargs)
  return r if axis is None else Array(r)


@wraps(jnp.diff)
def diff(a, n=1, axis: int = -1, prepend=None, append=None):
  a = _as_jax_array_(a)
  return Array(jnp.diff(a, n=n, axis=axis, prepend=prepend, append=append))


@wraps(jnp.median)
def median(a, axis=None, keepdims=False, **kwargs):
  a = _as_jax_array_(a)
  r = jnp.median(a, axis=axis, keepdims=keepdims, **kwargs)
  return r if axis is None else Array(r)


@wraps(jnp.nancumprod)
def nancumprod(a, axis=None, dtype=None):
  a = _as_jax_array_(a)
  return Array(jnp.nancumprod(a=a, axis=axis, dtype=dtype))


@wraps(jnp.nancumsum)
def nancumsum(a, axis=None, dtype=None):
  a = _as_jax_array_(a)
  return Array(jnp.nancumsum(a=a, axis=axis, dtype=dtype))


@wraps(jnp.cumprod)
def cumprod(a, axis=None, dtype=None):
  a = _as_jax_array_(a)
  return Array(jnp.cumprod(a=a, axis=axis, dtype=dtype))


cumproduct = cumprod


@wraps(jnp.cumsum)
def cumsum(a, axis=None, dtype=None):
  a = _as_jax_array_(a)
  return Array(jnp.cumsum(a=a, axis=axis, dtype=dtype))


@wraps(jnp.nanprod)
def nanprod(a, axis=None, dtype=None, keepdims=None, **kwargs):
  a = _as_jax_array_(a)
  r = jnp.nanprod(a=a, axis=axis, dtype=dtype, keepdims=keepdims, **kwargs)
  return r if axis is None else Array(r)


@wraps(jnp.nansum)
def nansum(a, axis=None, dtype=None, keepdims=None, **kwargs):
  a = _as_jax_array_(a)
  r = jnp.nansum(a=a, axis=axis, dtype=dtype, keepdims=keepdims, **kwargs)
  return r if axis is None else Array(r)


@wraps(jnp.ediff1d)
def ediff1d(a, to_end=None, to_begin=None):
  a = _as_jax_array_(a)
  to_end = _as_jax_array_(to_end)
  to_begin = _as_jax_array_(to_begin)
  return Array(jnp.ediff1d(a, to_end=to_end, to_begin=to_begin))


@wraps(jnp.cross)
def cross(a, b, axisa=-1, axisb=-1, axisc=-1, axis=None):
  a = _as_jax_array_(a)
  b = _as_jax_array_(b)
  return Array(jnp.cross(a, b, axisa=axisa, axisb=axisb, axisc=axisc, axis=axis))


@wraps(jnp.trapz)
def trapz(y, x=None, dx=1.0, axis: int = -1):
  y = _as_jax_array_(y)
  x = _as_jax_array_(x)
  return jnp.trapz(y, x=x, dx=dx, axis=axis)


# 8. floating_functions
@wraps(jnp.isfinite)
def isfinite(x):
  x = _as_jax_array_(x)
  return Array(jnp.isfinite(x))


@wraps(jnp.isinf)
def isinf(x):
  x = _as_jax_array_(x)
  return Array(jnp.isinf(x))


@wraps(jnp.isnan)
def isnan(x):
  x = _as_jax_array_(x)
  return Array(jnp.isnan(x))


@wraps(jnp.signbit)
def signbit(x):
  x = _as_jax_array_(x)
  return Array(jnp.signbit(x))


@wraps(jnp.nextafter)
def nextafter(x1, x2):
  x1 = _as_jax_array_(x1)
  x2 = _as_jax_array_(x2)
  return Array(jnp.nextafter(x1, x2))


@wraps(jnp.copysign)
def copysign(x1, x2):
  x1 = _as_jax_array_(x1)
  x2 = _as_jax_array_(x2)
  return Array(jnp.copysign(x1, x2))


@wraps(jnp.ldexp)
def ldexp(x1, x2):
  x1 = _as_jax_array_(x1)
  x2 = _as_jax_array_(x2)
  return Array(jnp.ldexp(x1, x2))


@wraps(jnp.frexp)
def frexp(x):
  x = _as_jax_array_(x)
  mantissa, exponent = jnp.frexp(x)
  return Array(mantissa), Array(exponent)


# 9. Miscellaneous
@wraps(jnp.convolve)
def convolve(a, v, mode='full', **kwargs):
  a = _as_jax_array_(a)
  v = _as_jax_array_(v)
  return Array(jnp.convolve(a, v, mode, **kwargs))


@wraps(jnp.sqrt)
def sqrt(x):
  x = _as_jax_array_(x)
  return Array(jnp.sqrt(x))


@wraps(jnp.cbrt)
def cbrt(x):
  x = _as_jax_array_(x)
  return Array(jnp.cbrt(x))


@wraps(jnp.square)
def square(x):
  x = _as_jax_array_(x)
  return Array(jnp.square(x))


@wraps(jnp.fabs)
def fabs(x):
  x = _as_jax_array_(x)
  return Array(jnp.fabs(x))


@wraps(jnp.sign)
def sign(x):
  x = _as_jax_array_(x)
  return Array(jnp.sign(x))


@wraps(jnp.heaviside)
def heaviside(x1, x2):
  x1 = _as_jax_array_(x1)
  x2 = _as_jax_array_(x2)
  return Array(jnp.heaviside(x1, x2))


@wraps(jnp.maximum)
def maximum(x1, x2):
  x1 = _as_jax_array_(x1)
  x2 = _as_jax_array_(x2)
  return Array(jnp.maximum(x1, x2))


@wraps(jnp.minimum)
def minimum(x1, x2):
  x1 = _as_jax_array_(x1)
  x2 = _as_jax_array_(x2)
  return Array(jnp.minimum(x1, x2))


@wraps(jnp.fmax)
def fmax(x1, x2):
  x1 = _as_jax_array_(x1)
  x2 = _as_jax_array_(x2)
  return Array(jnp.fmax(x1, x2))


@wraps(jnp.fmin)
def fmin(x1, x2):
  x1 = _as_jax_array_(x1)
  x2 = _as_jax_array_(x2)
  return Array(jnp.fmin(x1, x2))


@wraps(jnp.interp)
def interp(x, xp, fp, left=None, right=None, period=None):
  x = _as_jax_array_(x)
  xp = _as_jax_array_(xp)
  fp = _as_jax_array_(fp)
  return Array(jnp.interp(x, xp, fp, left=left, right=right, period=period))


@wraps(jnp.clip)
def clip(a, a_min=None, a_max=None):
  a = _as_jax_array_(a)
  a_min = _as_jax_array_(a_min)
  a_max = _as_jax_array_(a_max)
  return Array(jnp.clip(a, a_min, a_max))


@wraps(jnp.angle)
def angle(z, deg=False):
  z = _as_jax_array_(z)
  a = jnp.angle(z)
  if deg:
    a *= 180 / pi
  return Array(a)


# binary funcs
# -------------


@wraps(jnp.bitwise_not)
def bitwise_not(x):
  x = _as_jax_array_(x)
  return Array(jnp.bitwise_not(x))


@wraps(jnp.invert)
def invert(x):
  x = _as_jax_array_(x)
  return Array(jnp.invert(x))


@wraps(jnp.bitwise_and)
def bitwise_and(x1, x2):
  x1 = _as_jax_array_(x1)
  x2 = _as_jax_array_(x2)
  return Array(jnp.bitwise_and(x1, x2))


@wraps(jnp.bitwise_or)
def bitwise_or(x1, x2):
  x1 = _as_jax_array_(x1)
  x2 = _as_jax_array_(x2)
  return Array(jnp.bitwise_or(x1, x2))


@wraps(jnp.bitwise_xor)
def bitwise_xor(x1, x2):
  x1 = _as_jax_array_(x1)
  x2 = _as_jax_array_(x2)
  return Array(jnp.bitwise_xor(x1, x2))


@wraps(jnp.left_shift)
def left_shift(x1, x2):
  x1 = _as_jax_array_(x1)
  x2 = _as_jax_array_(x2)
  return Array(jnp.left_shift(x1, x2))


@wraps(jnp.right_shift)
def right_shift(x1, x2):
  x1 = _as_jax_array_(x1)
  x2 = _as_jax_array_(x2)
  return Array(jnp.right_shift(x1, x2))


# logic funcs
# -----------

# 1. Comparison
@wraps(jnp.equal)
def equal(x1, x2):
  x1 = _as_jax_array_(x1)
  x2 = _as_jax_array_(x2)
  return Array(jnp.equal(x1, x2))


@wraps(jnp.not_equal)
def not_equal(x1, x2):
  x1 = _as_jax_array_(x1)
  x2 = _as_jax_array_(x2)
  return Array(jnp.not_equal(x1, x2))


@wraps(jnp.greater)
def greater(x1, x2):
  x1 = _as_jax_array_(x1)
  x2 = _as_jax_array_(x2)
  return Array(jnp.greater(x1, x2))


@wraps(jnp.greater_equal)
def greater_equal(x1, x2):
  x1 = _as_jax_array_(x1)
  x2 = _as_jax_array_(x2)
  return Array(jnp.greater_equal(x1, x2))


@wraps(jnp.less)
def less(x1, x2):
  x1 = _as_jax_array_(x1)
  x2 = _as_jax_array_(x2)
  return Array(jnp.less(x1, x2))


@wraps(jnp.less_equal)
def less_equal(x1, x2):
  x1 = _as_jax_array_(x1)
  x2 = _as_jax_array_(x2)
  return Array(jnp.less_equal(x1, x2))


@wraps(jnp.array_equal)
def array_equal(a, b, equal_nan=False):
  a = _as_jax_array_(a)
  b = _as_jax_array_(b)
  return jnp.array_equal(a, b, equal_nan=equal_nan)


@wraps(jnp.isclose)
def isclose(a, b, rtol=1e-05, atol=1e-08, equal_nan=False):
  a = _as_jax_array_(a)
  b = _as_jax_array_(b)
  return Array(jnp.isclose(a, b, rtol=rtol, atol=atol, equal_nan=equal_nan))


@wraps(jnp.allclose)
def allclose(a, b, rtol=1e-05, atol=1e-08, equal_nan=False):
  a = _as_jax_array_(a)
  b = _as_jax_array_(b)
  return jnp.allclose(a, b, rtol=rtol, atol=atol, equal_nan=equal_nan)


# 2. Logical operations
@wraps(jnp.logical_not)
def logical_not(x):
  x = _as_jax_array_(x)
  return Array(jnp.logical_not(x))


@wraps(jnp.logical_and)
def logical_and(x1, x2):
  x1 = _as_jax_array_(x1)
  x2 = _as_jax_array_(x2)
  return Array(jnp.logical_and(x1, x2))


@wraps(jnp.logical_or)
def logical_or(x1, x2):
  x1 = _as_jax_array_(x1)
  x2 = _as_jax_array_(x2)
  return Array(jnp.logical_or(x1, x2))


@wraps(jnp.logical_xor)
def logical_xor(x1, x2):
  x1 = _as_jax_array_(x1)
  x2 = _as_jax_array_(x2)
  return Array(jnp.logical_xor(x1, x2))


# 3. Truth value testing
@wraps(jnp.all)
def all(a, axis=None, keepdims=None, where=None):
  a = _as_jax_array_(a)
  r = jnp.all(a=a, axis=axis, keepdims=keepdims, where=where)
  return r if axis is None else Array(r)


@wraps(jnp.any)
def any(a, axis=None, keepdims=None, where=None):
  a = _as_jax_array_(a)
  r = jnp.any(a=a, axis=axis, keepdims=keepdims, where=where)
  return r if axis is None else Array(r)


alltrue = all
sometrue = any


# array manipulation
# ------------------


@wraps(jnp.shape)
def shape(x):
  x = _as_jax_array_(x)
  return jnp.shape(x)


@wraps(jnp.size)
def size(x, axis=None):
  x = _as_jax_array_(x)
  r = jnp.size(x, axis=axis)
  return r if axis is None else Array(r)


@wraps(jnp.reshape)
def reshape(x, newshape, order="C"):
  x = _as_jax_array_(x)
  return Array(jnp.reshape(x, newshape, order=order))


@wraps(jnp.ravel)
def ravel(x, order="C"):
  x = _as_jax_array_(x)
  return Array(jnp.ravel(x, order=order))


@wraps(jnp.moveaxis)
def moveaxis(x, source, destination):
  x = _as_jax_array_(x)
  return Array(jnp.moveaxis(x, source, destination))


@wraps(jnp.transpose)
def transpose(x, axis=None):
  x = _as_jax_array_(x)
  return Array(jnp.transpose(x, axes=axis))


@wraps(jnp.swapaxes)
def swapaxes(x, axis1, axis2):
  x = _as_jax_array_(x)
  return Array(jnp.swapaxes(x, axis1, axis2))


@wraps(jnp.concatenate)
def concatenate(arrays, axis: int = 0):
  arrays = [_as_jax_array_(a) for a in arrays]
  return Array(jnp.concatenate(arrays, axis))


@wraps(jnp.stack)
def stack(arrays, axis: int = 0):
  arrays = [_as_jax_array_(a) for a in arrays]
  return Array(jnp.stack(arrays, axis))


@wraps(jnp.vstack)
def vstack(arrays):
  arrays = [_as_jax_array_(a) for a in arrays]
  return Array(jnp.vstack(arrays))


row_stack = vstack


@wraps(jnp.hstack)
def hstack(arrays):
  arrays = [_as_jax_array_(a) for a in arrays]
  return Array(jnp.hstack(arrays))


@wraps(jnp.dstack)
def dstack(arrays):
  arrays = [_as_jax_array_(a) for a in arrays]
  return Array(jnp.dstack(arrays))


@wraps(jnp.column_stack)
def column_stack(arrays):
  arrays = [_as_jax_array_(a) for a in arrays]
  return Array(jnp.column_stack(arrays))


@wraps(jnp.split)
def split(ary, indices_or_sections, axis=0):
  if isinstance(ary, Array): ary = ary.value
  if isinstance(indices_or_sections, Array): indices_or_sections = indices_or_sections.value
  return [Array(a) for a in jnp.split(ary, indices_or_sections, axis=axis)]


@wraps(jnp.dsplit)
def dsplit(ary, indices_or_sections):
  return split(ary, indices_or_sections, axis=2)


@wraps(jnp.hsplit)
def hsplit(ary, indices_or_sections):
  return split(ary, indices_or_sections, axis=1)


@wraps(jnp.vsplit)
def vsplit(ary, indices_or_sections):
  return split(ary, indices_or_sections, axis=0)


@wraps(jnp.tile)
def tile(A, reps):
  A = _as_jax_array_(A)
  return Array(jnp.tile(A, reps))


@wraps(jnp.repeat)
def repeat(x, repeats, axis=None, **kwargs):
  x = _as_jax_array_(x)
  return Array(jnp.repeat(x, repeats=repeats, axis=axis, **kwargs))


@wraps(jnp.unique)
def unique(x, return_index=False, return_inverse=False,
           return_counts=False, axis=None, **kwargs):
  x = _as_jax_array_(x)
  res = jnp.unique(x,
                   return_index=return_index,
                   return_inverse=return_inverse,
                   return_counts=return_counts,
                   axis=axis,
                   **kwargs)
  if isinstance(res, tuple):
    return tuple(Array(r) for r in res)
  else:
    return Array(res)


@wraps(jnp.append)
def append(arr, values, axis=None):
  arr = _as_jax_array_(arr)
  values = _as_jax_array_(values)
  return Array(jnp.append(arr, values, axis=axis))


@wraps(jnp.flip)
def flip(x, axis=None):
  x = _as_jax_array_(x)
  return Array(jnp.flip(x, axis=axis))


@wraps(jnp.fliplr)
def fliplr(x):
  x = _as_jax_array_(x)
  return Array(jnp.fliplr(x))


@wraps(jnp.flipud)
def flipud(x):
  x = _as_jax_array_(x)
  return Array(jnp.flipud(x))


@wraps(jnp.roll)
def roll(x, shift, axis=None):
  x = _as_jax_array_(x)
  return Array(jnp.roll(x, shift, axis=axis))


@wraps(jnp.atleast_1d)
def atleast_1d(*arys):
  return jnp.atleast_1d(*[_as_jax_array_(a) for a in arys])


@wraps(jnp.atleast_2d)
def atleast_2d(*arys):
  return jnp.atleast_2d(*[_as_jax_array_(a) for a in arys])


@wraps(jnp.atleast_3d)
def atleast_3d(*arys):
  return jnp.atleast_3d(*[_as_jax_array_(a) for a in arys])


@wraps(jnp.expand_dims)
def expand_dims(x, axis):
  x = _as_jax_array_(x)
  return Array(jnp.expand_dims(x, axis=axis))


@wraps(jnp.squeeze)
def squeeze(x, axis=None):
  x = _as_jax_array_(x)
  return Array(jnp.squeeze(x, axis=axis))


@wraps(jnp.sort)
def sort(x, axis=-1, kind='quicksort', order=None):
  x = _as_jax_array_(x)
  return Array(jnp.sort(x, axis=axis, kind=kind, order=order))


@wraps(jnp.argsort)
def argsort(x, axis=-1, kind='stable', order=None):
  x = _as_jax_array_(x)
  return Array(jnp.argsort(x, axis=axis, kind=kind, order=order))


@wraps(jnp.argmax)
def argmax(x, axis=None, **kwargs):
  x = _as_jax_array_(x)
  r = jnp.argmax(x, axis=axis, **kwargs)
  return r if axis is None else Array(r)


@wraps(jnp.argmin)
def argmin(x, axis=None, **kwargs):
  x = _as_jax_array_(x)
  r = jnp.argmin(x, axis=axis, **kwargs)
  return r if axis is None else Array(r)


@wraps(jnp.argwhere)
def argwhere(x, **kwargs):
  x = _as_jax_array_(x)
  return Array(jnp.argwhere(x, **kwargs))


@wraps(jnp.nonzero)
def nonzero(x, **kwargs):
  x = _as_jax_array_(x)
  res = jnp.nonzero(x, **kwargs)
  return tuple([Array(r) for r in res]) if isinstance(res, tuple) else Array(res)


@wraps(jnp.flatnonzero)
def flatnonzero(x, **kwargs):
  x = _as_jax_array_(x)
  return Array(jnp.flatnonzero(x, **kwargs))


@wraps(jnp.where)
def where(condition, x=None, y=None, **kwargs):
  condition = _as_jax_array_(condition)
  x = _as_jax_array_(x)
  y = _as_jax_array_(y)
  res = jnp.where(condition, x=x, y=y, **kwargs)
  if isinstance(res, tuple):
    return tuple(Array(r) for r in res)
  else:
    return Array(res)


@wraps(jnp.searchsorted)
def searchsorted(a, v, side='left', sorter=None):
  a = _as_jax_array_(a)
  v = _as_jax_array_(v)
  return Array(jnp.searchsorted(a, v, side=side, sorter=sorter))


@wraps(jnp.extract)
def extract(condition, arr):
  condition = _as_jax_array_(condition)
  arr = _as_jax_array_(arr)
  return Array(jnp.extract(condition, arr))


@wraps(jnp.count_nonzero)
def count_nonzero(a, axis=None, keepdims=False):
  a = _as_jax_array_(a)
  return jnp.count_nonzero(a, axis=axis, keepdims=keepdims)


@wraps(jnp.max)
def max(a, axis=None, out=None, keepdims=None, initial=None, where=None):
  a = _as_jax_array_(a)
  r = jnp.max(a, axis=axis, out=out, keepdims=keepdims, initial=initial, where=where)
  return r if axis is None else Array(r)


@wraps(jnp.min)
def min(a, axis=None, out=None, keepdims=None, initial=None, where=None):
  a = _as_jax_array_(a)
  r = jnp.min(a, axis=axis, out=out, keepdims=keepdims, initial=initial, where=where)
  return r if axis is None else Array(r)


amax = max
amin = min


@wraps(jnp.apply_along_axis)
def apply_along_axis(func1d, axis: int, arr, *args, **kwargs):
  arr = _as_jax_array_(arr)
  return jnp.apply_along_axis(func1d, axis, arr, *args, **kwargs)


@wraps(jnp.apply_over_axes)
def apply_over_axes(func, a, axes):
  a = _as_jax_array_(a)
  return jnp.apply_over_axes(func, a, axes)


@wraps(jnp.array_equiv)
def array_equiv(a1, a2):
  try:
    a1, a2 = asarray(a1), asarray(a2)
  except Exception:
    return False
  try:
    eq = equal(a1, a2)
  except ValueError:
    # shapes are not broadcastable
    return False
  return all(eq)


@wraps(jnp.array_repr)
def array_repr(arr, max_line_width=None, precision=None, suppress_small=None):
  arr = _as_jax_array_(arr)
  return jnp.array_repr(arr, max_line_width=max_line_width, precision=precision, suppress_small=suppress_small)


@wraps(jnp.array_str)
def array_str(a, max_line_width=None, precision=None, suppress_small=None):
  a = _as_jax_array_(a)
  return jnp.array_str(a, max_line_width=max_line_width, precision=precision, suppress_small=suppress_small)


@wraps(jnp.array_split)
def array_split(ary, indices_or_sections, axis: int = 0):
  ary = _as_jax_array_(ary)
  if isinstance(indices_or_sections, Array):
    indices_or_sections = indices_or_sections.value
  elif isinstance(indices_or_sections, (tuple, list)):
    indices_or_sections = [_as_jax_array_(i) for i in indices_or_sections]
  return tuple([Array(a) for a in jnp.array_split(ary, indices_or_sections, axis)])


@wraps(jnp.meshgrid)
def meshgrid(*xi, copy=True, sparse=False, indexing='xy'):
  xi = [_as_jax_array_(x) for x in xi]
  rr = jnp.meshgrid(*xi, copy=copy, sparse=sparse, indexing=indexing)
  return list(Array(r) for r in rr)


@wraps(jnp.vander)
def vander(x, N=None, increasing=False):
  x = _as_jax_array_(x)
  return Array(jnp.vander(x, N=N, increasing=increasing))



# indexing funcs
# --------------

tril_indices = jnp.tril_indices
triu_indices = jnp.triu_indices


@wraps(jnp.tril_indices_from)
def tril_indices_from(x, k=0):
  x = _as_jax_array_(x)
  res = jnp.tril_indices_from(x, k=k)
  if isinstance(res, tuple):
    return tuple(Array(r) for r in res)
  else:
    return Array(res)


@wraps(jnp.triu_indices_from)
def triu_indices_from(x, k=0):
  x = _as_jax_array_(x)
  res = jnp.triu_indices_from(x, k=k)
  if isinstance(res, tuple):
    return tuple(Array(r) for r in res)
  else:
    return Array(res)


@wraps(jnp.take)
def take(x, indices, axis=None, mode=None):
  x = _as_jax_array_(x)
  indices = _as_jax_array_(indices)
  return Array(jnp.take(x, indices=indices, axis=axis, mode=mode))


@wraps(jnp.select)
def select(condlist, choicelist, default=0):
  condlist = [_as_jax_array_(c) for c in condlist]
  choicelist = [_as_jax_array_(c) for c in choicelist]
  return Array(jnp.select(condlist, choicelist, default=default))


# statistic funcs
# ---------------
@wraps(jnp.nanmin)
def nanmin(x, axis=None, keepdims=None, **kwargs):
  x = _as_jax_array_(x)
  r = jnp.nanmin(x, axis=axis, keepdims=keepdims, **kwargs)
  return r if axis is None else Array(r)


@wraps(jnp.nanmax)
def nanmax(x, axis=None, keepdims=None, **kwargs):
  x = _as_jax_array_(x)
  r = jnp.nanmax(x, axis=axis, keepdims=keepdims, **kwargs)
  return r if axis is None else Array(r)


@wraps(jnp.ptp)
def ptp(x, axis=None, keepdims=None):
  x = _as_jax_array_(x)
  r = jnp.ptp(x, axis=axis, keepdims=keepdims)
  return r if axis is None else Array(r)


@wraps(jnp.percentile)
def percentile(a,
               q,
               axis=None,
               out=None,
               overwrite_input: bool = False,
               method: str = "linear",
               keepdims: bool = False,
               interpolation=None):
  a = _as_jax_array_(a)
  q = _as_jax_array_(q)
  r = jnp.percentile(a=a,
                     q=q,
                     axis=axis,
                     out=out,
                     overwrite_input=overwrite_input,
                     method=method,
                     keepdims=keepdims,
                     interpolation=interpolation)
  return r if axis is None else Array(r)


@wraps(jnp.nanpercentile)
def nanpercentile(a,
                  q,
                  axis=None,
                  out=None,
                  overwrite_input: bool = False,
                  method: str = "linear",
                  keepdims: bool = False,
                  interpolation=None):
  a = _as_jax_array_(a)
  q = _as_jax_array_(q)
  r = jnp.nanpercentile(a=a,
                        q=q,
                        axis=axis,
                        out=out,
                        overwrite_input=overwrite_input,
                        method=method,
                        keepdims=keepdims,
                        interpolation=interpolation)
  return r if axis is None else Array(r)


@wraps(jnp.quantile)
def quantile(a, q, axis=None, out=None, overwrite_input: bool = False, method: str = "linear",
             keepdims: bool = False,
             interpolation=None):
  a = _as_jax_array_(a)
  q = _as_jax_array_(q)
  r = jnp.quantile(a=a, q=q, axis=axis, out=out, overwrite_input=overwrite_input, method=method, keepdims=keepdims,
                   interpolation=interpolation)
  return r if axis is None else Array(r)


@wraps(jnp.nanquantile)
def nanquantile(a, q, axis=None, out=None, overwrite_input: bool = False, method: str = "linear",
                keepdims: bool = False,
                interpolation=None):
  a = _as_jax_array_(a)
  q = _as_jax_array_(q)
  r = jnp.nanquantile(a=a, q=q, axis=axis, out=out, overwrite_input=overwrite_input, method=method, keepdims=keepdims,
                      interpolation=interpolation)
  return r if axis is None else Array(r)


@wraps(jnp.average)
def average(a, axis=None, weights=None, returned=False):
  a = _as_jax_array_(a)
  weights = _as_jax_array_(weights)
  r = jnp.average(a, axis=axis, weights=weights, returned=returned)
  if axis is None:
    return r
  elif isinstance(r, tuple):
    return tuple(Array(_r) for _r in r)
  else:
    return Array(r)


@wraps(jnp.mean)
def mean(a, axis=None, dtype=None, keepdims=None, where=None):
  a = _as_jax_array_(a)
  r = jnp.mean(a, axis=axis, dtype=dtype, keepdims=keepdims, where=where)
  return r if axis is None else Array(r)


@wraps(jnp.std)
def std(a, axis=None, dtype=None, ddof=0, keepdims=None, where=None):
  a = _as_jax_array_(a)
  r = jnp.std(a=a, axis=axis, dtype=dtype, ddof=ddof, keepdims=keepdims, where=where)
  return r if axis is None else Array(r)


@wraps(jnp.var)
def var(a, axis=None, dtype=None, ddof=0, keepdims=None, where=None):
  a = _as_jax_array_(a)
  r = jnp.var(a, axis=axis, dtype=dtype, ddof=ddof, keepdims=keepdims, where=where)
  return r if axis is None else Array(r)


@wraps(jnp.nanmedian)
def nanmedian(a, axis=None, keepdims=False):
  return nanquantile(a, 0.5, axis=axis, keepdims=keepdims, interpolation='midpoint')


@wraps(jnp.nanmean)
def nanmean(a, axis=None, dtype=None, keepdims=None, **kwargs):
  a = _as_jax_array_(a)
  r = jnp.nanmean(a, axis=axis, dtype=dtype, keepdims=keepdims, **kwargs)
  return r if axis is None else Array(r)


@wraps(jnp.nanstd)
def nanstd(a, axis=None, dtype=None, ddof=0, keepdims=None, where=None):
  a = _as_jax_array_(a)
  r = jnp.nanstd(a=a, axis=axis, dtype=dtype, ddof=ddof, keepdims=keepdims, where=where)
  return r if axis is None else Array(r)


@wraps(jnp.nanvar)
def nanvar(a, axis=None, dtype=None, ddof=0, keepdims=None, where=None):
  a = _as_jax_array_(a)
  r = jnp.nanvar(a, axis=axis, dtype=dtype, ddof=ddof, keepdims=keepdims, where=where)
  return r if axis is None else Array(r)


@wraps(jnp.corrcoef)
def corrcoef(x, y=None, rowvar=True):
  x = _as_jax_array_(x)
  y = _as_jax_array_(y)
  return Array(jnp.corrcoef(x, y, rowvar))


@wraps(jnp.correlate)
def correlate(a, v, mode='valid', **kwargs):
  a = _as_jax_array_(a)
  v = _as_jax_array_(v)
  return Array(jnp.correlate(a, v, mode, **kwargs))


@wraps(jnp.cov)
def cov(m, y=None, rowvar=True, bias=False, ddof=None, fweights=None, aweights=None):
  m = _as_jax_array_(m)
  y = _as_jax_array_(y)
  fweights = _as_jax_array_(fweights)
  aweights = _as_jax_array_(aweights)
  return Array(jnp.cov(m, y=y, rowvar=rowvar, bias=bias, ddof=ddof,
                       fweights=fweights, aweights=aweights))


@wraps(jnp.histogram)
def histogram(a, bins=10, range=None, weights=None, density=None):
  a = _as_jax_array_(a)
  weights = _as_jax_array_(weights)
  hist, bin_edges = jnp.histogram(a=a, bins=bins, range=range, weights=weights, density=density)
  return Array(hist), Array(bin_edges)


@wraps(jnp.bincount)
def bincount(x, weights=None, minlength=0, length=None, **kwargs):
  x = _as_jax_array_(x)
  weights = _as_jax_array_(weights)
  res = jnp.bincount(x, weights=weights, minlength=minlength, length=length, **kwargs)
  return Array(res)


@wraps(jnp.digitize)
def digitize(x, bins, right=False):
  x = _as_jax_array_(x)
  bins = _as_jax_array_(bins)
  return Array(jnp.digitize(x, bins=bins, right=right))


@wraps(jnp.bartlett)
def bartlett(M):
  return Array(jnp.bartlett(M))


@wraps(jnp.blackman)
def blackman(M):
  return Array(jnp.blackman(M))


@wraps(jnp.hamming)
def hamming(M):
  return Array(jnp.hamming(M))


@wraps(jnp.hanning)
def hanning(M):
  return Array(jnp.hanning(M))


@wraps(jnp.kaiser)
def kaiser(M, beta):
  return Array(jnp.kaiser(M, beta))


# constants
# ---------

e = jnp.e
pi = jnp.pi
inf = jnp.inf


# linear algebra
# --------------


@wraps(jnp.dot)
def dot(x1, x2, **kwargs):
  x1 = _as_jax_array_(x1)
  x2 = _as_jax_array_(x2)
  return Array(jnp.dot(x1, x2, **kwargs))


@wraps(jnp.vdot)
def vdot(x1, x2, **kwargs):
  x1 = _as_jax_array_(x1)
  x2 = _as_jax_array_(x2)
  return Array(jnp.vdot(x1, x2, **kwargs))


@wraps(jnp.inner)
def inner(x1, x2, **kwargs):
  x1 = _as_jax_array_(x1)
  x2 = _as_jax_array_(x2)
  return Array(jnp.inner(x1, x2, **kwargs))


@wraps(jnp.outer)
def outer(x1, x2):
  x1 = _as_jax_array_(x1)
  x2 = _as_jax_array_(x2)
  return Array(jnp.outer(x1, x2))


@wraps(jnp.kron)
def kron(x1, x2):
  x1 = _as_jax_array_(x1)
  x2 = _as_jax_array_(x2)
  return Array(jnp.kron(x1, x2))


@wraps(jnp.matmul)
def matmul(x1, x2, **kwargs):
  x1 = _as_jax_array_(x1)
  x2 = _as_jax_array_(x2)
  return Array(jnp.matmul(x1, x2, **kwargs))


@wraps(jnp.trace)
def trace(x, offset=0, axis1=0, axis2=1, dtype=None):
  x = _as_jax_array_(x)
  return Array(jnp.trace(x, offset=offset, axis1=axis1, axis2=axis2, dtype=dtype))


# data types
# ----------

dtype = jnp.dtype
finfo = jnp.finfo
iinfo = jnp.iinfo

uint8 = jnp.uint8
uint16 = jnp.uint16
uint32 = jnp.uint32
uint64 = jnp.uint64
int8 = jnp.int8
int16 = jnp.int16
int32 = jnp.int32
int64 = jnp.int64
float16 = jnp.float16
float32 = jnp.float32
float64 = jnp.float64
complex64 = jnp.complex64
complex128 = jnp.complex128


#
@wraps(jnp.can_cast)
def can_cast(from_, to, casting=None):
  """    can_cast(from_, to, casting='safe')

    Returns True if cast between data types can occur according to the
    casting rule.  If from is a scalar or array scalar, also returns
    True if the scalar value can be cast without overflow or truncation
    to an integer.

    Parameters
    ----------
    from_ : dtype, dtype specifier, scalar, or array
        Data type, scalar, or array to cast from.
    to : dtype or dtype specifier
        Data type to cast to.
    casting : {'no', 'equiv', 'safe', 'same_kind', 'unsafe'}, optional
        Controls what kind of data casting may occur.

          * 'no' means the data types should not be cast at all.
          * 'equiv' means only byte-order changes are allowed.
          * 'safe' means only casts which can preserve values are allowed.
          * 'same_kind' means only safe casts or casts within a kind,
            like float64 to float32, are allowed.
          * 'unsafe' means any data conversions may be done.

    Returns
    -------
    out : bool
        True if cast can occur according to the casting rule.

  """
  from_ = _as_jax_array_(from_)
  to = _as_jax_array_(to)
  return jnp.can_cast(from_, to, casting=casting)


@wraps(jnp.choose)
def choose(a, choices, mode='raise'):
  a = _as_jax_array_(a)
  choices = [_as_jax_array_(c) for c in choices]
  return jnp.choose(a, choices, mode=mode)


def copy(a, order=None):
  return array(a, copy=True, order=order)


def frombuffer(buffer, dtype=float, count=-1, offset=0):
  return asarray(np.frombuffer(buffer=buffer, dtype=dtype, count=count, offset=offset))


def fromfile(file, dtype=None, count=-1, sep='', offset=0, *args, **kwargs):
  return asarray(np.fromfile(file, dtype=dtype, count=count, sep=sep, offset=offset, *args, **kwargs))


@wraps(jnp.fromfunction)
def fromfunction(function, shape, dtype=float, **kwargs):
  return jnp.fromfunction(function, shape, dtype=dtype, **kwargs)


def fromiter(iterable, dtype, count=-1, *args, **kwargs):
  iterable = _as_jax_array_(iterable)
  return asarray(np.fromiter(iterable, dtype=dtype, count=count, *args, **kwargs))


def fromstring(string, dtype=float, count=-1, *, sep):
  return asarray(np.fromstring(string=string, dtype=dtype, count=count, sep=sep))


get_printoptions = np.get_printoptions


def iscomplexobj(x):
  return np.iscomplexobj(_as_jax_array_(x))


@wraps(jnp.isneginf)
def isneginf(x):
  return Array(jnp.isneginf(_as_jax_array_(x)))


@wraps(jnp.isposinf)
def isposinf(x):
  return Array(jnp.isposinf(_as_jax_array_(x)))


def isrealobj(x):
  return not iscomplexobj(x)


issubdtype = jnp.issubdtype
issubsctype = jnp.issubsctype


def iterable(x):
  return np.iterable(_as_jax_array_(x))


@wraps(jnp.packbits)
def packbits(a, axis: Optional[int] = None, bitorder='big'):
  return Array(jnp.packbits(_as_jax_array_(a), axis=axis, bitorder=bitorder))


@wraps(jnp.piecewise)
def piecewise(x, condlist, funclist, *args, **kw):
  condlist = asarray(condlist, dtype=bool)
  return Array(jnp.piecewise(_as_jax_array_(x), condlist.value, funclist, *args, **kw))


printoptions = np.printoptions
set_printoptions = np.set_printoptions


@wraps(jnp.promote_types)
def promote_types(a, b):
  a = _as_jax_array_(a)
  b = _as_jax_array_(b)
  return jnp.promote_types(a, b)


@wraps(jnp.ravel_multi_index)
def ravel_multi_index(multi_index, dims, mode='raise', order='C'):
  multi_index = [_as_jax_array_(i) for i in multi_index]
  return Array(jnp.ravel_multi_index(multi_index, dims, mode=mode, order=order))


@wraps(jnp.result_type)
def result_type(*args):
  args = [_as_jax_array_(a) for a in args]
  return jnp.result_type(*args)


@wraps(jnp.sort_complex)
def sort_complex(a):
  return Array(jnp.sort_complex(_as_jax_array_(a)))


@wraps(jnp.unpackbits)
def unpackbits(a, axis: Optional[int] = None, count=None, bitorder='big'):
  a = _as_jax_array_(a)
  return Array(jnp.unpackbits(a, axis, count=count, bitorder=bitorder))


# Unique APIs
# -----------

add_docstring = np.add_docstring
add_newdoc = np.add_newdoc
add_newdoc_ufunc = np.add_newdoc_ufunc


@wraps(np.array2string)
def array2string(a, max_line_width=None, precision=None,
                 suppress_small=None, separator=' ', prefix="",
                 style=np._NoValue, formatter=None, threshold=None,
                 edgeitems=None, sign=None, floatmode=None, suffix="",
                 legacy=None):
  a = as_numpy(a)
  return array2string(a, max_line_width=max_line_width, precision=precision,
                      suppress_small=suppress_small, separator=separator, prefix=prefix,
                      style=style, formatter=formatter, threshold=threshold,
                      edgeitems=edgeitems, sign=sign, floatmode=floatmode, suffix=suffix,
                      legacy=legacy)


@wraps(np.asanyarray)
def asanyarray(a, dtype=None, order=None):
  return asarray(a, dtype=dtype, order=order)


@wraps(np.ascontiguousarray)
def ascontiguousarray(a, dtype=None, order=None):
  return asarray(a, dtype=dtype, order=order)


@wraps(np.asfarray)
def asfarray(a, dtype=np.float_):
  if not np.issubdtype(dtype, np.inexact):
    dtype = np.float_
  return asarray(a, dtype=dtype)


def asscalar(a):
  return a.item()


array_type = [[np.half, np.single, np.double, np.longdouble],
              [None, np.csingle, np.cdouble, np.clongdouble]]
array_precision = {np.half: 0,
                   np.single: 1,
                   np.double: 2,
                   np.longdouble: 3,
                   np.csingle: 1,
                   np.cdouble: 2,
                   np.clongdouble: 3}


@wraps(np.common_type)
def common_type(*arrays):
  is_complex = False
  precision = 0
  for a in arrays:
    t = a.dtype.type
    if iscomplexobj(a):
      is_complex = True
    if issubclass(t, jnp.integer):
      p = 2  # array_precision[_nx.double]
    else:
      p = array_precision.get(t, None)
      if p is None:
        raise TypeError("can't get common type for non-numeric array")
    precision = max(precision, p)
  if is_complex:
    return array_type[1][precision]
  else:
    return array_type[0][precision]


disp = np.disp

genfromtxt = lambda *args, **kwargs: asarray(np.genfromtxt(*args, **kwargs))
loadtxt = lambda *args, **kwargs: asarray(np.loadtxt(*args, **kwargs))

info = np.info
issubclass_ = np.issubclass_


@wraps(np.place)
def place(arr, mask, vals):
  if not isinstance(arr, Array):
    raise ValueError(f'Must be an instance of {Array.__name__}, but we got {type(arr)}')
  arr[mask] = vals


@wraps(jnp.polydiv)
def polydiv(u, v, **kwargs):
  u = _as_jax_array_(u)
  v = _as_jax_array_(v)
  res = jnp.polydiv(u, v, **kwargs)
  if isinstance(res, tuple):
    return tuple(Array(r) for r in res)
  else:
    return Array(res)


# @wraps(np.polydiv)
# def polydiv(u, v, **kwargs):
#   """
#   Returns the quotient and remainder of polynomial division.
#
#   .. note::
#      This forms part of the old polynomial API. Since version 1.4, the
#      new polynomial API defined in `numpy.polynomial` is preferred.
#      A summary of the differences can be found in the
#      :doc:`transition guide </reference/routines.polynomials>`.
#
#   The input arrays are the coefficients (including any coefficients
#   equal to zero) of the "numerator" (dividend) and "denominator"
#   (divisor) polynomials, respectively.
#
#   Parameters
#   ----------
#   u : array_like
#       Dividend polynomial's coefficients.
#
#   v : array_like
#       Divisor polynomial's coefficients.
#
#   Returns
#   -------
#   q : ArrayType
#       Coefficients, including those equal to zero, of the quotient.
#   r : ArrayType
#       Coefficients, including those equal to zero, of the remainder.
#
#   See Also
#   --------
#   poly, polyadd, polyder, polydiv, polyfit, polyint, polymul, polysub
#   polyval
#
#   Notes
#   -----
#   Both `u` and `v` must be 0-d or 1-d (ndim = 0 or 1), but `u.ndim` need
#   not equal `v.ndim`. In other words, all four possible combinations -
#   ``u.ndim = v.ndim = 0``, ``u.ndim = v.ndim = 1``,
#   ``u.ndim = 1, v.ndim = 0``, and ``u.ndim = 0, v.ndim = 1`` - work.
#
#   Examples
#   --------
#   .. math:: \\frac{3x^2 + 5x + 2}{2x + 1} = 1.5x + 1.75, remainder 0.25
#
#   >>> x = bm.array([3.0, 5.0, 2.0])
#   >>> y = bm.array([2.0, 1.0])
#   >>> bm.polydiv(x, y)
#   (ArrayType([1.5 , 1.75]), ArrayType([0.25]))
#
#   """
#   u = atleast_1d(u) + 0.0
#   v = atleast_1d(v) + 0.0
#   # w has the common type
#   w = u[0] + v[0]
#   m = len(u) - 1
#   n = len(v) - 1
#   scale = 1. / v[0]
#   q = zeros((max(m - n + 1, 1),), w.dtype)
#   r = u.astype(w.dtype)
#   for k in range(0, m - n + 1):
#     d = scale * r[k]
#     q[k] = d
#     r[k:k + n + 1] -= d * v
#   while allclose(r[0], 0, rtol=1e-14) and (r.shape[-1] > 1):
#     r = r[1:]
#   return ArrayType(q), ArrayType(r)


@wraps(np.put)
def put(a, ind, v):
  if not isinstance(a, Array):
    raise ValueError(f'Must be an instance of {Array.__name__}, but we got {type(a)}')
  a[ind] = v


@wraps(np.putmask)
def putmask(a, mask, values):
  if not isinstance(a, Array):
    raise ValueError(f'Must be an instance of {Array.__name__}, but we got {type(a)}')
  if a.shape != values.shape:
    raise ValueError('Only support the shapes of "a" and "values" are consistent.')
  a[mask] = values


@wraps(np.safe_eval)
def safe_eval(source):
  return tree_map(Array, np.safe_eval(source))


@wraps(np.savetxt)
def savetxt(fname, X, fmt='%.18e', delimiter=' ', newline='\n', header='',
            footer='', comments='# ', encoding=None):
  X = as_numpy(X)
  np.savetxt(fname, X, fmt=fmt, delimiter=delimiter, newline=newline, header=header,
             footer=footer, comments=comments, encoding=encoding)


@wraps(np.savez_compressed)
def savez_compressed(file, *args, **kwds):
  args = tuple([as_numpy(a) for a in args])
  kwds = {k: as_numpy(v) for k, v in kwds.items()}
  np.savez_compressed(file, *args, **kwds)


show_config = np.show_config
typename = np.typename


@wraps(np.copyto)
def copyto(dst, src):
  if not isinstance(dst, Array):
    raise ValueError('dst must be an instance of ArrayType.')
  dst[:] = src


@wraps(np.matrix)
def matrix(data, dtype=None):
  data = array(data, copy=True, dtype=dtype)
  if data.ndim > 2:
    raise ValueError(f'shape too large {data.shape} to be a matrix.')
  if data.ndim != 2:
    for i in range(2 - data.ndim):
      data = expand_dims(data, 0)
  return data


@wraps(np.asmatrix)
def asmatrix(data, dtype=None):
  data = array(data, dtype=dtype)
  if data.ndim > 2:
    raise ValueError(f'shape too large {data.shape} to be a matrix.')
  if data.ndim != 2:
    for i in range(2 - data.ndim):
      data = expand_dims(data, 0)
  return data


@wraps(np.mat)
def mat(data, dtype=None):
  return asmatrix(data, dtype=dtype)