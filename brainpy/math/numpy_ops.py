# -*- coding: utf-8 -*-

from typing import Optional
import jax.numpy as jnp
import numpy as np
from jax.tree_util import tree_map, tree_flatten, tree_unflatten

from brainpy.math.jaxarray import JaxArray, Variable

__all__ = [
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
  'empty', 'empty_like', 'ones', 'ones_like', 'zeros', 'zeros_like', 'full',
  'full_like', 'eye', 'identity', 'array', 'asarray', 'arange', 'linspace',
  'logspace', 'meshgrid', 'diag', 'tri', 'tril', 'triu', 'vander', 'fill_diagonal',
  'array_split',

  # indexing funcs
  'nonzero', 'where', 'tril_indices', 'tril_indices_from', 'triu_indices',
  'triu_indices_from', 'take', 'diag', 'select',

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
  'product', 'row_stack', 'apply_over_axes', 'apply_along_axis', 'array_equiv', 'array_repr', 'array_str', 'block',
  'broadcast_arrays', 'broadcast_shapes', 'broadcast_to', 'compress', 'cumproduct', 'diag_indices', 'diag_indices_from',
  'diagflat', 'diagonal', 'einsum', 'einsum_path', 'geomspace', 'gradient', 'histogram2d', 'histogram_bin_edges',
  'histogramdd', 'i0', 'in1d', 'indices', 'insert', 'intersect1d', 'iscomplex', 'isin', 'ix_', 'lexsort', 'load',
  'save', 'savez', 'mask_indices', 'msort', 'nan_to_num', 'nanargmax', 'nanargmin', 'pad', 'poly', 'polyadd', 'polyder',
  'polyfit', 'polyint', 'polymul', 'polysub', 'polyval', 'resize', 'rollaxis', 'roots', 'rot90', 'setdiff1d',
  'setxor1d', 'tensordot', 'trim_zeros', 'union1d', 'unravel_index', 'unwrap', 'take_along_axis',
  'can_cast', 'choose', 'copy', 'frombuffer', 'fromfile', 'fromfunction', 'fromiter', 'fromstring',
  'get_printoptions', 'iscomplexobj', 'isneginf', 'isposinf', 'isrealobj', 'issubdtype', 'issubsctype',
  'iterable', 'packbits', 'piecewise', 'printoptions', 'set_printoptions', 'promote_types', 'ravel_multi_index',
  'result_type', 'sort_complex', 'unpackbits',

  # unique
  'add_docstring', 'add_newdoc', 'add_newdoc_ufunc', 'array2string', 'asanyarray', 'ascontiguousarray', 'asfarray',
  'asscalar', 'common_type', 'disp', 'genfromtxt', 'loadtxt', 'info', 'issubclass_', 'place', 'polydiv', 'put',
  'putmask', 'safe_eval', 'savetxt', 'savez_compressed', 'show_config', 'typename',

  # others
  'clip_by_norm', 'as_device_array', 'as_variable', 'as_numpy', 'delete', 'remove_diag',
]

_min = min
_max = max


# others
# ------

# def as_jax_array(tensor):
#   return asarray(tensor)


def remove_diag(arr):
  if arr.ndim != 2:
    raise ValueError(f'Only support 2D matrix, while we got a {arr.ndim}D array.')
  eyes = ones(arr.shape, dtype=bool)
  fill_diagonal(eyes, False)
  return reshape(arr[eyes.value], (arr.shape[0], arr.shape[1] - 1))


def delete(arr, obj, axis=None):
  arr = _remove_jaxarray(arr)
  obj = _remove_jaxarray(obj)
  return JaxArray(jnp.delete(arr, obj, axis=axis))


def as_device_array(tensor):
  if isinstance(tensor, JaxArray):
    return tensor.value
  elif isinstance(tensor, jnp.ndarray):
    return tensor
  elif isinstance(tensor, np.ndarray):
    return jnp.asarray(tensor)
  else:
    return jnp.asarray(tensor)


def as_numpy(tensor):
  if isinstance(tensor, JaxArray):
    return tensor.numpy()
  else:
    return np.asarray(tensor)


def as_variable(tensor):
  return Variable(asarray(tensor))


def _remove_jaxarray(obj):
  if isinstance(obj, JaxArray):
    return obj.value
  else:
    return obj


def take_along_axis(a, indices, axis):
  a = _remove_jaxarray(a)
  if isinstance(indices, JaxArray): indices = indices.value
  return JaxArray(jnp.take_along_axis(a, indices, axis))


def clip_by_norm(t, clip_norm, axis=None):
  f = lambda l: l * clip_norm / maximum(sqrt(sum(l * l, axis=axis, keepdims=True)), clip_norm)
  return tree_map(f, t)


def block(arrays):
  leaves, tree = tree_flatten(arrays, is_leaf=lambda a: isinstance(a, JaxArray))
  leaves = [(l.value if isinstance(l, JaxArray) else l) for l in leaves]
  arrays = tree_unflatten(tree, leaves)
  return JaxArray(jnp.block(arrays))


def broadcast_arrays(*args):
  args = [(_remove_jaxarray(a)) for a in args]
  return jnp.broadcast_arrays(args)


broadcast_shapes = jnp.broadcast_shapes


def broadcast_to(arr, shape):
  arr = _remove_jaxarray(arr)
  return JaxArray(jnp.broadcast_to(arr, shape))


def compress(condition, a, axis=None, out=None):
  condition = _remove_jaxarray(condition)
  a = _remove_jaxarray(a)
  return JaxArray(jnp.compress(condition, a, axis, out))


def diag_indices(n, ndim=2):
  return JaxArray(jnp.diag_indices(n, ndim))


def diag_indices_from(arr):
  arr = _remove_jaxarray(arr)
  return JaxArray(jnp.diag_indices_from(arr))


def diagflat(v, k=0):
  v = _remove_jaxarray(v)
  return JaxArray(jnp.diagflat(v, k))


def diagonal(a, offset=0, axis1: int = 0, axis2: int = 1):
  a = _remove_jaxarray(a)
  return JaxArray(jnp.diagonal(a, offset, axis1, axis2))


def einsum(*operands, out=None, optimize='optimal', precision=None,
           _use_xeinsum=False):
  operands = tuple((_remove_jaxarray(a)) for a in operands)
  return JaxArray(jnp.einsum(*operands, out, optimize, precision, _use_xeinsum))


def einsum_path(subscripts, *operands, optimize='greedy'):
  operands = tuple((_remove_jaxarray(a)) for a in operands)
  return JaxArray(jnp.einsum_path(subscripts, *operands, optimize))


def geomspace(start, stop, num=50, endpoint=True, dtype=None, axis: int = 0):
  return JaxArray(jnp.geomspace(start, stop, num, endpoint, dtype, axis))


def gradient(f, *varargs, axis=None, edge_order=None):
  f = _remove_jaxarray(f)
  return JaxArray(jnp.gradient(f, *varargs, axis, edge_order))


def histogram2d(x, y, bins=10, range=None, weights=None, density=None):
  x = _remove_jaxarray(x)
  y = _remove_jaxarray(y)
  H, xedges, yedges = jnp.histogram2d(x, y, bins, range, weights, density)
  return JaxArray(H), JaxArray(xedges), JaxArray(yedges)


def histogram_bin_edges(a, bins=10, range=None, weights=None):
  a = _remove_jaxarray(a)
  return JaxArray(jnp.histogram_bin_edges(a, bins, range, weights))


def histogramdd(sample, bins=10, range=None, weights=None, density=None):
  sample = _remove_jaxarray(sample)
  return JaxArray(jnp.histogramdd(sample, bins, range, weights, density))


def i0(x):
  x = _remove_jaxarray(x)
  return JaxArray(jnp.i0(x))


def in1d(ar1, ar2, assume_unique=False, invert=False):
  ar1 = _remove_jaxarray(ar1)
  ar2 = _remove_jaxarray(ar2)
  return JaxArray(jnp.in1d(ar1, ar2, assume_unique, invert))


def indices(dimensions, dtype=None, sparse=False):
  dtype = jnp.int32 if dtype is None else dtype
  return JaxArray(jnp.indices(dimensions, dtype, sparse))


def insert(arr, obj, values, axis=None):
  arr = _remove_jaxarray(arr)
  values = _remove_jaxarray(values)
  return JaxArray(jnp.insert(arr, obj, values, axis))


def intersect1d(ar1, ar2, assume_unique=False, return_indices=False):
  ar1 = _remove_jaxarray(ar1)
  ar2 = _remove_jaxarray(ar2)
  r = jnp.intersect1d(ar1, ar2, assume_unique, return_indices)
  if return_indices:
    return JaxArray(r[0]), JaxArray(r[1]), JaxArray(r[2])
  else:
    return JaxArray(r[0])


def iscomplex(x):
  x = _remove_jaxarray(x)
  return jnp.iscomplex(x)


def isin(element, test_elements, assume_unique=False, invert=False):
  element = _remove_jaxarray(element)
  test_elements = _remove_jaxarray(test_elements)
  return JaxArray(jnp.isin(element, test_elements, assume_unique, invert))


def ix_(*args):
  args = [_remove_jaxarray(a) for a in args]
  return jnp.ix_(*args)


def lexsort(keys, axis=-1):
  leaves, tree = tree_flatten(keys, is_leaf=lambda x: isinstance(x, JaxArray))
  leaves = [_remove_jaxarray(l) for l in leaves]
  keys = tree_unflatten(tree, leaves)
  return JaxArray(jnp.lexsort(keys, axis))


load = jnp.load


def save(file, arr, allow_pickle=True, fix_imports=True):
  arr = _remove_jaxarray(arr)
  np.save(file, arr, allow_pickle, fix_imports)


def savez(file, *args, **kwds):
  args = [_remove_jaxarray(a) for a in args]
  kwds = {k: _remove_jaxarray(v) for k, v in kwds.items()}
  np.savez(file, *args, **kwds)


mask_indices = jnp.mask_indices


def msort(a):
  return JaxArray(jnp.msort(_remove_jaxarray(a)))


def nan_to_num(x, copy=True, nan=0.0, posinf=None, neginf=None):
  x = _remove_jaxarray(x)
  return JaxArray(jnp.nan_to_num(x, copy, nan=nan, posinf=posinf, neginf=neginf))


def nanargmax(a, axis=None):
  return JaxArray(jnp.nanargmax(_remove_jaxarray(a), axis))


def nanargmin(a, axis=None):
  return JaxArray(jnp.nanargmin(_remove_jaxarray(a), axis))


def pad(array, pad_width, mode="constant", **kwargs):
  array = _remove_jaxarray(array)
  pad_width = _remove_jaxarray(pad_width)
  kwargs = {k: _remove_jaxarray(v) for k, v in kwargs.items()}
  return JaxArray(jnp.pad(array, pad_width, mode, **kwargs))


def poly(seq_of_zeros):
  seq_of_zeros = _remove_jaxarray(seq_of_zeros)
  return JaxArray(jnp.poly(seq_of_zeros))


def polyadd(a1, a2):
  a1 = _remove_jaxarray(a1)
  a2 = _remove_jaxarray(a2)
  return JaxArray(jnp.polyadd(a1, a2))


def polyder(p, m=1):
  p = _remove_jaxarray(p)
  return JaxArray(jnp.polyder(p, m))


def polyfit(x, y, deg, rcond=None, full=False, w=None, cov=False):
  x = _remove_jaxarray(x)
  y = _remove_jaxarray(y)
  return jnp.polyfit(x, y, deg, rcond=rcond, full=full, w=w, cov=cov)


def polyint(p, m=1, k=None):
  p = _remove_jaxarray(p)
  return JaxArray(jnp.polyint(p, m, k))


def polymul(a1, a2):
  a1 = _remove_jaxarray(a1)
  a2 = _remove_jaxarray(a2)
  return JaxArray(jnp.polymul(a1, a2))


def polysub(a1, a2):
  a1 = _remove_jaxarray(a1)
  a2 = _remove_jaxarray(a2)
  return JaxArray(jnp.polysub(a1, a2))


def polyval(p, x):
  p = _remove_jaxarray(p)
  x = _remove_jaxarray(x)
  return JaxArray(jnp.polyval(p, x))


def resize(a, new_shape):
  a = _remove_jaxarray(a)
  return JaxArray(jnp.resize(a, new_shape))


def rollaxis(a, axis: int, start=0):
  a = _remove_jaxarray(a)
  return JaxArray(jnp.rollaxis(a, axis, start))


def roots(p):
  p = _remove_jaxarray(p)
  return JaxArray(jnp.roots(p))


def rot90(m, k=1, axes=(0, 1)):
  m = _remove_jaxarray(m)
  return JaxArray(jnp.rot90(m, k, axes))


def setdiff1d(ar1, ar2, assume_unique=False):
  return JaxArray(jnp.setdiff1d(_remove_jaxarray(ar1),
                                _remove_jaxarray(ar2),
                                assume_unique=assume_unique))


def setxor1d(ar1, ar2, assume_unique=False):
  return JaxArray(jnp.setxor1d(_remove_jaxarray(ar1),
                               _remove_jaxarray(ar2),
                               assume_unique=assume_unique))


def tensordot(a, b, axes=2):
  a = _remove_jaxarray(a)
  b = _remove_jaxarray(b)
  return JaxArray(jnp.tensordot(a, b, axes))


def trim_zeros(filt, trim='fb'):
  return JaxArray(jnp.trim_zeros(_remove_jaxarray(filt), trim))


def union1d(ar1, ar2):
  ar1 = _remove_jaxarray(ar1)
  ar2 = _remove_jaxarray(ar2)
  return JaxArray(jnp.union1d(ar1, ar2))


def unravel_index(indices, shape):
  indices = _remove_jaxarray(indices)
  shape = _remove_jaxarray(shape)
  return jnp.unravel_index(indices, shape)


def unwrap(p, discont=jnp.pi, axis: int = -1):
  p = _remove_jaxarray(p)
  return JaxArray(jnp.unwrap(p, discont, axis))


# math funcs
# ----------

# 1. Basics

def isreal(x):
  x = _remove_jaxarray(x)
  return jnp.isreal(x)


def isscalar(x):
  x = _remove_jaxarray(x)
  return jnp.isscalar(x)


def real(x):
  return x.real


def imag(x):
  return x.imag


def conj(x):
  return x.conj()


def conjugate(x):
  return x.conjugate()


def ndim(x):
  x = _remove_jaxarray(x)
  return jnp.ndim(x)


# 2. Arithmetic operations

def add(x, y):
  return x + y


def reciprocal(x):
  x = _remove_jaxarray(x)
  return JaxArray(jnp.reciprocal(x))


def negative(x):
  x = _remove_jaxarray(x)
  return JaxArray(jnp.negative(x))


def positive(x):
  x = _remove_jaxarray(x)
  return JaxArray(jnp.positive(x))


def multiply(x1, x2):
  x1 = _remove_jaxarray(x1)
  x2 = _remove_jaxarray(x2)
  return JaxArray(jnp.multiply(x1, x2))


def divide(x1, x2):
  x1 = _remove_jaxarray(x1)
  x2 = _remove_jaxarray(x2)
  return JaxArray(jnp.divide(x1, x2))


def power(x1, x2):
  x1 = _remove_jaxarray(x1)
  x2 = _remove_jaxarray(x2)
  return JaxArray(jnp.power(x1, x2))


def subtract(x1, x2):
  x1 = _remove_jaxarray(x1)
  x2 = _remove_jaxarray(x2)
  return JaxArray(jnp.subtract(x1, x2))


def true_divide(x1, x2):
  x1 = _remove_jaxarray(x1)
  x2 = _remove_jaxarray(x2)
  return JaxArray(jnp.true_divide(x1, x2))


def floor_divide(x1, x2):
  x1 = _remove_jaxarray(x1)
  x2 = _remove_jaxarray(x2)
  return JaxArray(jnp.floor_divide(x1, x2))


def float_power(x1, x2):
  x1 = _remove_jaxarray(x1)
  x2 = _remove_jaxarray(x2)
  return JaxArray(jnp.float_power(x1, x2))


def fmod(x1, x2):
  x1 = _remove_jaxarray(x1)
  x2 = _remove_jaxarray(x2)
  return JaxArray(jnp.fmod(x1, x2))


def mod(x1, x2):
  if isinstance(x1, JaxArray):  x1 = x1.value
  x2 = _remove_jaxarray(x2)
  return JaxArray(jnp.mod(x1, x2))


def divmod(x1, x2):
  x1 = _remove_jaxarray(x1)
  x2 = _remove_jaxarray(x2)
  return JaxArray(jnp.divmod(x1, x2))


def remainder(x1, x2):
  x1 = _remove_jaxarray(x1)
  x2 = _remove_jaxarray(x2)
  return JaxArray(jnp.remainder(x1, x2))


def modf(x):
  x = _remove_jaxarray(x)
  return JaxArray(jnp.modf(x))


def abs(x):
  x = _remove_jaxarray(x)
  return JaxArray(jnp.absolute(x))


def absolute(x):
  x = _remove_jaxarray(x)
  return JaxArray(jnp.absolute(x))


# 3. Exponents and logarithms
def exp(x):
  x = _remove_jaxarray(x)
  return JaxArray(jnp.exp(x))


def exp2(x):
  x = _remove_jaxarray(x)
  return JaxArray(jnp.exp2(x))


def expm1(x):
  x = _remove_jaxarray(x)
  return JaxArray(jnp.expm1(x))


def log(x):
  x = _remove_jaxarray(x)
  return JaxArray(jnp.log(x))


def log10(x):
  x = _remove_jaxarray(x)
  return JaxArray(jnp.log10(x))


def log1p(x):
  x = _remove_jaxarray(x)
  return JaxArray(jnp.log1p(x))


def log2(x):
  x = _remove_jaxarray(x)
  return JaxArray(jnp.log2(x))


def logaddexp(x1, x2):
  x1 = _remove_jaxarray(x1)
  x2 = _remove_jaxarray(x2)
  return JaxArray(jnp.logaddexp(x1, x2))


def logaddexp2(x1, x2):
  x1 = _remove_jaxarray(x1)
  x2 = _remove_jaxarray(x2)
  return JaxArray(jnp.logaddexp2(x1, x2))


# 4. Rational routines
def lcm(x1, x2):
  x1 = _remove_jaxarray(x1)
  x2 = _remove_jaxarray(x2)
  return JaxArray(jnp.lcm(x1, x2))


def gcd(x1, x2):
  x1 = _remove_jaxarray(x1)
  x2 = _remove_jaxarray(x2)
  return JaxArray(jnp.gcd(x1, x2))


# 5. trigonometric functions

def arccos(x):
  x = _remove_jaxarray(x)
  return JaxArray(jnp.arccos(x))


def arccosh(x):
  x = _remove_jaxarray(x)
  return JaxArray(jnp.arccosh(x))


def arcsin(x):
  x = _remove_jaxarray(x)
  return JaxArray(jnp.arcsin(x))


def arcsinh(x):
  x = _remove_jaxarray(x)
  return JaxArray(jnp.arcsinh(x))


def arctan(x):
  x = _remove_jaxarray(x)
  return JaxArray(jnp.arctan(x))


def arctan2(x):
  x = _remove_jaxarray(x)
  return JaxArray(jnp.arctan2(x))


def arctanh(x):
  x = _remove_jaxarray(x)
  return JaxArray(jnp.arctanh(x))


def cos(x):
  x = _remove_jaxarray(x)
  return JaxArray(jnp.cos(x))


def cosh(x):
  x = _remove_jaxarray(x)
  return JaxArray(jnp.cosh(x))


def sin(x):
  x = _remove_jaxarray(x)
  return JaxArray(jnp.sin(x))


def sinc(x):
  x = _remove_jaxarray(x)
  return JaxArray(jnp.sinc(x))


def sinh(x):
  x = _remove_jaxarray(x)
  return JaxArray(jnp.sinh(x))


def tan(x):
  x = _remove_jaxarray(x)
  return JaxArray(jnp.tan(x))


def tanh(x):
  x = _remove_jaxarray(x)
  return JaxArray(jnp.tanh(x))


def deg2rad(x):
  x = _remove_jaxarray(x)
  return JaxArray(jnp.deg2rad(x))


def rad2deg(x):
  x = _remove_jaxarray(x)
  return JaxArray(jnp.rad2deg(x))


def degrees(x):
  x = _remove_jaxarray(x)
  return JaxArray(jnp.degrees(x))


def radians(x):
  x = _remove_jaxarray(x)
  return JaxArray(jnp.radians(x))


def hypot(x1, x2):
  x1 = _remove_jaxarray(x1)
  x2 = _remove_jaxarray(x2)
  return JaxArray(jnp.hypot(x1, x2))


# 6. Rounding

def round(a, decimals=0):
  if isinstance(a, JaxArray):
    a = a.value
  return JaxArray(jnp.round(a, decimals=decimals))


around = round
round_ = round


def rint(x):
  x = _remove_jaxarray(x)
  return JaxArray(jnp.rint(x))


def floor(x):
  x = _remove_jaxarray(x)
  return JaxArray(jnp.floor(x))


def ceil(x):
  x = _remove_jaxarray(x)
  return JaxArray(jnp.ceil(x))


def trunc(x):
  x = _remove_jaxarray(x)
  return JaxArray(jnp.trunc(x))


def fix(x):
  x = _remove_jaxarray(x)
  return JaxArray(jnp.fix(x))


# 7. Sums, products, differences, Reductions


def prod(a, axis=None, dtype=None, keepdims=None, initial=None, where=None):
  a = _remove_jaxarray(a)
  r = jnp.prod(a, axis=axis, dtype=dtype, keepdims=keepdims, initial=initial, where=where)
  return r if axis is None else JaxArray(r)


product = prod


def sum(a, axis=None, dtype=None, keepdims=None, initial=None, where=None):
  a = _remove_jaxarray(a)
  r = jnp.sum(a, axis=axis, dtype=dtype, keepdims=keepdims, initial=initial, where=where)
  return r if axis is None else JaxArray(r)


def diff(a, n=1, axis: int = -1, prepend=None, append=None):
  a = _remove_jaxarray(a)
  return JaxArray(jnp.diff(a, n=n, axis=axis, prepend=prepend, append=append))


def median(a, axis=None, keepdims=False):
  a = _remove_jaxarray(a)
  r = jnp.median(a, axis=axis, keepdims=keepdims)
  return r if axis is None else JaxArray(r)


def nancumprod(a, axis=None, dtype=None):
  a = _remove_jaxarray(a)
  return JaxArray(jnp.nancumprod(a=a, axis=axis, dtype=dtype))


def nancumsum(a, axis=None, dtype=None):
  a = _remove_jaxarray(a)
  return JaxArray(jnp.nancumsum(a=a, axis=axis, dtype=dtype))


def cumprod(a, axis=None, dtype=None):
  a = _remove_jaxarray(a)
  return JaxArray(jnp.cumprod(a=a, axis=axis, dtype=dtype))


cumproduct = cumprod


def cumsum(a, axis=None, dtype=None):
  a = _remove_jaxarray(a)
  return JaxArray(jnp.cumsum(a=a, axis=axis, dtype=dtype))


def nanprod(a, axis=None, dtype=None, keepdims=None):
  a = _remove_jaxarray(a)
  r = jnp.nanprod(a=a, axis=axis, dtype=dtype, keepdims=keepdims)
  return r if axis is None else JaxArray(r)


def nansum(a, axis=None, dtype=None, keepdims=None):
  a = _remove_jaxarray(a)
  r = jnp.nansum(a=a, axis=axis, dtype=dtype, keepdims=keepdims)
  return r if axis is None else JaxArray(r)


def ediff1d(a, to_end=None, to_begin=None):
  a = _remove_jaxarray(a)
  if isinstance(to_end, JaxArray): to_end = to_end.value
  if isinstance(to_begin, JaxArray): to_begin = to_begin.value
  return JaxArray(jnp.ediff1d(a, to_end=to_end, to_begin=to_begin))


def cross(a, b, axisa=-1, axisb=-1, axisc=-1, axis=None):
  a = _remove_jaxarray(a)
  b = _remove_jaxarray(b)
  return JaxArray(jnp.cross(a, b, axisa=axisa, axisb=axisb, axisc=axisc, axis=axis))


def trapz(y, x=None, dx=1.0, axis: int = -1):
  y = _remove_jaxarray(y)
  x = _remove_jaxarray(x)
  return jnp.trapz(y, x=x, dx=dx, axis=axis)


# 8. floating_functions

def isfinite(x):
  if isinstance(x, JaxArray):
    return JaxArray(jnp.isfinite(x.value))
  else:
    return jnp.isfinite(x)


def isinf(x):
  if isinstance(x, JaxArray):
    return JaxArray(jnp.isinf(x.value))
  else:
    return jnp.isinf(x)


def isnan(x):
  if isinstance(x, JaxArray):
    return JaxArray(jnp.isnan(x.value))
  else:
    return jnp.isnan(x)


def signbit(x):
  if isinstance(x, JaxArray):
    return JaxArray(jnp.signbit(x.value))
  else:
    return jnp.signbit(x)


def nextafter(x1, x2):
  x1 = _remove_jaxarray(x1)
  x2 = _remove_jaxarray(x2)
  return JaxArray(jnp.nextafter(x1, x2))


def copysign(x1, x2):
  x1 = _remove_jaxarray(x1)
  x2 = _remove_jaxarray(x2)
  return JaxArray(jnp.copysign(x1, x2))


def ldexp(x1, x2):
  x1 = _remove_jaxarray(x1)
  x2 = _remove_jaxarray(x2)
  return JaxArray(jnp.ldexp(x1, x2))


def frexp(x):
  x = _remove_jaxarray(x)
  mantissa, exponent = jnp.frexp(x)
  return JaxArray(mantissa), JaxArray(exponent)


# 9. Miscellaneous

def convolve(a, v, mode='full'):
  a = _remove_jaxarray(a)
  v = _remove_jaxarray(v)
  return JaxArray(jnp.convolve(a, v, mode))


def sqrt(x):
  x = _remove_jaxarray(x)
  return JaxArray(jnp.sqrt(x))


def cbrt(x):
  x = _remove_jaxarray(x)
  return JaxArray(jnp.cbrt(x))


def square(x):
  x = _remove_jaxarray(x)
  return JaxArray(jnp.square(x))


def fabs(x):
  x = _remove_jaxarray(x)
  return JaxArray(jnp.fabs(x))


def sign(x):
  x = _remove_jaxarray(x)
  return JaxArray(jnp.sign(x))


def heaviside(x1, x2):
  x1 = _remove_jaxarray(x1)
  x2 = _remove_jaxarray(x2)
  return JaxArray(jnp.heaviside(x1, x2))


def maximum(x1, x2):
  x1 = _remove_jaxarray(x1)
  x2 = _remove_jaxarray(x2)
  return JaxArray(jnp.maximum(x1, x2))


def minimum(x1, x2):
  x1 = _remove_jaxarray(x1)
  x2 = _remove_jaxarray(x2)
  return JaxArray(jnp.minimum(x1, x2))


def fmax(x1, x2):
  x1 = _remove_jaxarray(x1)
  x2 = _remove_jaxarray(x2)
  return JaxArray(jnp.fmax(x1, x2))


def fmin(x1, x2):
  x1 = _remove_jaxarray(x1)
  x2 = _remove_jaxarray(x2)
  return JaxArray(jnp.fmin(x1, x2))


def interp(x, xp, fp, left=None, right=None, period=None):
  x = _remove_jaxarray(x)
  xp = _remove_jaxarray(xp)
  fp = _remove_jaxarray(fp)
  return JaxArray(jnp.interp(x, xp, fp, left=left, right=right, period=period))


def clip(a, a_min=None, a_max=None):
  a = _remove_jaxarray(a)
  a_min = _remove_jaxarray(a_min)
  a_max = _remove_jaxarray(a_max)
  return JaxArray(jnp.clip(a, a_min, a_max))


def angle(z, deg=False):
  z = _remove_jaxarray(z)
  a = jnp.angle(z)
  if deg:
    a *= 180 / pi
  return JaxArray(a)


# binary funcs
# -------------


def bitwise_not(x):
  x = _remove_jaxarray(x)
  return JaxArray(jnp.bitwise_not(x))


def invert(x):
  x = _remove_jaxarray(x)
  return JaxArray(jnp.invert(x))


def bitwise_and(x1, x2):
  x1 = _remove_jaxarray(x1)
  x2 = _remove_jaxarray(x2)
  return JaxArray(jnp.bitwise_and(x1, x2))


def bitwise_or(x1, x2):
  x1 = _remove_jaxarray(x1)
  x2 = _remove_jaxarray(x2)
  return JaxArray(jnp.bitwise_or(x1, x2))


def bitwise_xor(x1, x2):
  x1 = _remove_jaxarray(x1)
  x2 = _remove_jaxarray(x2)
  return JaxArray(jnp.bitwise_xor(x1, x2))


def left_shift(x1, x2):
  x1 = _remove_jaxarray(x1)
  x2 = _remove_jaxarray(x2)
  return JaxArray(jnp.left_shift(x1, x2))


def right_shift(x1, x2):
  x1 = _remove_jaxarray(x1)
  x2 = _remove_jaxarray(x2)
  return JaxArray(jnp.right_shift(x1, x2))


# logic funcs
# -----------

# 1. Comparison

def equal(x1, x2):
  x1 = _remove_jaxarray(x1)
  x2 = _remove_jaxarray(x2)
  return JaxArray(jnp.equal(x1, x2))


def not_equal(x1, x2):
  x1 = _remove_jaxarray(x1)
  x2 = _remove_jaxarray(x2)
  return JaxArray(jnp.not_equal(x1, x2))


def greater(x1, x2):
  x1 = _remove_jaxarray(x1)
  x2 = _remove_jaxarray(x2)
  return JaxArray(jnp.greater(x1, x2))


def greater_equal(x1, x2):
  x1 = _remove_jaxarray(x1)
  x2 = _remove_jaxarray(x2)
  return JaxArray(jnp.greater_equal(x1, x2))


def less(x1, x2):
  x1 = _remove_jaxarray(x1)
  x2 = _remove_jaxarray(x2)
  return JaxArray(jnp.less(x1, x2))


def less_equal(x1, x2):
  x1 = _remove_jaxarray(x1)
  x2 = _remove_jaxarray(x2)
  return JaxArray(jnp.less_equal(x1, x2))


def array_equal(a, b, equal_nan=False):
  a = _remove_jaxarray(a)
  b = _remove_jaxarray(b)
  return jnp.array_equal(a, b, equal_nan=equal_nan)


def isclose(a, b, rtol=1e-05, atol=1e-08, equal_nan=False):
  a = _remove_jaxarray(a)
  b = _remove_jaxarray(b)
  return JaxArray(jnp.isclose(a, b, rtol=rtol, atol=atol, equal_nan=equal_nan))


def allclose(a, b, rtol=1e-05, atol=1e-08, equal_nan=False):
  a = _remove_jaxarray(a)
  b = _remove_jaxarray(b)
  return jnp.allclose(a, b, rtol=rtol, atol=atol, equal_nan=equal_nan)


# 2. Logical operations
def logical_not(x):
  x = _remove_jaxarray(x)
  return JaxArray(jnp.logical_not(x))


def logical_and(x1, x2):
  x1 = _remove_jaxarray(x1)
  x2 = _remove_jaxarray(x2)
  return JaxArray(jnp.logical_and(x1, x2))


def logical_or(x1, x2):
  x1 = _remove_jaxarray(x1)
  x2 = _remove_jaxarray(x2)
  return JaxArray(jnp.logical_or(x1, x2))


def logical_xor(x1, x2):
  x1 = _remove_jaxarray(x1)
  x2 = _remove_jaxarray(x2)
  return JaxArray(jnp.logical_xor(x1, x2))


# 3. Truth value testing

def all(a, axis=None, keepdims=None, where=None):
  a = _remove_jaxarray(a)
  r = jnp.all(a=a, axis=axis, keepdims=keepdims, where=where)
  return r if axis is None else JaxArray(r)


def any(a, axis=None, keepdims=None, where=None):
  a = _remove_jaxarray(a)
  r = jnp.any(a=a, axis=axis, keepdims=keepdims, where=where)
  return r if axis is None else JaxArray(r)


alltrue = all
sometrue = any


# array manipulation
# ------------------


def shape(x):
  x = _remove_jaxarray(x)
  return jnp.shape(x)


def size(x, axis=None):
  x = _remove_jaxarray(x)
  r = jnp.size(x, axis=axis)
  return r if axis is None else JaxArray(r)


def reshape(x, newshape, order="C"):
  x = _remove_jaxarray(x)
  return JaxArray(jnp.reshape(x, newshape, order=order))


def ravel(x, order="C"):
  x = _remove_jaxarray(x)
  return JaxArray(jnp.ravel(x, order=order))


def moveaxis(x, source, destination):
  x = _remove_jaxarray(x)
  return JaxArray(jnp.moveaxis(x, source, destination))


def transpose(x, axis=None):
  x = _remove_jaxarray(x)
  return JaxArray(jnp.transpose(x, axes=axis))


def swapaxes(x, axis1, axis2):
  x = _remove_jaxarray(x)
  return JaxArray(jnp.swapaxes(x, axis1, axis2))


def concatenate(arrays, axis: int = 0):
  arrays = [_remove_jaxarray(a) for a in arrays]
  return JaxArray(jnp.concatenate(arrays, axis))


def stack(arrays, axis: int = 0):
  arrays = [_remove_jaxarray(a) for a in arrays]
  return JaxArray(jnp.stack(arrays, axis))


def vstack(arrays):
  arrays = [_remove_jaxarray(a) for a in arrays]
  return JaxArray(jnp.vstack(arrays))


row_stack = vstack


def hstack(arrays):
  arrays = [_remove_jaxarray(a) for a in arrays]
  return JaxArray(jnp.hstack(arrays))


def dstack(arrays):
  arrays = [_remove_jaxarray(a) for a in arrays]
  return JaxArray(jnp.dstack(arrays))


def column_stack(arrays):
  arrays = [_remove_jaxarray(a) for a in arrays]
  return JaxArray(jnp.column_stack(arrays))


def split(ary, indices_or_sections, axis=0):
  if isinstance(ary, JaxArray): ary = ary.value
  if isinstance(indices_or_sections, JaxArray): indices_or_sections = indices_or_sections.value
  return [JaxArray(a) for a in jnp.split(ary, indices_or_sections, axis=axis)]


def dsplit(ary, indices_or_sections):
  return split(ary, indices_or_sections, axis=2)


def hsplit(ary, indices_or_sections):
  return split(ary, indices_or_sections, axis=1)


def vsplit(ary, indices_or_sections):
  return split(ary, indices_or_sections, axis=0)


def tile(A, reps):
  A = _remove_jaxarray(A)
  return JaxArray(jnp.tile(A, reps))


def repeat(x, repeats, axis=None):
  x = _remove_jaxarray(x)
  return JaxArray(jnp.repeat(x, repeats=repeats, axis=axis))


def unique(x, return_index=False, return_inverse=False,
           return_counts=False, axis=None):
  x = _remove_jaxarray(x)
  return JaxArray(jnp.unique(x,
                             return_index=return_index,
                             return_inverse=return_inverse,
                             return_counts=return_counts,
                             axis=axis))


def append(arr, values, axis=None):
  arr = _remove_jaxarray(arr)
  values = _remove_jaxarray(values)
  return JaxArray(jnp.append(arr, values, axis=axis))


def flip(x, axis=None):
  x = _remove_jaxarray(x)
  return JaxArray(jnp.flip(x, axis=axis))


def fliplr(x):
  x = _remove_jaxarray(x)
  return JaxArray(jnp.fliplr(x))


def flipud(x):
  x = _remove_jaxarray(x)
  return JaxArray(jnp.flipud(x))


def roll(x, shift, axis=None):
  x = _remove_jaxarray(x)
  return JaxArray(jnp.roll(x, shift, axis=axis))


def atleast_1d(*arys):
  return jnp.atleast_1d(*[_remove_jaxarray(a) for a in arys])


def atleast_2d(*arys):
  return jnp.atleast_2d(*[_remove_jaxarray(a) for a in arys])


def atleast_3d(*arys):
  return jnp.atleast_3d(*[_remove_jaxarray(a) for a in arys])


def expand_dims(x, axis):
  x = _remove_jaxarray(x)
  return JaxArray(jnp.expand_dims(x, axis=axis))


def squeeze(x, axis=None):
  x = _remove_jaxarray(x)
  return JaxArray(jnp.squeeze(x, axis=axis))


def sort(x, axis=-1, kind='quicksort', order=None):
  x = _remove_jaxarray(x)
  return JaxArray(jnp.sort(x, axis=axis, kind=kind, order=order))


def argsort(x, axis=-1, kind='stable', order=None):
  x = _remove_jaxarray(x)
  return JaxArray(jnp.argsort(x, axis=axis, kind=kind, order=order))


def argmax(x, axis=None):
  x = _remove_jaxarray(x)
  r = jnp.argmax(x, axis=axis)
  return r if axis is None else JaxArray(r)


def argmin(x, axis=None):
  x = _remove_jaxarray(x)
  r = jnp.argmin(x, axis=axis)
  return r if axis is None else JaxArray(r)


def argwhere(x):
  x = _remove_jaxarray(x)
  return JaxArray(jnp.argwhere(x))


def nonzero(x):
  x = _remove_jaxarray(x)
  res = jnp.nonzero(x)
  return tuple([JaxArray(r) for r in res]) if isinstance(res, tuple) else JaxArray(res)


def flatnonzero(x):
  x = _remove_jaxarray(x)
  return JaxArray(jnp.flatnonzero(x))


def where(condition, x=None, y=None):
  condition = _remove_jaxarray(condition)
  x = _remove_jaxarray(x)
  y = _remove_jaxarray(y)
  return JaxArray(jnp.where(condition, x=x, y=y))


def searchsorted(a, v, side='left', sorter=None):
  a = _remove_jaxarray(a)
  v = _remove_jaxarray(v)
  return JaxArray(jnp.searchsorted(a, v, side=side, sorter=sorter))


def extract(condition, arr):
  condition = _remove_jaxarray(condition)
  arr = _remove_jaxarray(arr)
  return JaxArray(jnp.extract(condition, arr))


def count_nonzero(a, axis=None, keepdims=False):
  a = _remove_jaxarray(a)
  return jnp.count_nonzero(a, axis=axis, keepdims=keepdims)


def max(a, axis=None, keepdims=None, initial=None, where=None):
  a = _remove_jaxarray(a)
  r = jnp.max(a, axis=axis, keepdims=keepdims, initial=initial, where=where)
  return r if axis is None else JaxArray(r)


def min(a, axis=None, keepdims=None, initial=None, where=None):
  a = _remove_jaxarray(a)
  r = jnp.min(a, axis=axis, keepdims=keepdims, initial=initial, where=where)
  return r if axis is None else JaxArray(r)


amax = max
amin = min


def apply_along_axis(func1d, axis: int, arr, *args, **kwargs):
  arr = _remove_jaxarray(arr)
  return jnp.apply_along_axis(func1d, axis, arr, *args, **kwargs)


def apply_over_axes(func, a, axes):
  a = _remove_jaxarray(a)
  return jnp.apply_over_axes(func, a, axes)


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


def array_repr(arr, max_line_width=None, precision=None, suppress_small=None):
  arr = _remove_jaxarray(arr)
  return jnp.array_repr(arr, max_line_width=max_line_width, precision=precision, suppress_small=suppress_small)


def array_str(a, max_line_width=None, precision=None, suppress_small=None):
  a = _remove_jaxarray(a)
  return jnp.array_str(a, max_line_width=max_line_width, precision=precision, suppress_small=suppress_small)


def array_split(ary, indices_or_sections, axis: int = 0):
  ary = _remove_jaxarray(ary)
  if isinstance(indices_or_sections, JaxArray):
    indices_or_sections = indices_or_sections.value
  elif isinstance(indices_or_sections, (tuple, list)):
    indices_or_sections = [_remove_jaxarray(i) for i in indices_or_sections]
  return tuple([JaxArray(a) for a in jnp.array_split(ary, indices_or_sections, axis)])


# array creation
# --------------

def zeros(shape, dtype=None):
  return JaxArray(jnp.zeros(shape, dtype=dtype))


def ones(shape, dtype=None):
  return JaxArray(jnp.ones(shape, dtype=dtype))


def full(shape, fill_value, dtype=None):
  return JaxArray(jnp.full(shape, fill_value, dtype=dtype))


def empty(shape, dtype=None):
  return JaxArray(jnp.zeros(shape, dtype=dtype))


def zeros_like(a, dtype=None, shape=None):
  a = _remove_jaxarray(a)
  return JaxArray(jnp.zeros_like(a, dtype=dtype, shape=shape))


def ones_like(a, dtype=None, shape=None):
  a = _remove_jaxarray(a)
  return JaxArray(jnp.ones_like(a, dtype=dtype, shape=shape))


def empty_like(a, dtype=None, shape=None):
  a = _remove_jaxarray(a)
  return JaxArray(jnp.zeros_like(a, dtype=dtype, shape=shape))


def full_like(a, fill_value, dtype=None, shape=None):
  a = _remove_jaxarray(a)
  return JaxArray(jnp.full_like(a, fill_value, dtype=dtype, shape=shape))


def eye(N, M=None, k=0, dtype=None):
  return JaxArray(jnp.eye(N, M=M, k=k, dtype=dtype))


def identity(n, dtype=None):
  return JaxArray(jnp.identity(n, dtype=dtype))


def array(a, dtype=None, copy=True, order="K", ndmin=0):
  a = _remove_jaxarray(a)
  try:
    res = jnp.array(a, dtype=dtype, copy=copy, order=order, ndmin=ndmin)
  except TypeError:
    leaves, tree = tree_flatten(a, is_leaf=lambda a: isinstance(a, JaxArray))
    leaves = [_remove_jaxarray(l) for l in leaves]
    a = tree_unflatten(tree, leaves)
    res = jnp.array(a, dtype=dtype, copy=copy, order=order, ndmin=ndmin)
  return JaxArray(res)


def asarray(a, dtype=None, order=None):
  a = _remove_jaxarray(a)
  try:
    res = jnp.asarray(a=a, dtype=dtype, order=order)
  except TypeError:
    leaves, tree = tree_flatten(a, is_leaf=lambda a: isinstance(a, JaxArray))
    leaves = [_remove_jaxarray(l) for l in leaves]
    arrays = tree_unflatten(tree, leaves)
    res = jnp.asarray(a=arrays, dtype=dtype, order=order)
  return JaxArray(res)


def arange(*args, **kwargs):
  return JaxArray(jnp.arange(*args, **kwargs))


def linspace(*args, **kwargs):
  return JaxArray(jnp.linspace(*args, **kwargs))


def logspace(*args, **kwargs):
  return JaxArray(jnp.logspace(*args, **kwargs))


def meshgrid(*xi, copy=True, sparse=False, indexing='xy'):
  xi = [_remove_jaxarray(x) for x in xi]
  rr = jnp.meshgrid(*xi, copy=copy, sparse=sparse, indexing=indexing)
  return tuple(JaxArray(r) for r in rr)


def diag(a, k=0):
  a = _remove_jaxarray(a)
  return JaxArray(jnp.diag(a, k))


def tri(N, M=None, k=0, dtype=None):
  return JaxArray(jnp.tri(N, M=M, k=k, dtype=dtype))


def tril(a, k=0):
  a = _remove_jaxarray(a)
  return JaxArray(jnp.tril(a, k))


def triu(a, k=0):
  a = _remove_jaxarray(a)
  return JaxArray(jnp.triu(a, k))


def vander(x, N=None, increasing=False):
  x = _remove_jaxarray(x)
  return JaxArray(jnp.vander(x, N=N, increasing=increasing))


def fill_diagonal(a, val):
  if not isinstance(a, JaxArray):
    raise ValueError(f'Must be a JaxArray, but got {type(a)}')
  if a.ndim < 2:
    raise ValueError(f'Only support tensor has dimension >= 2, but got {a.shape}')
  val = _remove_jaxarray(val)
  i, j = jnp.diag_indices(_min(a.shape[-2:]))
  a._value = a.value.at[..., i, j].set(val)


# indexing funcs
# --------------

tril_indices = jnp.tril_indices
triu_indices = jnp.triu_indices


def tril_indices_from(x, k=0):
  x = _remove_jaxarray(x)
  return jnp.tril_indices_from(x, k=k)


def triu_indices_from(x, k=0):
  x = _remove_jaxarray(x)
  return jnp.triu_indices_from(x, k=k)


def take(x, indices, axis=None, mode=None):
  x = _remove_jaxarray(x)
  if isinstance(indices, JaxArray): indices = indices.value
  return JaxArray(jnp.take(x, indices=indices, axis=axis, mode=mode))


def select(condlist, choicelist, default=0):
  condlist = [_remove_jaxarray(c) for c in condlist]
  choicelist = [_remove_jaxarray(c) for c in choicelist]
  return JaxArray(jnp.select(condlist, choicelist, default=default))


# statistic funcs
# ---------------

def nanmin(x, axis=None, keepdims=None):
  x = _remove_jaxarray(x)
  r = jnp.nanmin(x, axis=axis, keepdims=keepdims)
  return r if axis is None else JaxArray(r)


def nanmax(x, axis=None, keepdims=None):
  x = _remove_jaxarray(x)
  r = jnp.nanmax(x, axis=axis, keepdims=keepdims)
  return r if axis is None else JaxArray(r)


def ptp(x, axis=None, keepdims=None):
  x = _remove_jaxarray(x)
  r = jnp.ptp(x, axis=axis, keepdims=keepdims)
  return r if axis is None else JaxArray(r)


def percentile(a, q, axis=None, interpolation='linear', keepdims=False):
  a = _remove_jaxarray(a)
  q = _remove_jaxarray(q)
  r = jnp.percentile(a=a, q=q, axis=axis, interpolation=interpolation, keepdims=keepdims)
  return r if axis is None else JaxArray(r)


def nanpercentile(a, q, axis=None, interpolation='linear', keepdims=False):
  a = _remove_jaxarray(a)
  q = _remove_jaxarray(q)
  r = jnp.nanpercentile(a=a, q=q, axis=axis, interpolation=interpolation, keepdims=keepdims)
  return r if axis is None else JaxArray(r)


def quantile(a, q, axis=None, interpolation='linear', keepdims=False):
  a = _remove_jaxarray(a)
  q = _remove_jaxarray(q)
  r = jnp.quantile(a=a, q=q, axis=axis, interpolation=interpolation, keepdims=keepdims)
  return r if axis is None else JaxArray(r)


def nanquantile(a, q, axis=None, interpolation='linear', keepdims=False):
  a = _remove_jaxarray(a)
  q = _remove_jaxarray(q)
  r = jnp.nanquantile(a=a, q=q, axis=axis, interpolation=interpolation, keepdims=keepdims)
  return r if axis is None else JaxArray(r)


def average(a, axis=None, weights=None, returned=False):
  a = _remove_jaxarray(a)
  weights = _remove_jaxarray(weights)
  r = jnp.average(a, axis=axis, weights=weights, returned=returned)
  return r if axis is None else JaxArray(r)


def mean(a, axis=None, dtype=None, keepdims=None, where=None):
  a = _remove_jaxarray(a)
  r = jnp.mean(a, axis=axis, dtype=dtype, keepdims=keepdims, where=where)
  return r if axis is None else JaxArray(r)


def std(a, axis=None, dtype=None, ddof=0, keepdims=None, where=None):
  a = _remove_jaxarray(a)
  r = jnp.std(a=a, axis=axis, dtype=dtype, ddof=ddof, keepdims=keepdims, where=where)
  return r if axis is None else JaxArray(r)


def var(a, axis=None, dtype=None, ddof=0, keepdims=None, where=None):
  a = _remove_jaxarray(a)
  r = jnp.var(a, axis=axis, dtype=dtype, ddof=ddof, keepdims=keepdims, where=where)
  return r if axis is None else JaxArray(r)


def nanmedian(a, axis=None, keepdims=False):
  return nanquantile(a, 0.5, axis=axis, keepdims=keepdims, interpolation='midpoint')


def nanmean(a, axis=None, dtype=None, keepdims=None):
  a = _remove_jaxarray(a)
  r = jnp.nanmean(a, axis=axis, dtype=dtype, keepdims=keepdims)
  return r if axis is None else JaxArray(r)


def nanstd(a, axis=None, dtype=None, ddof=0, keepdims=None, where=None):
  a = _remove_jaxarray(a)
  r = jnp.nanstd(a=a, axis=axis, dtype=dtype, ddof=ddof, keepdims=keepdims, where=where)
  return r if axis is None else JaxArray(r)


def nanvar(a, axis=None, dtype=None, ddof=0, keepdims=None, where=None):
  a = _remove_jaxarray(a)
  r = jnp.nanvar(a, axis=axis, dtype=dtype, ddof=ddof, keepdims=keepdims, where=where)
  return r if axis is None else JaxArray(r)


def corrcoef(x, y=None, rowvar=True):
  x = _remove_jaxarray(x)
  y = _remove_jaxarray(y)
  return JaxArray(jnp.corrcoef(x, y, rowvar))


def correlate(a, v, mode='valid'):
  a = _remove_jaxarray(a)
  v = _remove_jaxarray(v)
  return JaxArray(jnp.correlate(a, v, mode))


def cov(m, y=None, rowvar=True, bias=False, ddof=None, fweights=None, aweights=None):
  m = _remove_jaxarray(m)
  y = _remove_jaxarray(y)
  if isinstance(fweights, JaxArray): fweights = fweights.value
  if isinstance(aweights, JaxArray): aweights = aweights.value
  return JaxArray(jnp.cov(m, y=y, rowvar=rowvar, bias=bias, ddof=ddof,
                          fweights=fweights, aweights=aweights))


def histogram(a, bins=10, range=None, weights=None, density=None):
  a = _remove_jaxarray(a)
  weights = _remove_jaxarray(weights)
  hist, bin_edges = jnp.histogram(a=a, bins=bins, range=range, weights=weights, density=density)
  return JaxArray(hist), JaxArray(bin_edges)


def bincount(x, weights=None, minlength=None):
  x = _remove_jaxarray(x)
  weights = _remove_jaxarray(weights)
  return JaxArray(jnp.bincount(x, weights=weights, minlength=minlength))


def digitize(x, bins, right=False):
  x = _remove_jaxarray(x)
  if isinstance(bins, JaxArray): bins = bins.value
  return JaxArray(jnp.digitize(x, bins=bins, right=right))


def bartlett(M):
  return JaxArray(jnp.bartlett(M))


def blackman(M):
  return JaxArray(jnp.blackman(M))


def hamming(M):
  return JaxArray(jnp.hamming(M))


def hanning(M):
  return JaxArray(jnp.hanning(M))


def kaiser(M, beta):
  return JaxArray(jnp.kaiser(M, beta))


# constants
# ---------

e = jnp.e
pi = jnp.pi
inf = jnp.inf


# linear algebra
# --------------


def dot(x1, x2):
  x1 = _remove_jaxarray(x1)
  x2 = _remove_jaxarray(x2)
  return JaxArray(jnp.dot(x1, x2))


def vdot(x1, x2):
  x1 = _remove_jaxarray(x1)
  x2 = _remove_jaxarray(x2)
  return JaxArray(jnp.vdot(x1, x2))


def inner(x1, x2):
  x1 = _remove_jaxarray(x1)
  x2 = _remove_jaxarray(x2)
  return JaxArray(jnp.inner(x1, x2))


def outer(x1, x2):
  x1 = _remove_jaxarray(x1)
  x2 = _remove_jaxarray(x2)
  return JaxArray(jnp.outer(x1, x2))


def kron(x1, x2):
  x1 = _remove_jaxarray(x1)
  x2 = _remove_jaxarray(x2)
  return JaxArray(jnp.kron(x1, x2))


def matmul(x1, x2):
  x1 = _remove_jaxarray(x1)
  x2 = _remove_jaxarray(x2)
  return JaxArray(jnp.matmul(x1, x2))


def trace(x, offset=0, axis1=0, axis2=1, dtype=None):
  x = _remove_jaxarray(x)
  return JaxArray(jnp.trace(x, offset=offset, axis1=axis1, axis2=axis2, dtype=dtype))


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
  from_ = _remove_jaxarray(from_)
  to = _remove_jaxarray(to)
  return jnp.can_cast(from_, to, casting=casting)


def choose(a, choices, mode='raise'):
  a = _remove_jaxarray(a)
  choices = [_remove_jaxarray(c) for c in choices]
  return jnp.choose(a, choices, mode=mode)


def copy(a, order=None):
  return array(a, copy=True, order=order)


def frombuffer(buffer, dtype=float, count=-1, offset=0):
  return asarray(np.frombuffer(buffer=buffer, dtype=dtype, count=count, offset=offset))


def fromfile(file, dtype=None, count=-1, sep='', offset=0, *args, **kwargs):
  return asarray(np.fromfile(file, dtype=dtype, count=count, sep=sep, offset=offset, *args, **kwargs))


def fromfunction(function, shape, dtype=float, **kwargs):
  return jnp.fromfunction(function, shape, dtype=dtype, **kwargs)


def fromiter(iterable, dtype, count=-1, *args, **kwargs):
  iterable = _remove_jaxarray(iterable)
  return asarray(np.fromiter(iterable, dtype=dtype, count=count, *args, **kwargs))


def fromstring(string, dtype=float, count=-1, *, sep):
  return asarray(np.fromstring(string=string, dtype=dtype, count=count, sep=sep))


get_printoptions = np.get_printoptions


def iscomplexobj(x):
  return np.iscomplexobj(_remove_jaxarray(x))


def isneginf(x):
  return JaxArray(jnp.isneginf(_remove_jaxarray(x)))


def isposinf(x):
  return JaxArray(jnp.isposinf(_remove_jaxarray(x)))


def isrealobj(x):
  return not iscomplexobj(x)


issubdtype = jnp.issubdtype
issubsctype = jnp.issubsctype


def iterable(x):
  return np.iterable(_remove_jaxarray(x))


def packbits(a, axis: Optional[int] = None, bitorder='big'):
  return JaxArray(jnp.packbits(_remove_jaxarray(a), axis=axis, bitorder=bitorder))


def piecewise(x, condlist, funclist, *args, **kw):
  condlist = asarray(condlist, dtype=bool)
  return JaxArray(jnp.piecewise(_remove_jaxarray(x), condlist, funclist, *args, **kw))


printoptions = np.printoptions
set_printoptions = np.set_printoptions


def promote_types(a, b):
  a = _remove_jaxarray(a)
  b = _remove_jaxarray(b)
  return jnp.promote_types(a, b)


def ravel_multi_index(multi_index, dims, mode='raise', order='C'):
  multi_index = [_remove_jaxarray(i) for i in multi_index]
  return JaxArray(jnp.ravel_multi_index(multi_index, dims, mode=mode, order=order))


def result_type(*args):
  args = [_remove_jaxarray(a) for a in args]
  return jnp.result_type(*args)


def sort_complex(a):
  return JaxArray(jnp.sort_complex(_remove_jaxarray(a)))


def unpackbits(a, axis: Optional[int] = None, count=None, bitorder='big'):
  a = _remove_jaxarray(a)
  return JaxArray(jnp.unpackbits(a, axis, count=count, bitorder=bitorder))


# Unique APIs
# -----------

add_docstring = np.add_docstring
add_newdoc = np.add_newdoc
add_newdoc_ufunc = np.add_newdoc_ufunc


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


def asanyarray(a, dtype=None, order=None):
  return asarray(a, dtype=dtype, order=order)


def ascontiguousarray(a, dtype=None, order=None):
  return asarray(a, dtype=dtype, order=order)


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


def place(arr, mask, vals):
  if not isinstance(arr, JaxArray):
    raise ValueError(f'Must be an instance of {JaxArray.__name__}, but we got {type(arr)}')
  arr[mask] = vals


def polydiv(u, v):
  """
  Returns the quotient and remainder of polynomial division.

  .. note::
     This forms part of the old polynomial API. Since version 1.4, the
     new polynomial API defined in `numpy.polynomial` is preferred.
     A summary of the differences can be found in the
     :doc:`transition guide </reference/routines.polynomials>`.

  The input arrays are the coefficients (including any coefficients
  equal to zero) of the "numerator" (dividend) and "denominator"
  (divisor) polynomials, respectively.

  Parameters
  ----------
  u : array_like
      Dividend polynomial's coefficients.

  v : array_like
      Divisor polynomial's coefficients.

  Returns
  -------
  q : JaxArray
      Coefficients, including those equal to zero, of the quotient.
  r : JaxArray
      Coefficients, including those equal to zero, of the remainder.

  See Also
  --------
  poly, polyadd, polyder, polydiv, polyfit, polyint, polymul, polysub
  polyval

  Notes
  -----
  Both `u` and `v` must be 0-d or 1-d (ndim = 0 or 1), but `u.ndim` need
  not equal `v.ndim`. In other words, all four possible combinations -
  ``u.ndim = v.ndim = 0``, ``u.ndim = v.ndim = 1``,
  ``u.ndim = 1, v.ndim = 0``, and ``u.ndim = 0, v.ndim = 1`` - work.

  Examples
  --------
  .. math:: \\frac{3x^2 + 5x + 2}{2x + 1} = 1.5x + 1.75, remainder 0.25

  >>> x = bm.array([3.0, 5.0, 2.0])
  >>> y = bm.array([2.0, 1.0])
  >>> bm.polydiv(x, y)
  (JaxArray([1.5 , 1.75]), JaxArray([0.25]))

  """
  u = atleast_1d(u) + 0.0
  v = atleast_1d(v) + 0.0
  # w has the common type
  w = u[0] + v[0]
  m = len(u) - 1
  n = len(v) - 1
  scale = 1. / v[0]
  q = zeros((max(m - n + 1, 1),), w.dtype)
  r = u.astype(w.dtype)
  for k in range(0, m - n + 1):
    d = scale * r[k]
    q[k] = d
    r[k:k + n + 1] -= d * v
  while allclose(r[0], 0, rtol=1e-14) and (r.shape[-1] > 1):
    r = r[1:]
  return JaxArray(q), JaxArray(r)


def put(a, ind, v):
  if not isinstance(a, JaxArray):
    raise ValueError(f'Must be an instance of {JaxArray.__name__}, but we got {type(a)}')
  a[ind] = v


def putmask(a, mask, values):
  if not isinstance(a, JaxArray):
    raise ValueError(f'Must be an instance of {JaxArray.__name__}, but we got {type(a)}')
  if a.shape != values.shape:
    raise ValueError('Only support the shapes of "a" and "values" are consistent.')
  a[mask] = values


def safe_eval(source):
  return tree_map(JaxArray, np.safe_eval(source))


def savetxt(fname, X, fmt='%.18e', delimiter=' ', newline='\n', header='',
            footer='', comments='# ', encoding=None):
  X = as_numpy(X)
  np.savetxt(fname, X, fmt=fmt, delimiter=delimiter, newline=newline, header=header,
             footer=footer, comments=comments, encoding=encoding)


def savez_compressed(file, *args, **kwds):
  args = tuple([as_numpy(a) for a in args])
  kwds = {k: as_numpy(v) for k, v in kwds.items()}
  np.savez_compressed(file, *args, **kwds)


show_config = np.show_config
typename = np.typename
