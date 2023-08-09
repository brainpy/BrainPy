# -*- coding: utf-8 -*-

import jax
import jax.numpy as jnp
import numpy as np
from jax.tree_util import tree_flatten, tree_unflatten
from jax.tree_util import tree_map

from ._utils import _compatible_with_brainpy_array, _as_jax_array_
from .interoperability import *
from .ndarray import Array


__all__ = [
  'full', 'full_like', 'eye', 'identity', 'diag', 'tri', 'tril', 'triu',
  'empty', 'empty_like', 'ones', 'ones_like', 'zeros', 'zeros_like',
  'array', 'asarray', 'arange', 'linspace', 'logspace', 'fill_diagonal',

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
  'dtype', 'finfo', 'iinfo',

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


def fill_diagonal(a, val, inplace=True):
  if a.ndim < 2:
    raise ValueError(f'Only support tensor has dimension >= 2, but got {a.shape}')
  if not isinstance(a, Array) and inplace:
    raise ValueError('``fill_diagonal()`` is used in in-place updating, therefore '
                     'it requires a brainpy Array. If you want to disable '
                     'inplace updating, use ``fill_diagonal(inplace=False)``.')
  val = val.value if isinstance(val, Array) else val
  i, j = jnp.diag_indices(_min(a.shape[-2:]))
  r = as_jax(a).at[..., i, j].set(val)
  if inplace:
    a.value = r
  else:
    return r


def zeros(shape, dtype=None):
  return Array(jnp.zeros(shape, dtype=dtype))


def ones(shape, dtype=None):
  return Array(jnp.ones(shape, dtype=dtype))


def empty(shape, dtype=None):
  return Array(jnp.zeros(shape, dtype=dtype))


def zeros_like(a, dtype=None, shape=None):
  a = _as_jax_array_(a)
  return Array(jnp.zeros_like(a, dtype=dtype, shape=shape))


def ones_like(a, dtype=None, shape=None):
  a = _as_jax_array_(a)
  return Array(jnp.ones_like(a, dtype=dtype, shape=shape))


def empty_like(a, dtype=None, shape=None):
  a = _as_jax_array_(a)
  return Array(jnp.zeros_like(a, dtype=dtype, shape=shape))


def array(a, dtype=None, copy=True, order="K", ndmin=0) -> Array:
  a = _as_jax_array_(a)
  try:
    res = jnp.array(a, dtype=dtype, copy=copy, order=order, ndmin=ndmin)
  except TypeError:
    leaves, tree = tree_flatten(a, is_leaf=lambda a: isinstance(a, Array))
    leaves = [_as_jax_array_(l) for l in leaves]
    a = tree_unflatten(tree, leaves)
    res = jnp.array(a, dtype=dtype, copy=copy, order=order, ndmin=ndmin)
  return Array(res)


def asarray(a, dtype=None, order=None):
  a = _as_jax_array_(a)
  try:
    res = jnp.asarray(a=a, dtype=dtype, order=order)
  except TypeError:
    leaves, tree = tree_flatten(a, is_leaf=lambda a: isinstance(a, Array))
    leaves = [_as_jax_array_(l) for l in leaves]
    arrays = tree_unflatten(tree, leaves)
    res = jnp.asarray(a=arrays, dtype=dtype, order=order)
  return Array(res)


def arange(*args, **kwargs):
  args = [_as_jax_array_(a) for a in args]
  kwargs = {k: _as_jax_array_(v) for k, v in kwargs.items()}
  return Array(jnp.arange(*args, **kwargs))


def linspace(*args, **kwargs):
  args = [_as_jax_array_(a) for a in args]
  kwargs = {k: _as_jax_array_(v) for k, v in kwargs.items()}
  res = jnp.linspace(*args, **kwargs)
  if isinstance(res, tuple):
    return Array(res[0]), res[1]
  else:
    return Array(res)


def logspace(*args, **kwargs):
  args = [_as_jax_array_(a) for a in args]
  kwargs = {k: _as_jax_array_(v) for k, v in kwargs.items()}
  return Array(jnp.logspace(*args, **kwargs))


def asanyarray(a, dtype=None, order=None):
  return asarray(a, dtype=dtype, order=order)


def ascontiguousarray(a, dtype=None, order=None):
  return asarray(a, dtype=dtype, order=order)


def asfarray(a, dtype=np.float_):
  if not np.issubdtype(dtype, np.inexact):
    dtype = np.float_
  return asarray(a, dtype=dtype)


# Others
# ------
meshgrid = _compatible_with_brainpy_array(jnp.meshgrid)
vander = _compatible_with_brainpy_array(jnp.vander)
full = _compatible_with_brainpy_array(jnp.full)
full_like = _compatible_with_brainpy_array(jnp.full_like)
eye = _compatible_with_brainpy_array(jnp.eye)
identity = _compatible_with_brainpy_array(jnp.identity)
diag = _compatible_with_brainpy_array(jnp.diag)
tri = _compatible_with_brainpy_array(jnp.tri)
tril = _compatible_with_brainpy_array(jnp.tril)
triu = _compatible_with_brainpy_array(jnp.triu)
delete = _compatible_with_brainpy_array(jnp.delete)
take_along_axis = _compatible_with_brainpy_array(jnp.take_along_axis)
block = _compatible_with_brainpy_array(jnp.block)
broadcast_arrays = _compatible_with_brainpy_array(jnp.broadcast_arrays)
broadcast_shapes = _compatible_with_brainpy_array(jnp.broadcast_shapes)
broadcast_to = _compatible_with_brainpy_array(jnp.broadcast_to)
compress = _compatible_with_brainpy_array(jnp.compress)
diag_indices = _compatible_with_brainpy_array(jnp.diag_indices)
diag_indices_from = _compatible_with_brainpy_array(jnp.diag_indices_from)
diagflat = _compatible_with_brainpy_array(jnp.diagflat)
diagonal = _compatible_with_brainpy_array(jnp.diagonal)
einsum = _compatible_with_brainpy_array(jnp.einsum)
einsum_path = _compatible_with_brainpy_array(jnp.einsum_path)
geomspace = _compatible_with_brainpy_array(jnp.geomspace)
gradient = _compatible_with_brainpy_array(jnp.gradient)
histogram2d = _compatible_with_brainpy_array(jnp.histogram2d)
histogram_bin_edges = _compatible_with_brainpy_array(jnp.histogram_bin_edges)
histogramdd = _compatible_with_brainpy_array(jnp.histogramdd)
i0 = _compatible_with_brainpy_array(jnp.i0)
in1d = _compatible_with_brainpy_array(jnp.in1d)
indices = _compatible_with_brainpy_array(jnp.indices)
insert = _compatible_with_brainpy_array(jnp.insert)
intersect1d = _compatible_with_brainpy_array(jnp.intersect1d)
iscomplex = _compatible_with_brainpy_array(jnp.iscomplex)
isin = _compatible_with_brainpy_array(jnp.isin)
ix_ = _compatible_with_brainpy_array(jnp.ix_)
lexsort = _compatible_with_brainpy_array(jnp.lexsort)
load = _compatible_with_brainpy_array(jnp.load)
save = _compatible_with_brainpy_array(jnp.save)
savez = _compatible_with_brainpy_array(jnp.savez)
mask_indices = _compatible_with_brainpy_array(jnp.mask_indices)


def msort(a):
  """
  Return a copy of an array sorted along the first axis.

  Parameters
  ----------
  a : array_like
      Array to be sorted.

  Returns
  -------
  sorted_array : ndarray
      Array of the same type and shape as `a`.

  See Also
  --------
  sort

  Notes
  -----
  ``brainpy.math.msort(a)`` is equivalent to  ``brainpy.math.sort(a, axis=0)``.

  """
  return sort(a, axis=0)


nan_to_num = _compatible_with_brainpy_array(jnp.nan_to_num)
nanargmax = _compatible_with_brainpy_array(jnp.nanargmax)
nanargmin = _compatible_with_brainpy_array(jnp.nanargmin)
pad = _compatible_with_brainpy_array(jnp.pad)
poly = _compatible_with_brainpy_array(jnp.poly)
polyadd = _compatible_with_brainpy_array(jnp.polyadd)
polyder = _compatible_with_brainpy_array(jnp.polyder)
polyfit = _compatible_with_brainpy_array(jnp.polyfit)
polyint = _compatible_with_brainpy_array(jnp.polyint)
polymul = _compatible_with_brainpy_array(jnp.polymul)
polysub = _compatible_with_brainpy_array(jnp.polysub)
polyval = _compatible_with_brainpy_array(jnp.polyval)
resize = _compatible_with_brainpy_array(jnp.resize)
rollaxis = _compatible_with_brainpy_array(jnp.rollaxis)
roots = _compatible_with_brainpy_array(jnp.roots)
rot90 = _compatible_with_brainpy_array(jnp.rot90)
setdiff1d = _compatible_with_brainpy_array(jnp.setdiff1d)
setxor1d = _compatible_with_brainpy_array(jnp.setxor1d)
tensordot = _compatible_with_brainpy_array(jnp.tensordot)
trim_zeros = _compatible_with_brainpy_array(jnp.trim_zeros)
union1d = _compatible_with_brainpy_array(jnp.union1d)
unravel_index = _compatible_with_brainpy_array(jnp.unravel_index)
unwrap = _compatible_with_brainpy_array(jnp.unwrap)

# math funcs
# ----------
isreal = _compatible_with_brainpy_array(jnp.isreal)
isscalar = _compatible_with_brainpy_array(jnp.isscalar)
real = _compatible_with_brainpy_array(jnp.real)
imag = _compatible_with_brainpy_array(jnp.imag)
conj = _compatible_with_brainpy_array(jnp.conj)
conjugate = _compatible_with_brainpy_array(jnp.conjugate)
ndim = _compatible_with_brainpy_array(jnp.ndim)
add = _compatible_with_brainpy_array(jnp.add)
reciprocal = _compatible_with_brainpy_array(jnp.reciprocal)
negative = _compatible_with_brainpy_array(jnp.negative)
positive = _compatible_with_brainpy_array(jnp.positive)
multiply = _compatible_with_brainpy_array(jnp.multiply)
divide = _compatible_with_brainpy_array(jnp.divide)
power = _compatible_with_brainpy_array(jnp.power)
subtract = _compatible_with_brainpy_array(jnp.subtract)
true_divide = _compatible_with_brainpy_array(jnp.true_divide)
floor_divide = _compatible_with_brainpy_array(jnp.floor_divide)
float_power = _compatible_with_brainpy_array(jnp.float_power)
fmod = _compatible_with_brainpy_array(jnp.fmod)
mod = _compatible_with_brainpy_array(jnp.mod)
divmod = _compatible_with_brainpy_array(jnp.divmod)
remainder = _compatible_with_brainpy_array(jnp.remainder)
modf = _compatible_with_brainpy_array(jnp.modf)
abs = _compatible_with_brainpy_array(jnp.abs)
absolute = _compatible_with_brainpy_array(jnp.absolute)
exp = _compatible_with_brainpy_array(jnp.exp)
exp2 = _compatible_with_brainpy_array(jnp.exp2)
expm1 = _compatible_with_brainpy_array(jnp.expm1)
log = _compatible_with_brainpy_array(jnp.log)
log10 = _compatible_with_brainpy_array(jnp.log10)
log1p = _compatible_with_brainpy_array(jnp.log1p)
log2 = _compatible_with_brainpy_array(jnp.log2)
logaddexp = _compatible_with_brainpy_array(jnp.logaddexp)
logaddexp2 = _compatible_with_brainpy_array(jnp.logaddexp2)
lcm = _compatible_with_brainpy_array(jnp.lcm)
gcd = _compatible_with_brainpy_array(jnp.gcd)
arccos = _compatible_with_brainpy_array(jnp.arccos)
arccosh = _compatible_with_brainpy_array(jnp.arccosh)
arcsin = _compatible_with_brainpy_array(jnp.arcsin)
arcsinh = _compatible_with_brainpy_array(jnp.arcsinh)
arctan = _compatible_with_brainpy_array(jnp.arctan)
arctan2 = _compatible_with_brainpy_array(jnp.arctan2)
arctanh = _compatible_with_brainpy_array(jnp.arctanh)
cos = _compatible_with_brainpy_array(jnp.cos)
cosh = _compatible_with_brainpy_array(jnp.cosh)
sin = _compatible_with_brainpy_array(jnp.sin)
sinc = _compatible_with_brainpy_array(jnp.sinc)
sinh = _compatible_with_brainpy_array(jnp.sinh)
tan = _compatible_with_brainpy_array(jnp.tan)
tanh = _compatible_with_brainpy_array(jnp.tanh)
deg2rad = _compatible_with_brainpy_array(jnp.deg2rad)
rad2deg = _compatible_with_brainpy_array(jnp.rad2deg)
degrees = _compatible_with_brainpy_array(jnp.degrees)
radians = _compatible_with_brainpy_array(jnp.radians)
hypot = _compatible_with_brainpy_array(jnp.hypot)
round = _compatible_with_brainpy_array(jnp.round)
around = round
round_ = round
rint = _compatible_with_brainpy_array(jnp.rint)
floor = _compatible_with_brainpy_array(jnp.floor)
ceil = _compatible_with_brainpy_array(jnp.ceil)
trunc = _compatible_with_brainpy_array(jnp.trunc)
fix = _compatible_with_brainpy_array(jnp.fix)
prod = _compatible_with_brainpy_array(jnp.prod)

sum = _compatible_with_brainpy_array(jnp.sum)

diff = _compatible_with_brainpy_array(jnp.diff)
median = _compatible_with_brainpy_array(jnp.median)
nancumprod = _compatible_with_brainpy_array(jnp.nancumprod)
nancumsum = _compatible_with_brainpy_array(jnp.nancumsum)
cumprod = _compatible_with_brainpy_array(jnp.cumprod)
cumproduct = cumprod
cumsum = _compatible_with_brainpy_array(jnp.cumsum)
nanprod = _compatible_with_brainpy_array(jnp.nanprod)
nansum = _compatible_with_brainpy_array(jnp.nansum)
ediff1d = _compatible_with_brainpy_array(jnp.ediff1d)
cross = _compatible_with_brainpy_array(jnp.cross)
trapz = _compatible_with_brainpy_array(jnp.trapz)
isfinite = _compatible_with_brainpy_array(jnp.isfinite)
isinf = _compatible_with_brainpy_array(jnp.isinf)
isnan = _compatible_with_brainpy_array(jnp.isnan)
signbit = _compatible_with_brainpy_array(jnp.signbit)
nextafter = _compatible_with_brainpy_array(jnp.nextafter)
copysign = _compatible_with_brainpy_array(jnp.copysign)
ldexp = _compatible_with_brainpy_array(jnp.ldexp)
frexp = _compatible_with_brainpy_array(jnp.frexp)
convolve = _compatible_with_brainpy_array(jnp.convolve)
sqrt = _compatible_with_brainpy_array(jnp.sqrt)
cbrt = _compatible_with_brainpy_array(jnp.cbrt)
square = _compatible_with_brainpy_array(jnp.square)
fabs = _compatible_with_brainpy_array(jnp.fabs)
sign = _compatible_with_brainpy_array(jnp.sign)
heaviside = _compatible_with_brainpy_array(jnp.heaviside)
maximum = _compatible_with_brainpy_array(jnp.maximum)
minimum = _compatible_with_brainpy_array(jnp.minimum)
fmax = _compatible_with_brainpy_array(jnp.fmax)
fmin = _compatible_with_brainpy_array(jnp.fmin)
interp = _compatible_with_brainpy_array(jnp.interp)
clip = _compatible_with_brainpy_array(jnp.clip)
angle = _compatible_with_brainpy_array(jnp.angle)
bitwise_not = _compatible_with_brainpy_array(jnp.bitwise_not)
invert = _compatible_with_brainpy_array(jnp.invert)
bitwise_and = _compatible_with_brainpy_array(jnp.bitwise_and)
bitwise_or = _compatible_with_brainpy_array(jnp.bitwise_or)
bitwise_xor = _compatible_with_brainpy_array(jnp.bitwise_xor)
left_shift = _compatible_with_brainpy_array(jnp.left_shift)
right_shift = _compatible_with_brainpy_array(jnp.right_shift)
equal = _compatible_with_brainpy_array(jnp.equal)
not_equal = _compatible_with_brainpy_array(jnp.not_equal)
greater = _compatible_with_brainpy_array(jnp.greater)
greater_equal = _compatible_with_brainpy_array(jnp.greater_equal)
less = _compatible_with_brainpy_array(jnp.less)
less_equal = _compatible_with_brainpy_array(jnp.less_equal)
array_equal = _compatible_with_brainpy_array(jnp.array_equal)
isclose = _compatible_with_brainpy_array(jnp.isclose)
allclose = _compatible_with_brainpy_array(jnp.allclose)
logical_not = _compatible_with_brainpy_array(jnp.logical_not)
logical_and = _compatible_with_brainpy_array(jnp.logical_and)
logical_or = _compatible_with_brainpy_array(jnp.logical_or)
logical_xor = _compatible_with_brainpy_array(jnp.logical_xor)
all = _compatible_with_brainpy_array(jnp.all)
any = _compatible_with_brainpy_array(jnp.any)

alltrue = all
sometrue = any



def shape(a):
  """
  Return the shape of an array.

  Parameters
  ----------
  a : array_like
      Input array.

  Returns
  -------
  shape : tuple of ints
      The elements of the shape tuple give the lengths of the
      corresponding array dimensions.

  See Also
  --------
  len : ``len(a)`` is equivalent to ``np.shape(a)[0]`` for N-D arrays with
        ``N>=1``.
  ndarray.shape : Equivalent array method.

  Examples
  --------
  >>> brainpy.math.shape(brainpy.math.eye(3))
  (3, 3)
  >>> brainpy.math.shape([[1, 3]])
  (1, 2)
  >>> brainpy.math.shape([0])
  (1,)
  >>> brainpy.math.shape(0)
  ()

  """
  if isinstance(a, (Array, jax.Array, np.ndarray)):
    return a.shape
  else:
    return np.shape(a)


def size(a, axis=None):
  """
  Return the number of elements along a given axis.

  Parameters
  ----------
  a : array_like
      Input data.
  axis : int, optional
      Axis along which the elements are counted.  By default, give
      the total number of elements.

  Returns
  -------
  element_count : int
      Number of elements along the specified axis.

  See Also
  --------
  shape : dimensions of array
  Array.shape : dimensions of array
  Array.size : number of elements in array

  Examples
  --------
  >>> a = brainpy.math.array([[1,2,3], [4,5,6]])
  >>> brainpy.math.size(a)
  6
  >>> brainpy.math.size(a, 1)
  3
  >>> brainpy.math.size(a, 0)
  2
  """
  if isinstance(a, (Array, jax.Array, np.ndarray)):
    if axis is None:
      return a.size
    else:
      return a.shape[axis]
  else:
    return np.size(a, axis=axis)


reshape = _compatible_with_brainpy_array(jnp.reshape)
ravel = _compatible_with_brainpy_array(jnp.ravel)
moveaxis = _compatible_with_brainpy_array(jnp.moveaxis)
transpose = _compatible_with_brainpy_array(jnp.transpose)
swapaxes = _compatible_with_brainpy_array(jnp.swapaxes)
concatenate = _compatible_with_brainpy_array(jnp.concatenate)
stack = _compatible_with_brainpy_array(jnp.stack)
vstack = _compatible_with_brainpy_array(jnp.vstack)
product = prod
row_stack = vstack
hstack = _compatible_with_brainpy_array(jnp.hstack)
dstack = _compatible_with_brainpy_array(jnp.dstack)
column_stack = _compatible_with_brainpy_array(jnp.column_stack)
split = _compatible_with_brainpy_array(jnp.split)
dsplit = _compatible_with_brainpy_array(jnp.dsplit)
hsplit = _compatible_with_brainpy_array(jnp.hsplit)
vsplit = _compatible_with_brainpy_array(jnp.vsplit)
tile = _compatible_with_brainpy_array(jnp.tile)
repeat = _compatible_with_brainpy_array(jnp.repeat)
unique = _compatible_with_brainpy_array(jnp.unique)
append = _compatible_with_brainpy_array(jnp.append)
flip = _compatible_with_brainpy_array(jnp.flip)
fliplr = _compatible_with_brainpy_array(jnp.fliplr)
flipud = _compatible_with_brainpy_array(jnp.flipud)
roll = _compatible_with_brainpy_array(jnp.roll)
atleast_1d = _compatible_with_brainpy_array(jnp.atleast_1d)
atleast_2d = _compatible_with_brainpy_array(jnp.atleast_2d)
atleast_3d = _compatible_with_brainpy_array(jnp.atleast_3d)
expand_dims = _compatible_with_brainpy_array(jnp.expand_dims)
squeeze = _compatible_with_brainpy_array(jnp.squeeze)
sort = _compatible_with_brainpy_array(jnp.sort)
argsort = _compatible_with_brainpy_array(jnp.argsort)
argmax = _compatible_with_brainpy_array(jnp.argmax)
argmin = _compatible_with_brainpy_array(jnp.argmin)
argwhere = _compatible_with_brainpy_array(jnp.argwhere)
nonzero = _compatible_with_brainpy_array(jnp.nonzero)
flatnonzero = _compatible_with_brainpy_array(jnp.flatnonzero)
where = _compatible_with_brainpy_array(jnp.where)
searchsorted = _compatible_with_brainpy_array(jnp.searchsorted)
extract = _compatible_with_brainpy_array(jnp.extract)
count_nonzero = _compatible_with_brainpy_array(jnp.count_nonzero)
max = _compatible_with_brainpy_array(jnp.max)

min = _compatible_with_brainpy_array(jnp.min)

amax = max
amin = min
apply_along_axis = _compatible_with_brainpy_array(jnp.apply_along_axis)
apply_over_axes = _compatible_with_brainpy_array(jnp.apply_over_axes)
array_equiv = _compatible_with_brainpy_array(jnp.array_equiv)
array_repr = _compatible_with_brainpy_array(jnp.array_repr)
array_str = _compatible_with_brainpy_array(jnp.array_str)
array_split = _compatible_with_brainpy_array(jnp.array_split)

# indexing funcs
# --------------

tril_indices = jnp.tril_indices
triu_indices = jnp.triu_indices
tril_indices_from = _compatible_with_brainpy_array(jnp.tril_indices_from)
triu_indices_from = _compatible_with_brainpy_array(jnp.triu_indices_from)
take = _compatible_with_brainpy_array(jnp.take)
select = _compatible_with_brainpy_array(jnp.select)
nanmin = _compatible_with_brainpy_array(jnp.nanmin)
nanmax = _compatible_with_brainpy_array(jnp.nanmax)
ptp = _compatible_with_brainpy_array(jnp.ptp)
percentile = _compatible_with_brainpy_array(jnp.percentile)
nanpercentile = _compatible_with_brainpy_array(jnp.nanpercentile)
quantile = _compatible_with_brainpy_array(jnp.quantile)
nanquantile = _compatible_with_brainpy_array(jnp.nanquantile)
average = _compatible_with_brainpy_array(jnp.average)
mean = _compatible_with_brainpy_array(jnp.mean)
std = _compatible_with_brainpy_array(jnp.std)
var = _compatible_with_brainpy_array(jnp.var)
nanmedian = _compatible_with_brainpy_array(jnp.nanmedian)
nanmean = _compatible_with_brainpy_array(jnp.nanmean)
nanstd = _compatible_with_brainpy_array(jnp.nanstd)
nanvar = _compatible_with_brainpy_array(jnp.nanvar)
corrcoef = _compatible_with_brainpy_array(jnp.corrcoef)
correlate = _compatible_with_brainpy_array(jnp.correlate)
cov = _compatible_with_brainpy_array(jnp.cov)
histogram = _compatible_with_brainpy_array(jnp.histogram)
bincount = _compatible_with_brainpy_array(jnp.bincount)
digitize = _compatible_with_brainpy_array(jnp.digitize)
bartlett = _compatible_with_brainpy_array(jnp.bartlett)
blackman = _compatible_with_brainpy_array(jnp.blackman)
hamming = _compatible_with_brainpy_array(jnp.hamming)
hanning = _compatible_with_brainpy_array(jnp.hanning)
kaiser = _compatible_with_brainpy_array(jnp.kaiser)

# constants
# ---------

e = jnp.e
pi = jnp.pi
inf = jnp.inf

# linear algebra
# --------------

dot = _compatible_with_brainpy_array(jnp.dot)
vdot = _compatible_with_brainpy_array(jnp.vdot)
inner = _compatible_with_brainpy_array(jnp.inner)
outer = _compatible_with_brainpy_array(jnp.outer)
kron = _compatible_with_brainpy_array(jnp.kron)
matmul = _compatible_with_brainpy_array(jnp.matmul)
trace = _compatible_with_brainpy_array(jnp.trace)

dtype = jnp.dtype
finfo = jnp.finfo
iinfo = jnp.iinfo


can_cast = _compatible_with_brainpy_array(jnp.can_cast)
choose = _compatible_with_brainpy_array(jnp.choose)
copy = _compatible_with_brainpy_array(jnp.copy)
frombuffer = _compatible_with_brainpy_array(jnp.frombuffer)
fromfile = _compatible_with_brainpy_array(jnp.fromfile)
fromfunction = _compatible_with_brainpy_array(jnp.fromfunction)
fromiter = _compatible_with_brainpy_array(jnp.fromiter)
fromstring = _compatible_with_brainpy_array(jnp.fromstring)
get_printoptions = np.get_printoptions
iscomplexobj = _compatible_with_brainpy_array(jnp.iscomplexobj)
isneginf = _compatible_with_brainpy_array(jnp.isneginf)
isposinf = _compatible_with_brainpy_array(jnp.isposinf)
isrealobj = _compatible_with_brainpy_array(jnp.isrealobj)
issubdtype = jnp.issubdtype
issubsctype = jnp.issubsctype
iterable = _compatible_with_brainpy_array(jnp.iterable)
packbits = _compatible_with_brainpy_array(jnp.packbits)
piecewise = _compatible_with_brainpy_array(jnp.piecewise)
printoptions = np.printoptions
set_printoptions = np.set_printoptions
promote_types = _compatible_with_brainpy_array(jnp.promote_types)
ravel_multi_index = _compatible_with_brainpy_array(jnp.ravel_multi_index)
result_type = _compatible_with_brainpy_array(jnp.result_type)
sort_complex = _compatible_with_brainpy_array(jnp.sort_complex)
unpackbits = _compatible_with_brainpy_array(jnp.unpackbits)

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
    precision = _max(precision, p)
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
  if not isinstance(arr, Array):
    raise ValueError(f'Must be an instance of brainpy Array, but we got {type(arr)}')
  arr[mask] = vals


polydiv = _compatible_with_brainpy_array(jnp.polydiv)


def put(a, ind, v):
  if not isinstance(a, Array):
    raise ValueError(f'Must be an instance of brainpy Array, but we got {type(a)}')
  a[ind] = v


def putmask(a, mask, values):
  if not isinstance(a, Array):
    raise ValueError(f'Must be an instance of brainpy Array, but we got {type(a)}')
  if a.shape != values.shape:
    raise ValueError('Only support the shapes of "a" and "values" are consistent.')
  a[mask] = values


def safe_eval(source):
  return tree_map(Array, np.safe_eval(source))


def savetxt(fname, X, fmt='%.18e', delimiter=' ', newline='\n', header='',
            footer='', comments='# ', encoding=None):
  X = as_numpy(X)
  np.savetxt(fname, X, fmt=fmt, delimiter=delimiter, newline=newline, header=header,
             footer=footer, comments=comments, encoding=encoding)


def savez_compressed(file, *args, **kwds):
  args = tuple([(as_numpy(a) if isinstance(a, (jnp.ndarray, Array)) else a) for a in args])
  kwds = {k: (as_numpy(v) if isinstance(v, (jnp.ndarray, Array)) else v)
          for k, v in kwds.items()}
  np.savez_compressed(file, *args, **kwds)


show_config = np.show_config
typename = np.typename


def copyto(dst, src):
  if not isinstance(dst, Array):
    raise ValueError('dst must be an instance of ArrayType.')
  dst[:] = src


def matrix(data, dtype=None):
  data = array(data, copy=True, dtype=dtype)
  if data.ndim > 2:
    raise ValueError(f'shape too large {data.shape} to be a matrix.')
  if data.ndim != 2:
    for i in range(2 - data.ndim):
      data = expand_dims(data, 0)
  return data


def asmatrix(data, dtype=None):
  data = array(data, dtype=dtype)
  if data.ndim > 2:
    raise ValueError(f'shape too large {data.shape} to be a matrix.')
  if data.ndim != 2:
    for i in range(2 - data.ndim):
      data = expand_dims(data, 0)
  return data


def mat(data, dtype=None):
  return asmatrix(data, dtype=dtype)
