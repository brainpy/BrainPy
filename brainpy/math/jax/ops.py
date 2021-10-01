# -*- coding: utf-8 -*-

import jax.numpy as jnp
import numpy as np

from brainpy.math.jax.jaxarray import JaxArray
from brainpy.math.numpy import ops
from brainpy.tools import copy_doc


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
  'logical_or', 'logical_xor', 'all', 'any',

  # array manipulation
  'shape', 'size', 'reshape', 'ravel', 'moveaxis', 'transpose', 'swapaxes',
  'concatenate', 'stack', 'vstack', 'hstack', 'dstack', 'column_stack',
  'split', 'dsplit', 'hsplit', 'vsplit', 'tile', 'repeat', 'unique',
  'append', 'flip', 'fliplr', 'flipud', 'roll', 'atleast_1d', 'atleast_2d',
  'atleast_3d', 'expand_dims', 'squeeze', 'sort', 'argsort', 'argmax', 'argmin',
  'argwhere', 'nonzero', 'flatnonzero', 'where', 'searchsorted', 'extract',
  'count_nonzero', 'max', 'min',

  # array creation
  'empty', 'empty_like', 'ones', 'ones_like', 'zeros', 'zeros_like', 'full',
  'full_like', 'eye', 'identity', 'array', 'asarray', 'arange', 'linspace',
  'logspace', 'meshgrid', 'diag', 'tri', 'tril', 'triu', 'vander', 'fill_diagonal',

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
  'dtype', 'finfo', 'iinfo', 'bool_', 'uint8', 'uint16', 'uint32', 'uint64',
  'int_', 'int8', 'int16', 'int32', 'int64', 'float_', 'float16', 'float32',
  'float64', 'complex_', 'complex64', 'complex128', 'set_int_', 'set_float_', 'set_complex_',

  # others
  'take_along_axis', 'clip_by_norm',
]


# math funcs
# ----------

# 1. Basics

def isreal(x):
  x = x.value if isinstance(x, JaxArray) else x
  return jnp.isreal(x)


def isscalar(x):
  x = x.value if isinstance(x, JaxArray) else x
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
  x = x.value if isinstance(x, JaxArray) else x
  return jnp.ndim(x)


# 2. Arithmetic operations

def add(x, y):
  return x + y


def reciprocal(x):
  if isinstance(x, JaxArray): x = x.value
  return JaxArray(jnp.reciprocal(x))


def negative(x):
  if isinstance(x, JaxArray): x = x.value
  return JaxArray(jnp.negative(x))


def positive(x):
  if isinstance(x, JaxArray): x = x.value
  return JaxArray(jnp.positive(x.value))


def multiply(x1, x2):
  if isinstance(x1, JaxArray): x1 = x1.value
  if isinstance(x2, JaxArray): x2 = x2.value
  return JaxArray(jnp.multiply(x1, x2))


def divide(x1, x2):
  if isinstance(x1, JaxArray): x1 = x1.value
  if isinstance(x2, JaxArray): x2 = x2.value
  return JaxArray(jnp.divide(x1, x2))


def power(x1, x2):
  if isinstance(x1, JaxArray): x1 = x1.value
  if isinstance(x2, JaxArray): x2 = x2.value
  return JaxArray(jnp.power(x1, x2))


def subtract(x1, x2):
  if isinstance(x1, JaxArray): x1 = x1.value
  if isinstance(x2, JaxArray): x2 = x2.value
  return JaxArray(jnp.subtract(x1, x2))


def true_divide(x1, x2):
  if isinstance(x1, JaxArray): x1 = x1.value
  if isinstance(x2, JaxArray): x2 = x2.value
  return JaxArray(jnp.true_divide(x1, x2))


def floor_divide(x1, x2):
  if isinstance(x1, JaxArray): x1 = x1.value
  if isinstance(x2, JaxArray): x2 = x2.value
  return JaxArray(jnp.floor_divide(x1, x2))


def float_power(x1, x2):
  if isinstance(x1, JaxArray): x1 = x1.value
  if isinstance(x2, JaxArray): x2 = x2.value
  return JaxArray(jnp.float_power(x1, x2))


def fmod(x1, x2):
  if isinstance(x1, JaxArray): x1 = x1.value
  if isinstance(x2, JaxArray): x2 = x2.value
  return JaxArray(jnp.fmod(x1, x2))


def mod(x1, x2):
  if isinstance(x1, JaxArray):  x1 = x1.value
  if isinstance(x2, JaxArray): x2 = x2.value
  return JaxArray(jnp.mod(x1, x2))


def divmod(x1, x2):
  if isinstance(x1, JaxArray): x1 = x1.value
  if isinstance(x2, JaxArray): x2 = x2.value
  return JaxArray(jnp.divmod(x1, x2))


def remainder(x1, x2):
  if isinstance(x1, JaxArray): x1 = x1.value
  if isinstance(x2, JaxArray): x2 = x2.value
  return JaxArray(jnp.remainder(x1, x2))


def modf(x):
  if isinstance(x, JaxArray): x = x.value
  return JaxArray(jnp.modf(x))


def abs(x):
  if isinstance(x, JaxArray): x = x.value
  return JaxArray(jnp.absolute(x))


def absolute(x):
  if isinstance(x, JaxArray): x = x.value
  return JaxArray(jnp.absolute(x))


# 3. Exponents and logarithms
def exp(x):
  if isinstance(x, JaxArray): x = x.value
  return JaxArray(jnp.exp(x))


def exp2(x):
  if isinstance(x, JaxArray): x = x.value
  return JaxArray(jnp.exp2(x))


def expm1(x):
  if isinstance(x, JaxArray): x = x.value
  return JaxArray(jnp.expm1(x))


def log(x):
  if isinstance(x, JaxArray): x = x.value
  return JaxArray(jnp.log(x))


def log10(x):
  if isinstance(x, JaxArray): x = x.value
  return JaxArray(jnp.log10(x))


def log1p(x):
  if isinstance(x, JaxArray): x = x.value
  return JaxArray(jnp.log1p(x))


def log2(x):
  if isinstance(x, JaxArray): x = x.value
  return JaxArray(jnp.log2(x))


def logaddexp(x1, x2):
  if isinstance(x1, JaxArray): x1 = x1.value
  if isinstance(x2, JaxArray): x2 = x2.value
  return JaxArray(jnp.logaddexp(x1, x2))


def logaddexp2(x1, x2):
  if isinstance(x1, JaxArray): x1 = x1.value
  if isinstance(x2, JaxArray): x2 = x2.value
  return JaxArray(jnp.logaddexp2(x1, x2))


# 4. Rational routines
def lcm(x1, x2):
  if isinstance(x1, JaxArray): x1 = x1.value
  if isinstance(x2, JaxArray): x2 = x2.value
  return JaxArray(jnp.lcm(x1, x2))


def gcd(x1, x2):
  if isinstance(x1, JaxArray): x1 = x1.value
  if isinstance(x2, JaxArray): x2 = x2.value
  return JaxArray(jnp.gcd(x1, x2))


# 5. trigonometric functions

def arccos(x):
  if isinstance(x, JaxArray): x = x.value
  return JaxArray(jnp.arccos(x))


def arccosh(x):
  if isinstance(x, JaxArray): x = x.value
  return JaxArray(jnp.arccosh(x))


def arcsin(x):
  if isinstance(x, JaxArray): x = x.value
  return JaxArray(jnp.arcsin(x))


def arcsinh(x):
  if isinstance(x, JaxArray): x = x.value
  return JaxArray(jnp.arcsinh(x))


def arctan(x):
  if isinstance(x, JaxArray): x = x.value
  return JaxArray(jnp.arctan(x))


def arctan2(x):
  if isinstance(x, JaxArray): x = x.value
  return JaxArray(jnp.arctan2(x))


def arctanh(x):
  if isinstance(x, JaxArray): x = x.value
  return JaxArray(jnp.arctanh(x))


def cos(x):
  if isinstance(x, JaxArray): x = x.value
  return JaxArray(jnp.cos(x))


def cosh(x):
  if isinstance(x, JaxArray): x = x.value
  return JaxArray(jnp.cosh(x))


def sin(x):
  if isinstance(x, JaxArray): x = x.value
  return JaxArray(jnp.sin(x))


def sinc(x):
  if isinstance(x, JaxArray): x = x.value
  return JaxArray(jnp.sinc(x))


def sinh(x):
  if isinstance(x, JaxArray): x = x.value
  return JaxArray(jnp.sinh(x))


def tan(x):
  if isinstance(x, JaxArray): x = x.value
  return JaxArray(jnp.tan(x))


def tanh(x):
  if isinstance(x, JaxArray): x = x.value
  return JaxArray(jnp.tanh(x))


def deg2rad(x):
  if isinstance(x, JaxArray): x = x.value
  return JaxArray(jnp.deg2rad(x))


def rad2deg(x):
  if isinstance(x, JaxArray): x = x.value
  return JaxArray(jnp.rad2deg(x))


def degrees(x):
  if isinstance(x, JaxArray): x = x.value
  return JaxArray(jnp.degrees(x))


def radians(x):
  if isinstance(x, JaxArray): x = x.value
  return JaxArray(jnp.radians(x))


def hypot(x1, x2):
  if isinstance(x1, JaxArray): x1 = x1.value
  if isinstance(x2, JaxArray): x2 = x2.value
  return JaxArray(jnp.hypot(x1, x2))


# 6. Rounding

def round(a, decimals=0):
  if isinstance(a, JaxArray):
    a = a.value
  return JaxArray(jnp.round(a, decimals=decimals))


around = round
round_ = round


def rint(x):
  if isinstance(x, JaxArray): x = x.value
  return JaxArray(jnp.rint(x))


def floor(x):
  if isinstance(x, JaxArray): x = x.value
  return JaxArray(jnp.floor(x))


def ceil(x):
  if isinstance(x, JaxArray): x = x.value
  return JaxArray(jnp.ceil(x))


def trunc(x):
  if isinstance(x, JaxArray): x = x.value
  return JaxArray(jnp.trunc(x))


def fix(x):
  if isinstance(x, JaxArray): x = x.value
  return JaxArray(jnp.fix(x))


# 7. Sums, products, differences, Reductions


def prod(a, axis=None, dtype=None, keepdims=None, initial=None, where=None):
  if isinstance(a, JaxArray): a = a.value
  r = jnp.prod(a, axis=axis, dtype=dtype, keepdims=keepdims, initial=initial, where=where)
  return r if axis is None else JaxArray(r)


def sum(a, axis=None, dtype=None, keepdims=None, initial=None, where=None):
  if isinstance(a, JaxArray): a = a.value
  r = jnp.sum(a, axis=axis, dtype=dtype, keepdims=keepdims, initial=initial, where=where)
  return r if axis is None else JaxArray(r)


def diff(a, n=1, axis: int = -1, prepend=None, append=None):
  if isinstance(a, JaxArray): a = a.value
  return JaxArray(jnp.diff(a, n=n, axis=axis, prepend=prepend, append=append))


def median(a, axis=None, keepdims=False):
  if isinstance(a, JaxArray): a = a.value
  r = jnp.median(a, axis=axis, keepdims=keepdims)
  return r if axis is None else JaxArray(r)


def nancumprod(a, axis=None, dtype=None):
  if isinstance(a, JaxArray): a = a.value
  return JaxArray(jnp.nancumprod(a=a, axis=axis, dtype=dtype))


def nancumsum(a, axis=None, dtype=None):
  if isinstance(a, JaxArray): a = a.value
  return JaxArray(jnp.nancumsum(a=a, axis=axis, dtype=dtype))


def cumprod(a, axis=None, dtype=None):
  if isinstance(a, JaxArray): a = a.value
  return JaxArray(jnp.cumprod(a=a, axis=axis, dtype=dtype))


def cumsum(a, axis=None, dtype=None):
  if isinstance(a, JaxArray): a = a.value
  return JaxArray(jnp.cumsum(a=a, axis=axis, dtype=dtype))


def nanprod(a, axis=None, dtype=None, keepdims=None):
  if isinstance(a, JaxArray): a = a.value
  r = jnp.nanprod(a=a, axis=axis, dtype=dtype, keepdims=keepdims)
  return r if axis is None else JaxArray(r)


def nansum(a, axis=None, dtype=None, keepdims=None):
  if isinstance(a, JaxArray): a = a.value
  r = jnp.nansum(a=a, axis=axis, dtype=dtype, keepdims=keepdims)
  return r if axis is None else JaxArray(r)


def ediff1d(a, to_end=None, to_begin=None):
  if isinstance(a, JaxArray): a = a.value
  if isinstance(to_end, JaxArray): to_end = to_end.value
  if isinstance(to_begin, JaxArray): to_begin = to_begin.value
  return JaxArray(jnp.ediff1d(a, to_end=to_end, to_begin=to_begin))


def cross(a, b, axisa=-1, axisb=-1, axisc=-1, axis=None):
  if isinstance(a, JaxArray): a = a.value
  if isinstance(b, JaxArray): b = b.value
  return JaxArray(jnp.cross(a, b, axisa=axisa, axisb=axisb, axisc=axisc, axis=axis))


def trapz(y, x=None, dx=1.0, axis: int = -1):
  if isinstance(y, JaxArray): y = y.value
  if isinstance(x, JaxArray): x = x.value
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
  if isinstance(x1, JaxArray): x1 = x1.value
  if isinstance(x2, JaxArray): x2 = x2.value
  return JaxArray(jnp.nextafter(x1, x2))


def copysign(x1, x2):
  if isinstance(x1, JaxArray): x1 = x1.value
  if isinstance(x2, JaxArray): x2 = x2.value
  return JaxArray(jnp.copysign(x1, x2))


def ldexp(x1, x2):
  if isinstance(x1, JaxArray): x1 = x1.value
  if isinstance(x2, JaxArray): x2 = x2.value
  return JaxArray(jnp.ldexp(x1, x2))


def frexp(x):
  if isinstance(x, JaxArray): x = x.value
  mantissa, exponent = jnp.frexp(x)
  return JaxArray(mantissa), JaxArray(exponent)


# 9. Miscellaneous

def convolve(a, v, mode='full'):
  if isinstance(a, JaxArray): a = a.value
  if isinstance(v, JaxArray): v = v.value
  return JaxArray(jnp.convolve(a, v, mode))


def sqrt(x):
  if isinstance(x, JaxArray):
    return JaxArray(jnp.sqrt(x.value))
  else:
    return jnp.sqrt(x)


def cbrt(x):
  if isinstance(x, JaxArray): x = x.value
  return JaxArray(jnp.cbrt(x))


def square(x):
  if isinstance(x, JaxArray): x = x.value
  return JaxArray(jnp.square(x))


def fabs(x):
  if isinstance(x, JaxArray): x = x.value
  return JaxArray(jnp.fabs(x))


def sign(x):
  if isinstance(x, JaxArray): x = x.value
  return JaxArray(jnp.sign(x))


def heaviside(x1, x2):
  if isinstance(x1, JaxArray): x1 = x1.value
  if isinstance(x2, JaxArray): x2 = x2.value
  return JaxArray(jnp.heaviside(x1, x2))


def maximum(x1, x2):
  if isinstance(x1, JaxArray): x1 = x1.value
  if isinstance(x2, JaxArray): x2 = x2.value
  return JaxArray(jnp.maximum(x1, x2))


def minimum(x1, x2):
  if isinstance(x1, JaxArray): x1 = x1.value
  if isinstance(x2, JaxArray): x2 = x2.value
  return JaxArray(jnp.minimum(x1, x2))


def fmax(x1, x2):
  if isinstance(x1, JaxArray): x1 = x1.value
  if isinstance(x2, JaxArray): x2 = x2.value
  return JaxArray(jnp.fmax(x1, x2))


def fmin(x1, x2):
  if isinstance(x1, JaxArray): x1 = x1.value
  if isinstance(x2, JaxArray): x2 = x2.value
  return JaxArray(jnp.fmin(x1, x2))


def interp(x, xp, fp, left=None, right=None, period=None):
  if isinstance(x, JaxArray): x = x.value
  if isinstance(xp, JaxArray): xp = xp.value
  if isinstance(fp, JaxArray): fp = fp.value
  return JaxArray(jnp.interp(x, xp, fp, left=left, right=right, period=period))


def clip(a, a_min=None, a_max=None):
  if isinstance(a, JaxArray): a = a.value
  if isinstance(a_min, JaxArray): a_min = a_min.value
  if isinstance(a_max, JaxArray): a_max = a_max.value
  return JaxArray(jnp.clip(a, a_min, a_max))


def angle(z, deg=False):
  if isinstance(z, JaxArray): z = z.value
  a = jnp.angle(z)
  if deg:
    a *= 180 / pi
  return JaxArray(a)


# binary funcs
# -------------


def bitwise_not(x):
  if isinstance(x, JaxArray): x = x.value
  return JaxArray(jnp.bitwise_not(x))


def invert(x):
  if isinstance(x, JaxArray): x = x.value
  return JaxArray(jnp.invert(x))


def bitwise_and(x1, x2):
  if isinstance(x1, JaxArray): x1 = x1.value
  if isinstance(x2, JaxArray): x2 = x2.value
  return JaxArray(jnp.bitwise_and(x1, x2))


def bitwise_or(x1, x2):
  if isinstance(x1, JaxArray): x1 = x1.value
  if isinstance(x2, JaxArray): x2 = x2.value
  return JaxArray(jnp.bitwise_or(x1, x2))


def bitwise_xor(x1, x2):
  if isinstance(x1, JaxArray): x1 = x1.value
  if isinstance(x2, JaxArray): x2 = x2.value
  return JaxArray(jnp.bitwise_xor(x1, x2))


def left_shift(x1, x2):
  if isinstance(x1, JaxArray): x1 = x1.value
  if isinstance(x2, JaxArray): x2 = x2.value
  return JaxArray(jnp.left_shift(x1, x2))


def right_shift(x1, x2):
  if isinstance(x1, JaxArray): x1 = x1.value
  if isinstance(x2, JaxArray): x2 = x2.value
  return JaxArray(jnp.right_shift(x1, x2))


# logic funcs
# -----------

# 1. Comparison

def equal(x1, x2):
  if isinstance(x1, JaxArray): x1 = x1.value
  if isinstance(x2, JaxArray): x2 = x2.value
  return JaxArray(jnp.equal(x1, x2))


def not_equal(x1, x2):
  if isinstance(x1, JaxArray): x1 = x1.value
  if isinstance(x2, JaxArray): x2 = x2.value
  return JaxArray(jnp.not_equal(x1, x2))


def greater(x1, x2):
  if isinstance(x1, JaxArray): x1 = x1.value
  if isinstance(x2, JaxArray): x2 = x2.value
  return JaxArray(jnp.greater(x1, x2))


def greater_equal(x1, x2):
  if isinstance(x1, JaxArray): x1 = x1.value
  if isinstance(x2, JaxArray): x2 = x2.value
  return JaxArray(jnp.greater_equal(x1, x2))


def less(x1, x2):
  if isinstance(x1, JaxArray): x1 = x1.value
  if isinstance(x2, JaxArray): x2 = x2.value
  return JaxArray(jnp.less(x1, x2))


def less_equal(x1, x2):
  if isinstance(x1, JaxArray): x1 = x1.value
  if isinstance(x2, JaxArray): x2 = x2.value
  return JaxArray(jnp.less_equal(x1, x2))


def array_equal(a, b, equal_nan=False):
  if isinstance(a, JaxArray): a = a.value
  if isinstance(b, JaxArray): b = b.value
  return jnp.array_equal(a, b, equal_nan=equal_nan)


def isclose(a, b, rtol=1e-05, atol=1e-08, equal_nan=False):
  if isinstance(a, JaxArray): a = a.value
  if isinstance(b, JaxArray): b = b.value
  return JaxArray(jnp.isclose(a, b, rtol=rtol, atol=atol, equal_nan=equal_nan))


def allclose(a, b, rtol=1e-05, atol=1e-08, equal_nan=False):
  if isinstance(a, JaxArray): a = a.value
  if isinstance(b, JaxArray): b = b.value
  return jnp.allclose(a, b, rtol=rtol, atol=atol, equal_nan=equal_nan)


# 2. Logical operations
def logical_not(x):
  if isinstance(x, JaxArray): x = x.value
  return JaxArray(jnp.logical_not(x))


def logical_and(x1, x2):
  if isinstance(x1, JaxArray): x1 = x1.value
  if isinstance(x2, JaxArray): x2 = x2.value
  return JaxArray(jnp.logical_and(x1, x2))


def logical_or(x1, x2):
  if isinstance(x1, JaxArray): x1 = x1.value
  if isinstance(x2, JaxArray): x2 = x2.value
  return JaxArray(jnp.logical_or(x1, x2))


def logical_xor(x1, x2):
  if isinstance(x1, JaxArray): x1 = x1.value
  if isinstance(x2, JaxArray): x2 = x2.value
  return JaxArray(jnp.logical_xor(x1, x2))


# 3. Truth value testing

def all(a, axis=None, keepdims=None, where=None):
  if isinstance(a, JaxArray): a = a.value
  r = jnp.all(a=a, axis=axis, keepdims=keepdims, where=where)
  return r if axis is None else JaxArray(r)


def any(a, axis=None, keepdims=None, where=None):
  if isinstance(a, JaxArray): a = a.value
  r = jnp.any(a=a, axis=axis, keepdims=keepdims, where=where)
  return r if axis is None else JaxArray(r)


# array manipulation
# ------------------


def shape(x):
  if isinstance(x, JaxArray): x = x.value
  return jnp.shape(x)


def size(x, axis=None):
  if isinstance(x, JaxArray): x = x.value
  r = jnp.size(x, axis=axis)
  return r if axis is None else JaxArray(r)


def reshape(x, newshape, order="C"):
  if isinstance(x, JaxArray): x = x.value
  return JaxArray(jnp.reshape(x, newshape, order=order))


def ravel(x, order="C"):
  if isinstance(x, JaxArray): x = x.value
  return JaxArray(jnp.ravel(x, order=order))


def moveaxis(x, source, destination):
  if isinstance(x, JaxArray): x = x.value
  return JaxArray(jnp.moveaxis(x, source, destination))


def transpose(x, axis=None):
  if isinstance(x, JaxArray): x = x.value
  return JaxArray(jnp.transpose(x, axes=axis))


def swapaxes(x, axis1, axis2):
  if isinstance(x, JaxArray): x = x.value
  return JaxArray(jnp.swapaxes(x, axis1, axis2))


def concatenate(arrays, axis: int = 0):
  arrays = [a.value if isinstance(a, JaxArray) else a for a in arrays]
  return JaxArray(jnp.concatenate(arrays, axis))


def stack(arrays, axis: int = 0):
  arrays = [a.value if isinstance(a, JaxArray) else a for a in arrays]
  return JaxArray(jnp.stack(arrays, axis))


def vstack(arrays):
  arrays = [a.value if isinstance(a, JaxArray) else a for a in arrays]
  return JaxArray(jnp.vstack(arrays))


def hstack(arrays):
  arrays = [a.value if isinstance(a, JaxArray) else a for a in arrays]
  return JaxArray(jnp.hstack(arrays))


def dstack(arrays):
  arrays = [a.value if isinstance(a, JaxArray) else a for a in arrays]
  return JaxArray(jnp.dstack(arrays))


def column_stack(arrays):
  arrays = [a.value if isinstance(a, JaxArray) else a for a in arrays]
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
  if isinstance(A, JaxArray): A = A.value
  return JaxArray(jnp.tile(A, reps))


def repeat(x, repeats, axis=None):
  if isinstance(x, JaxArray): x = x.value
  return JaxArray(jnp.repeat(x, repeats=repeats, axis=axis))


def unique(x, return_index=False, return_inverse=False,
           return_counts=False, axis=None):
  if isinstance(x, JaxArray): x = x.value
  return JaxArray(jnp.unique(x,
                             return_index=return_index,
                             return_inverse=return_inverse,
                             return_counts=return_counts,
                             axis=axis))


def append(arr, values, axis=None):
  if isinstance(arr, JaxArray): arr = arr.value
  if isinstance(values, JaxArray): values = values.value
  return JaxArray(jnp.append(arr, values, axis=axis))


def flip(x, axis=None):
  if isinstance(x, JaxArray): x = x.value
  return JaxArray(jnp.flip(x, axis=axis))


def fliplr(x):
  if isinstance(x, JaxArray): x = x.value
  return JaxArray(jnp.fliplr(x))


def flipud(x):
  if isinstance(x, JaxArray): x = x.value
  return JaxArray(jnp.flipud(x))


def roll(x, shift, axis=None):
  if isinstance(x, JaxArray): x = x.value
  return JaxArray(jnp.roll(x, shift, axis=axis))


def atleast_1d(*arys):
  return jnp.atleast_1d(*[a.value if isinstance(a, JaxArray) else a for a in arys])


def atleast_2d(*arys):
  return jnp.atleast_2d(*[a.value if isinstance(a, JaxArray) else a for a in arys])


def atleast_3d(*arys):
  return jnp.atleast_3d(*[a.value if isinstance(a, JaxArray) else a for a in arys])


def expand_dims(x, axis):
  if isinstance(x, JaxArray): x = x.value
  return JaxArray(jnp.expand_dims(x, axis=axis))


def squeeze(x, axis=None):
  if isinstance(x, JaxArray): x = x.value
  return JaxArray(jnp.squeeze(x, axis=axis))


def sort(x, axis=-1, kind='quicksort', order=None):
  if isinstance(x, JaxArray): x = x.value
  return JaxArray(jnp.sort(x, axis=axis, kind=kind, order=order))


def argsort(x, axis=-1, kind='quicksort', order=None):
  if isinstance(x, JaxArray): x = x.value
  return JaxArray(jnp.argsort(x, axis=axis, kind=kind, order=order))


def argmax(x, axis=None):
  if isinstance(x, JaxArray): x = x.value
  r = jnp.argmax(x, axis=axis)
  return r if axis is None else JaxArray(r)


def argmin(x, axis=None):
  if isinstance(x, JaxArray): x = x.value
  r = jnp.argmin(x, axis=axis)
  return r if axis is None else JaxArray(r)


def argwhere(x):
  if isinstance(x, JaxArray): x = x.value
  return JaxArray(jnp.argwhere(x))


def nonzero(x):
  if isinstance(x, JaxArray): x = x.value
  return jnp.nonzero(x)


def flatnonzero(x):
  if isinstance(x, JaxArray): x = x.value
  return JaxArray(jnp.flatnonzero(x))


def where(condition, x=None, y=None):
  if isinstance(condition, JaxArray): condition = condition.value
  if isinstance(x, JaxArray): x = x.value
  if isinstance(y, JaxArray): y = y.value
  return jnp.where(condition, x=x, y=y)


def searchsorted(a, v, side='left', sorter=None):
  if isinstance(a, JaxArray): a = a.value
  if isinstance(v, JaxArray): v = v.value
  return JaxArray(jnp.searchsorted(a, v, side=side, sorter=sorter))


def extract(condition, arr):
  if isinstance(condition, JaxArray): condition = condition.value
  if isinstance(arr, JaxArray): arr = arr.value
  return JaxArray(jnp.extract(condition, arr))


def count_nonzero(a, axis=None, keepdims=False):
  if isinstance(a, JaxArray): a = a.value
  return jnp.count_nonzero(a, axis=axis, keepdims=keepdims)


def max(a, axis=None, keepdims=None, initial=None, where=None):
  if isinstance(a, JaxArray): a = a.value
  r = jnp.max(a, axis=axis, keepdims=keepdims, initial=initial, where=where)
  return r if axis is None else JaxArray(r)


def min(a, axis=None, keepdims=None, initial=None, where=None):
  if isinstance(a, JaxArray): a = a.value
  r = jnp.min(a, axis=axis, keepdims=keepdims, initial=initial, where=where)
  return r if axis is None else JaxArray(r)


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
  if isinstance(a, JaxArray): a = a.value
  return JaxArray(jnp.zeros_like(a, dtype=dtype, shape=shape))


def ones_like(a, dtype=None, shape=None):
  if isinstance(a, JaxArray): a = a.value
  return JaxArray(jnp.ones_like(a, dtype=dtype, shape=shape))


def empty_like(a, dtype=None, shape=None):
  if isinstance(a, JaxArray): a = a.value
  return JaxArray(jnp.zeros_like(a, dtype=dtype, shape=shape))


def full_like(a, fill_value, dtype=None, shape=None):
  if isinstance(a, JaxArray): a = a.value
  return JaxArray(jnp.full_like(a, fill_value, dtype=dtype, shape=shape))


def eye(N, M=None, k=0, dtype=None):
  return JaxArray(jnp.eye(N, M=M, k=k, dtype=dtype))


def identity(n, dtype=None):
  return JaxArray(jnp.identity(n, dtype=dtype))


def array(object, dtype=None, copy=True, order="K", ndmin=0):
  if isinstance(object, JaxArray): object = object.value
  return JaxArray(jnp.array(object, dtype=dtype, copy=copy, order=order, ndmin=ndmin))


def asarray(a, dtype=None, order=None):
  if isinstance(a, JaxArray): a = a.value
  return JaxArray(jnp.asarray(a=a, dtype=dtype, order=order))


def arange(*args, **kwargs):
  return JaxArray(jnp.arange(*args, **kwargs))


def linspace(*args, **kwargs):
  return JaxArray(jnp.linspace(*args, **kwargs))


def logspace(*args, **kwargs):
  return JaxArray(jnp.logspace(*args, **kwargs))


def meshgrid(*xi, copy=True, sparse=False, indexing='xy'):
  xi = [x.value if isinstance(x, JaxArray) else x for x in xi]
  return JaxArray(jnp.meshgrid(*xi, copy=copy, sparse=sparse, indexing=indexing))


def diag(a, k=0):
  if isinstance(a, JaxArray): a = a.value
  return JaxArray(jnp.diag(a, k))


def tri(N, M=None, k=0, dtype=None):
  return JaxArray(jnp.tri(N, M=M, k=k, dtype=dtype))


def tril(a, k=0):
  if isinstance(a, JaxArray): a = a.value
  return JaxArray(jnp.tril(a, k))


def triu(a, k=0):
  if isinstance(a, JaxArray): a = a.value
  return JaxArray(jnp.triu(a, k))


def vander(x, N=None, increasing=False):
  if isinstance(x, JaxArray): x = x.value
  return JaxArray(jnp.vander(x, N=N, increasing=increasing))


def fill_diagonal(a, val):
  if isinstance(a, JaxArray): a = a.value
  assert a.ndim >= 2
  i, j = jnp.diag_indices(min(a.shape[-2:]))
  return JaxArray(a.at[..., i, j].set(val))


# indexing funcs
# --------------

tril_indices = jnp.tril_indices
triu_indices = jnp.triu_indices


def tril_indices_from(x, k=0):
  if isinstance(x, JaxArray): x = x.value
  return jnp.tril_indices_from(x, k=k)


def triu_indices_from(x, k=0):
  if isinstance(x, JaxArray): x = x.value
  return jnp.triu_indices_from(x, k=k)


def take(x, indices, axis=None, mode=None):
  if isinstance(x, JaxArray): x = x.value
  if isinstance(indices, JaxArray): indices = indices.value
  return JaxArray(jnp.take(x, indices=indices, axis=axis, mode=mode))


def select(condlist, choicelist, default=0):
  condlist = [c.value if isinstance(c, JaxArray) else c for c in condlist]
  choicelist = [c.value if isinstance(c, JaxArray) else c for c in choicelist]
  return JaxArray(jnp.select(condlist, choicelist, default=default))


# statistic funcs
# ---------------

def nanmin(x, axis=None, keepdims=None):
  if isinstance(x, JaxArray): x = x.value
  r = jnp.nanmin(x, axis=axis, keepdims=keepdims)
  return r if axis is None else JaxArray(r)


def nanmax(x, axis=None, keepdims=None):
  if isinstance(x, JaxArray): x = x.value
  r = jnp.nanmax(x, axis=axis, keepdims=keepdims)
  return r if axis is None else JaxArray(r)


def ptp(x, axis=None, keepdims=None):
  if isinstance(x, JaxArray): x = x.value
  r = jnp.ptp(x, axis=axis, keepdims=keepdims)
  return r if axis is None else JaxArray(r)


def percentile(a, q, axis=None, interpolation='linear', keepdims=False):
  if isinstance(a, JaxArray): a = a.value
  if isinstance(q, JaxArray): q = q.value
  r = jnp.percentile(a=a, q=q, axis=axis, interpolation=interpolation, keepdims=keepdims)
  return r if axis is None else JaxArray(r)


def nanpercentile(a, q, axis=None, interpolation='linear', keepdims=False):
  if isinstance(a, JaxArray): a = a.value
  if isinstance(q, JaxArray): q = q.value
  r = jnp.nanpercentile(a=a, q=q, axis=axis, interpolation=interpolation, keepdims=keepdims)
  return r if axis is None else JaxArray(r)


def quantile(a, q, axis=None, interpolation='linear', keepdims=False):
  if isinstance(a, JaxArray): a = a.value
  if isinstance(q, JaxArray): q = q.value
  r = jnp.quantile(a=a, q=q, axis=axis, interpolation=interpolation, keepdims=keepdims)
  return r if axis is None else JaxArray(r)


def nanquantile(a, q, axis=None, interpolation='linear', keepdims=False):
  if isinstance(a, JaxArray): a = a.value
  if isinstance(q, JaxArray): q = q.value
  r = jnp.nanquantile(a=a, q=q, axis=axis, interpolation=interpolation, keepdims=keepdims)
  return r if axis is None else JaxArray(r)


def average(a, axis=None, weights=None, returned=False):
  if isinstance(a, JaxArray): a = a.value
  if isinstance(weights, JaxArray): weights = weights.value
  r = jnp.average(a, axis=axis, weights=weights, returned=returned)
  return r if axis is None else JaxArray(r)


def mean(a, axis=None, dtype=None, keepdims=None, where=None):
  if isinstance(a, JaxArray): a = a.value
  r = jnp.mean(a, axis=axis, dtype=dtype, keepdims=keepdims, where=where)
  return r if axis is None else JaxArray(r)


def std(a, axis=None, dtype=None, ddof=0, keepdims=None, where=None):
  if isinstance(a, JaxArray): a = a.value
  r = jnp.std(a=a, axis=axis, dtype=dtype, ddof=ddof, keepdims=keepdims, where=where)
  return r if axis is None else JaxArray(r)


def var(a, axis=None, dtype=None, ddof=0, keepdims=None, where=None):
  if isinstance(a, JaxArray): a = a.value
  r = jnp.var(a, axis=axis, dtype=dtype, ddof=ddof, keepdims=keepdims, where=where)
  return r if axis is None else JaxArray(r)


def nanmedian(a, axis=None, keepdims=False):
  return nanquantile(a, 0.5, axis=axis, keepdims=keepdims, interpolation='midpoint')


def nanmean(a, axis=None, dtype=None, keepdims=None):
  if isinstance(a, JaxArray): a = a.value
  r = jnp.nanmean(a, axis=axis, dtype=dtype, keepdims=keepdims)
  return r if axis is None else JaxArray(r)


def nanstd(a, axis=None, dtype=None, ddof=0, keepdims=None, where=None):
  if isinstance(a, JaxArray): a = a.value
  r = jnp.nanstd(a=a, axis=axis, dtype=dtype, ddof=ddof, keepdims=keepdims, where=where)
  return r if axis is None else JaxArray(r)


def nanvar(a, axis=None, dtype=None, ddof=0, keepdims=None, where=None):
  if isinstance(a, JaxArray): a = a.value
  r = jnp.nanvar(a, axis=axis, dtype=dtype, ddof=ddof, keepdims=keepdims, where=where)
  return r if axis is None else JaxArray(r)


def corrcoef(x, y=None, rowvar=True):
  if isinstance(x, JaxArray): x = x.value
  if isinstance(y, JaxArray): y = y.value
  return JaxArray(jnp.corrcoef(x, y, rowvar))


def correlate(a, v, mode='valid'):
  if isinstance(a, JaxArray): a = a.value
  if isinstance(v, JaxArray): v = v.value
  return JaxArray(jnp.correlate(a, v, mode))


def cov(m, y=None, rowvar=True, bias=False, ddof=None, fweights=None, aweights=None):
  if isinstance(m, JaxArray): m = m.value
  if isinstance(y, JaxArray): y = y.value
  if isinstance(fweights, JaxArray): fweights = fweights.value
  if isinstance(aweights, JaxArray): aweights = aweights.value
  return JaxArray(jnp.cov(m, y=y, rowvar=rowvar, bias=bias, ddof=ddof,
                          fweights=fweights, aweights=aweights))


def histogram(a, bins=10, range=None, weights=None, density=None):
  if isinstance(a, JaxArray): a = a.value
  if isinstance(weights, JaxArray): weights = weights.value
  hist, bin_edges = jnp.histogram(a=a, bins=bins, range=range, weights=weights, density=density)
  return JaxArray(hist), JaxArray(bin_edges)


def bincount(x, weights=None, minlength=None):
  if isinstance(x, JaxArray): x = x.value
  if isinstance(weights, JaxArray): weights = weights.value
  return JaxArray(jnp.bincount(x, weights=weights, minlength=minlength))


def digitize(x, bins, right=False):
  if isinstance(x, JaxArray): x = x.value
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
  if isinstance(x1, JaxArray): x1 = x1.value
  if isinstance(x2, JaxArray): x2 = x2.value
  return JaxArray(jnp.dot(x1, x2))


def vdot(x1, x2):
  if isinstance(x1, JaxArray): x1 = x1.value
  if isinstance(x2, JaxArray): x2 = x2.value
  return JaxArray(jnp.vdot(x1, x2))


def inner(x1, x2):
  if isinstance(x1, JaxArray): x1 = x1.value
  if isinstance(x2, JaxArray): x2 = x2.value
  return JaxArray(jnp.inner(x1, x2))


def outer(x1, x2):
  if isinstance(x1, JaxArray): x1 = x1.value
  if isinstance(x2, JaxArray): x2 = x2.value
  return JaxArray(jnp.outer(x1, x2))


def kron(x1, x2):
  if isinstance(x1, JaxArray): x1 = x1.value
  if isinstance(x2, JaxArray): x2 = x2.value
  return JaxArray(jnp.kron(x1, x2))


def matmul(x1, x2):
  if isinstance(x1, JaxArray): x1 = x1.value
  if isinstance(x2, JaxArray): x2 = x2.value
  return JaxArray(jnp.matmul(x1, x2))


def trace(x, offset=0, axis1=0, axis2=1, dtype=None):
  if isinstance(x, JaxArray): x = x.value
  return JaxArray(jnp.trace(x, offset=offset, axis1=axis1, axis2=axis2, dtype=dtype))


# data types
# ----------

dtype = jnp.dtype
finfo = jnp.finfo
iinfo = jnp.iinfo

bool_ = jnp.bool_
uint8 = jnp.uint8
uint16 = jnp.uint16
uint32 = jnp.uint32
uint64 = jnp.uint64
int_ = jnp.int_
int8 = jnp.int8
int16 = jnp.int16
int32 = jnp.int32
int64 = jnp.int64
float_ = jnp.float_
float16 = jnp.float16
float32 = jnp.float32
float64 = jnp.float64
complex_ = jnp.complex_
complex64 = jnp.complex64
complex128 = jnp.complex128


def set_int_(int_type):
  global int_
  assert isinstance(int_type, type)
  int_ = int_type


def set_float_(float_type):
  global float_
  assert isinstance(float_type, type)
  float_ = float_type


def set_complex_(complex_type):
  global complex_
  assert isinstance(complex_type, type)
  complex_ = complex_type


# others
# ------
@copy_doc(np.take_along_axis)
def take_along_axis(a, indices, axis):
  if isinstance(a, JaxArray): a = a.value
  if isinstance(indices, JaxArray): indices = indices.value
  return JaxArray(jnp.take_along_axis(a, indices, axis))


@copy_doc(ops.clip_by_norm)
def clip_by_norm(t, clip_norm, axis=None):
  l2norm = sqrt(sum(t * t, axis=axis, keepdims=True))
  clip_values = t * clip_norm / maximum(l2norm, clip_norm)
  return clip_values

