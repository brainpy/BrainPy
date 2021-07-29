# -*- coding: utf-8 -*-

from brainpy.math.driver import *
from brainpy.math.numpy import *
from brainpy.math.function import *


# 1. numerical precision
# --------------------------

__dt = 0.1


def set_dt(dt):
  """Set the numerical integrator precision.

  Parameters
  ----------
  dt : float
      Numerical integration precision.
  """
  assert isinstance(dt, float), f'"dt" must a float, but we got {dt}'
  global __dt
  __dt = dt


def get_dt():
  """Get the numerical integrator precision.

  Returns
  -------
  dt : float
      Numerical integration precision.
  """
  return __dt


# 2. backend name
# --------------------------

BACKEND_NAME = 'numpy'


def get_backend_name():
  """Get the current backend name.

  Returns
  -------
  backend : str
      The name of the current backend name.
  """
  return BACKEND_NAME


# 3. mathematical operations
# --------------------------
__math_fs = [
  # Basics
  'real', 'imag', 'conj', 'conjugate', 'ndim', 'isreal', 'isscalar',

  # Arithmetic operations
  'add', 'reciprocal', 'negative', 'positive', 'multiply', 'divide',
  'power', 'subtract', 'true_divide', 'floor_divide', 'float_power',
  'fmod', 'mod', 'modf', 'divmod', 'remainder', 'abs',

  # Exponents and logarithms
  'exp', 'exp2', 'expm1', 'log', 'log10', 'log1p', 'log2',
  'logaddexp', 'logaddexp2',

  # Rational routines
  'lcm', 'gcd',

  # trigonometric functions
  'arccos', 'arccosh', 'arcsin', 'arcsinh', 'arctan', 'arctan2',
  'arctanh', 'cos', 'cosh', 'sin', 'sinc', 'sinh', 'tan', 'tanh',
  'deg2rad', 'hypot', 'rad2deg', 'degrees', 'radians',

  # Rounding
  'around', 'round_', 'rint', 'floor', 'ceil', 'trunc', 'fix',

  # Sums, products, differences, Reductions
  'prod', 'sum', 'diff', 'median', 'nancumprod', 'nancumsum',
  'nanprod', 'nansum', 'cumprod', 'cumsum', 'ediff1d', 'cross',
  'trapz',

  # floating_functions
  'isfinite', 'isinf', 'isnan', 'signbit', 'copysign', 'nextafter',
  'ldexp', 'frexp',

  # Miscellaneous
  'convolve', 'sqrt', 'cbrt', 'square', 'absolute', 'fabs', 'sign',
  'heaviside', 'maximum', 'minimum', 'fmax', 'fmin', 'interp', 'clip',
]

__binary_fs = [
  # https://numpy.org/doc/stable/reference/routines.bitwise.html

  # Elementwise bit operations
  'bitwise_and', 'bitwise_not', 'bitwise_or', 'bitwise_xor', 'invert',
  'left_shift', 'right_shift',
]

__logic_fs = [
  # https://numpy.org/doc/stable/reference/routines.logic.html

  # Comparison
  'equal', 'not_equal', 'greater', 'greater_equal', 'less', 'less_equal',
  'array_equal', 'isclose', 'allclose',

  # Logical operations
  'logical_and', 'logical_not', 'logical_or', 'logical_xor',

  # Truth value testing
  'all', 'any',
]

__array_manipulation_fs = [
  # https://numpy.org/doc/stable/reference/routines.array-manipulation.html
  # https://numpy.org/doc/stable/reference/routines.sort.html

  # Changing array shape
  'shape', 'size', 'reshape', 'ravel',

  # Transpose-like operations
  'moveaxis', 'transpose', 'swapaxes',

  # Joining arrays
  'concatenate', 'stack', 'vstack', 'hstack', 'dstack', 'column_stack',

  # Splitting arrays
  'split', 'dsplit', 'hsplit', 'vsplit',

  # Tiling arrays
  'tile', 'repeat',

  # Adding and removing elements
  'unique', 'append',

  # Rearranging elements
  'flip', 'fliplr', 'flipud', 'roll',

  # Changing number of dimensions
  'atleast_1d', 'atleast_2d', 'atleast_3d', 'expand_dims', 'squeeze',

  # Sorting
  'sort', 'argsort',

  # searching
  'argmax', 'argmin', 'argwhere', 'nonzero', 'flatnonzero', 'where',
  'searchsorted', 'extract',

  # counting
  'count_nonzero',

  # array intrinsic methods
  'max', 'min',

]

__array_creation_fs = [
  # https://numpy.org/doc/stable/reference/routines.array-creation.html

  'ndarray',

  # Ones and zeros
  'empty', 'empty_like', 'ones', 'ones_like', 'zeros', 'zeros_like',
  'full', 'full_like', 'eye', 'identity',

  # From existing data
  'array', 'asarray',

  # Numerical ranges
  'arange', 'linspace', 'logspace', 'meshgrid',

  # Building matrices
  'diag', 'tri', 'tril', 'triu', 'vander',
]

__indexing_fs = [
  # https://numpy.org/doc/stable/reference/routines.indexing.html

  # Generating index arrays
  'nonzero', 'where', 'tril_indices', 'tril_indices_from', 'triu_indices',
  'triu_indices_from',

  # Indexing-like operations
  'take', 'diag', 'select',
]

__statistic_fs = [
  # https://numpy.org/doc/stable/reference/routines.statistics.html

  # Order statistics
  'nanmin', 'nanmax', 'ptp', 'percentile', 'nanpercentile',
  'quantile', 'nanquantile',

  # Averages and variances
  'median', 'average', 'mean', 'std', 'var', 'nanmedian',
  'nanmean', 'nanstd', 'nanvar',

  # Correlating
  'corrcoef', 'correlate', 'cov',

  # Histograms
  'histogram', 'bincount', 'digitize',
]

__window_fs = [
  # https://numpy.org/doc/stable/reference/routines.window.html

  'bartlett', 'blackman', 'hamming', 'hanning', 'kaiser'
]

__constants = [
  # https://numpy.org/doc/stable/reference/constants.html

  'e', 'pi', 'inf'
]

__linear_algebra_fs = [
  # https://numpy.org/doc/stable/reference/routines.linalg.html
  'dot', 'vdot', 'inner', 'outer', 'kron', 'matmul', 'trace',
  # 'tensordot', 'einsum', 'einsum_path',
]

__data_types = [
  # https://numpy.org/doc/stable/reference/routines.dtype.html

  # functions
  'dtype', 'finfo', 'iinfo',

  # objects
  'bool_',
  'uint8', 'uint16', 'uint32', 'uint64',
  'int_', 'int8', 'int16', 'int32', 'int64',
  'float_', 'float16', 'float32', 'float64',
  'complex_', 'complex64', 'complex128',
]

__drivers = [
  'DriverForDS',
  'DriverForDiffInt',
]

__all = __math_fs + __binary_fs + __logic_fs + __array_manipulation_fs + \
        __array_creation_fs + __indexing_fs + __statistic_fs + __window_fs + \
        __constants + __linear_algebra_fs + __data_types + __drivers


# 4. backend setting
# ------------------


def use_backend(name, module=None):
  # check name
  if not isinstance(name, str):
    raise ValueError(f'"name" must be a str, but we got {type(name)}: {name}')

  # check module
  if module is None:
    if name == 'numpy':
      from brainpy.math import numpy as module
    elif name == 'jax':
      from brainpy.math import jax as module
    else:
      raise ValueError(f'Unknown backend "{name}", now we only '
                       f'support: numpy, jax.')
  else:
    from types import ModuleType
    if not isinstance(module, ModuleType):
      raise ValueError(f'"module" must be a module, but we got a '
                       f'type of {type(module)}: {module}')


  global_vars = globals()

  # replace operations
  essential_ops = set(__all)
  global_vars['BACKEND_NAME'] = name
  for key, value in module.__dict__.items():
    if key.startswith('_'):
      if key not in ['__name__', '__doc__', '__file__', '__path__']:
        continue
    global_vars[key] = value
    if key in essential_ops:
      essential_ops.remove(key)
  if len(essential_ops):
    raise ValueError(f'The following operations are essential for BrainPy backends:\n\n'
                     f'{essential_ops}\n\n'
                     f'But they are not provided in the {name} backend.\n'
                     f'Please provide their implementations in the corresponding module.')

