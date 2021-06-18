# -*- coding: utf-8 -*-

from brainpy.backend.math.numpy import *

_math_funcs = [
  # Basics
  'real', 'imag', 'conj', 'conjugate', 'ndim', 'isreal', 'isscalar',

  # Arithmetic operations
  'add', 'reciprocal', 'negative', 'positive', 'multiply', 'divide',
  'power', 'subtract', 'true_divide', 'floor_divide', 'float_power',
  'fmod', 'mod', 'modf', 'divmod', 'remainder', 'abs',

  # Exponents and logarithms
  'exp', 'exp2', 'expm1', 'log', 'log10', 'log1p', 'log2', 'logaddexp', 'logaddexp2',

  # Rational routines
  'lcm', 'gcd',

  # trigonometric functions
  'arccos', 'arccosh', 'arcsin', 'arcsinh', 'arctan', 'arctan2', 'arctanh', 'cos', 'cosh', 'sin',
  'sinc', 'sinh', 'tan', 'tanh', 'deg2rad', 'hypot', 'rad2deg', 'degrees', 'radians',

  # Rounding
  'around', 'round_', 'rint', 'floor', 'ceil', 'trunc', 'fix',

  # Sums, products, differences, Reductions
  'prod', 'sum', 'diff', 'median', 'nancumprod', 'nancumsum', 'nanprod', 'nansum',
  'cumprod', 'cumsum', 'ediff1d', 'cross', 'trapz',

  # floating_functions
  'isfinite', 'isinf', 'isnan', 'signbit', 'copysign', 'nextafter', 'ldexp', 'frexp',

  # Miscellaneous
  'convolve', 'sqrt', 'cbrt', 'square', 'absolute', 'fabs', 'sign',
  'heaviside', 'maximum', 'minimum', 'fmax', 'fmin', 'interp', 'clip',
]

_binary_funcs = [
  # https://numpy.org/doc/stable/reference/routines.bitwise.html

  # Elementwise bit operations
  'bitwise_and', 'bitwise_not', 'bitwise_or', 'bitwise_xor', 'invert', 'left_shift', 'right_shift',
]

_logic_funcs = [
  # https://numpy.org/doc/stable/reference/routines.logic.html

  # Comparison
  'equal', 'not_equal', 'greater', 'greater_equal', 'less', 'less_equal',
  'array_equal', 'isclose', 'allclose',

  # Logical operations
  'logical_and', 'logical_not', 'logical_or', 'logical_xor',

  # Truth value testing
  'all', 'any',
]

_array_manipulation = [
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
  'unique', 'delete', 'append',

  # Rearranging elements
  'flip', 'fliplr', 'flipud', 'roll',

  # Changing number of dimensions
  'atleast_1d', 'atleast_2d', 'atleast_3d', 'expand_dims', 'squeeze',

  # Sorting
  'sort', 'argsort',

  # searching
  'argmax', 'argmin', 'argwhere', 'nonzero', 'flatnonzero', 'where', 'searchsorted', 'extract',

  # counting
  'count_nonzero',

  # array intrinsic methods
  'max', 'min',

]

_array_creation = [
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

_indexing_funcs = [
  # https://numpy.org/doc/stable/reference/routines.indexing.html

  # Generating index arrays
  'nonzero', 'where', 'tril_indices', 'tril_indices_from', 'triu_indices', 'triu_indices_from',

  # Indexing-like operations
  'take', 'diag', 'select',

]

_statistic_funcs = [
  # https://numpy.org/doc/stable/reference/routines.statistics.html

  # Order statistics
  'nanmin', 'nanmax', 'ptp', 'percentile', 'nanpercentile', 'quantile', 'nanquantile',

  # Averages and variances
  'median', 'average', 'mean', 'std', 'var', 'nanmedian', 'nanmean', 'nanstd', 'nanvar',

  # Correlating
  'corrcoef', 'correlate', 'cov',

  # Histograms
  'histogram', 'bincount', 'digitize',
]

_window_funcs = [
  # https://numpy.org/doc/stable/reference/routines.window.html

  'bartlett', 'blackman', 'hamming', 'hanning', 'kaiser'
]

_constants = [
  # https://numpy.org/doc/stable/reference/constants.html

  'e', 'pi', 'inf'
]

_linear_algebra = [
  # https://numpy.org/doc/stable/reference/routines.linalg.html
  'dot', 'vdot', 'inner', 'outer', 'kron', 'matmul', 'trace',
  # 'tensordot', 'einsum', 'einsum_path',
]

_data_types = [
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

_all = _math_funcs + _binary_funcs + _logic_funcs + _array_manipulation + \
       _array_creation + _indexing_funcs + _statistic_funcs + _window_funcs + \
       _constants + _linear_algebra + _data_types


def use_backend(name):
  global_vars = globals()

  if name == 'numpy':
    from brainpy.backend.math import numpy as module

  elif name == 'numba':
    from brainpy.backend.math import numba as module

  elif name == 'jax':
    from brainpy.backend.math import jax as module

  else:
    raise ValueError(f'Unknown backend "{name}", now we only '
                     f'support: numpy, numba, jax.')

  for key, value in module.__dict__.items():
    if key.startswith('_'):
      if key not in ['__name__',  '__doc__',  '__file__', '__path__']:
        continue
    global_vars[key] = value
