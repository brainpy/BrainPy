# -*- coding: utf-8 -*-

import numpy

from . import linalg
from . import random

# https://numpy.org/doc/stable/reference/routines.math.html
math_operations = [
    # Basics
    # --------
    'real', 'imag', 'conj', 'conjugate',  # 'angle',

    # Arithmetic operations
    # ----------------------
    'add', 'reciprocal', 'negative', 'positive', 'multiply', 'divide',
    'power', 'subtract', 'true_divide', 'floor_divide', 'float_power',
    'fmod', 'mod', 'modf', 'divmod', 'remainder',

    # Exponents and logarithms
    # -------------------------
    'exp', 'exp2', 'expm1', 'log', 'log10', 'log1p', 'log2', 'logaddexp', 'logaddexp2',

    # Rational routines
    # -------------------------
    'lcm', 'gcd',

    # trigonometric functions
    # --------------------------
    'arccos', 'arccosh', 'arcsin', 'arcsinh', 'arctan', 'arctan2', 'arctanh', 'cos', 'cosh', 'sin',
    'sinc', 'sinh', 'tan', 'tanh', 'deg2rad', 'hypot', 'rad2deg', 'degrees', 'radians',  # 'unwrap'

    # Rounding
    # --------
    'around', 'round_', 'rint', 'floor', 'ceil', 'trunc', 'fix',

    # Sums, products, differences, Reductions
    # --------------------------------------------
    'prod', 'sum', 'diff', 'median', 'nancumprod', 'nancumsum', 'nanprod', 'nansum',
    'cumprod', 'cumsum', 'ediff1d', 'cross', 'trapz',  # 'gradient',

    # floating_functions
    # -------------------
    'isfinite', 'isinf', 'isnan', 'signbit', 'copysign', 'nextafter', 'modf',
    'ldexp', 'frexp', 'floor', 'spacing',

    # Miscellaneous
    # --------------
    'convolve', 'sqrt', 'cbrt', 'square', 'absolute', 'fabs', 'sign',
    'heaviside', 'maximum', 'minimum', 'fmax', 'fmin', 'interp', 'clip',
]

# https://numpy.org/doc/stable/reference/routines.bitwise.html
binary_operations = [
    # Elementwise bit operations
    # ----------------------------
    'bitwise_and', 'bitwise_not', 'bitwise_or', 'bitwise_xor',
    'invert', 'left_shift', 'right_shift',
]

# https://numpy.org/doc/stable/reference/routines.logic.html
logic_functions = [
    # Comparison
    # --------------
    'equal', 'not_equal', 'greater', 'greater_equal', 'less', 'less_equal',
    'array_equal', 'isclose', 'allclose',

    # Logical operations
    # ---------------------
    'logical_and', 'logical_not', 'logical_or', 'logical_xor',
    'maximum', 'minimum', 'fmax', 'fmin',

    # Truth value testing
    # ----------------------
    'all', 'any',
]

# https://numpy.org/doc/stable/reference/routines.array-manipulation.html
# https://numpy.org/doc/stable/reference/routines.sort.html
array_manipulation = [
    # Changing array shape
    # ---------------------
    'shape', 'reshape', 'ravel',

    # Transpose-like operations
    # --------------------------
    'moveaxis', 'transpose', 'swapaxes',

    # Joining arrays
    # ---------------
    'concatenate', 'stack', 'vstack', 'hstack', 'dstack', 'column_stack',

    # Splitting arrays
    # -----------------
    'split', 'dsplit', 'hsplit', 'vsplit',

    # Tiling arrays
    # ---------------
    'tile', 'repeat',

    # Adding and removing elements
    # ------------------------------
    'unique', 'delete', 'append',

    # Rearranging elements
    # ---------------------
    'flip', 'fliplr', 'flipud', 'roll',

    # Changing number of dimensions
    # -------------------------------
    'atleast_1d', 'atleast_2d', 'atleast_3d', 'expand_dims', 'squeeze',

    # Sorting
    # -----------
    'sort', 'argsort', 'partition',

    # searching
    # ------------
    'argmax', 'argmin', 'argwhere', 'nonzero', 'flatnonzero', 'where',
    'searchsorted', 'extract',  # 'nanargmax', 'nanargmin',

    # counting
    # ------------
    'count_nonzero',

    # array intrinsic methods
    # -------------------------
    'max', 'min', 'astype', 'flatten', 'item', 'itemset', 'view',

    # padding
    # ---------
    # 'pad'
]

# https://numpy.org/doc/stable/reference/routines.array-creation.html
array_creation = [
    # Ones and zeros
    # ---------------
    'empty', 'empty_like', 'ones', 'ones_like', 'zeros', 'zeros_like',
    'full', 'full_like', 'eye', 'identity',

    # From existing data
    # --------------------
    'array', 'asarray',

    # Numerical ranges
    # ------------------
    'arange', 'linspace', 'logspace', 'meshgrid', 'copy',

    # Building matrices
    # -------------------
    'diag', 'tri', 'tril', 'triu', 'vander',
]

# https://numpy.org/doc/stable/reference/routines.indexing.html
indexing_routines = [
    # Generating index arrays
    # -------------------------
    'nonzero', 'where', 'tril_indices', 'tril_indices_from',
    'triu_indices', 'triu_indices_from',

    # Indexing-like operations
    # --------------------------
    'take', 'diag', 'select',

    # Iterating over arrays
    # -----------------------
    'nditer', 'ndenumerate', 'ndindex',
]

# https://numpy.org/doc/stable/reference/routines.statistics.html
statistic_functions = [
    # Order statistics
    # ------------------
    'nanmin', 'nanmax', 'ptp', 'percentile', 'nanpercentile', 'quantile', 'nanquantile',
    # 'amin', 'amax',

    # Averages and variances
    # ----------------------
    'median', 'average', 'mean', 'std', 'var', 'nanmedian', 'nanmean', 'nanstd', 'nanvar',

    # Correlating
    # ----------------------
    'corrcoef', 'correlate', 'cov',

    # Histograms
    # ----------------------
    'histogram', 'bincount', 'digitize',  # 'histogram2d', 'histogramdd', 'histogram_bin_edges'
]

# https://numpy.org/doc/stable/reference/routines.window.html
window_functions = ['bartlett', 'blackman', 'hamming', 'hanning', 'kaiser']

# https://numpy.org/doc/stable/reference/constants.html
constants = ['e', 'pi', 'inf', 'nan', 'newaxis', 'euler_gamma']

# https://numpy.org/doc/stable/reference/routines.linalg.html
linear_algebra = [
    'dot', 'vdot', 'inner', 'outer', 'kron', 'matmul', 'trace',
    # 'tensordot', 'einsum', 'einsum_path',
]

# https://numpy.org/doc/stable/reference/routines.dtype.html
data_type_routines = [
    # functions
    # ---------
    'dtype', 'finfo', 'iinfo', 'MachAr',

    # objects
    # --------
    'bool_',
    'uint8', 'uint16', 'uint32', 'uint64',
    'int_', 'int8', 'int16', 'int32', 'int64',
    'float_', 'float16', 'float32', 'float64',
    'complex_', 'complex64', 'complex128',
]

_all_funcs_and_objs = math_operations + binary_operations + logic_functions + array_manipulation + \
                      array_creation + indexing_routines + statistic_functions + window_functions + \
                      constants + linear_algebra + data_type_routines

__all__ = []
for __ops in _all_funcs_and_objs:
    __all__.append(getattr(numpy, __ops))


def _reload(backend):
    global_vars = globals()

    if backend == 'numpy':
        for __ops in _all_funcs_and_objs:
            global_vars[__ops] = getattr(numpy, __ops)

    elif backend == 'numba':
        from ._backends import numba
        for __ops in _all_funcs_and_objs:
            if hasattr(numba, __ops):
                global_vars[__ops] = getattr(numba, __ops)
            else:
                global_vars[__ops] = getattr(numpy, __ops)

    else:
        raise ValueError(f'Unknown backend device: {backend}')
