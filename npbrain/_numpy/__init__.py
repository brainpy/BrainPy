# -*- coding: utf-8 -*-

from importlib import import_module

import numpy

from . import random

math_operations = [
    'exp', 'exp2', 'expm1', 'log', 'log10', 'log1p', 'log2', 'matmul',
    'sign', 'sqrt', 'square', 'power', 'absolute', 'add', 'cbrt', 'conj',
    'conjugate', 'divide', 'divmod', 'fabs', 'float_power', 'floor_divide',
    'gcd', 'heaviside', 'lcm', 'logaddexp', 'logaddexp2', 'mod', 'multiply',
    'negative', 'positive', 'reciprocal', 'remainder', 'rint', 'subtract',
    'true_divide',

    # not defined in PyTorch
    # ----------------------
    # 'absolute', 'add', 'cbrt', 'conj', 'conjugate', 'divide', 'divmod', 'exp2',
    # 'expm1', 'fabs', 'float_power', 'floor_divide', 'gcd', 'heaviside', 'lcm',
    # 'logaddexp', 'logaddexp2', 'multiply', 'negative', 'positive', 'reciprocal',
    # 'remainder', 'rint', 'subtract', 'true_divide'
]

trigonometric_functions = [
    'arccos', 'arccosh', 'arcsin', 'arcsinh', 'arctan', 'arctan2', 'arctanh',
    'cos', 'cosh', 'sin', 'sinc', 'sinh', 'tan', 'tanh',
    'deg2rad', 'hypot', 'rad2deg', 'degrees', 'radians',
]

bitwise_functions = [
    'bitwise_and', 'bitwise_not', 'bitwise_or', 'bitwise_xor', 'invert',
    # 'left_shift', 'right_shift'
]

comparison_functions = [
    'equal', 'not_equal', 'greater', 'greater_equal', 'less', 'less_equal',
    'logical_and', 'logical_not', 'logical_or', 'logical_xor',
    'maximum', 'minimum', 'fmax', 'fmin',

    # not defined in PyTorch
    # ----------------------
    # 'fmax', 'fmin',
]

floating_functions = [
    'ceil', 'floor', 'fmod', 'isfinite', 'isinf', 'isnan', 'trunc',
    'fabs', 'nextafter', 'signbit',
    # 'copysign', 'frexp', 'isnat', 'ldexp', 'modf',
    # 'spacing',

    # not defined in PyTorch
    # ----------------------
    # 'fabs', 'nextafter', 'signbit',
]

array_manipulation = [
    'shape', 'reshape', 'ravel', 'moveaxis', 'transpose', 'reshape',
    'concatenate', 'stack', 'split', 'tile', 'repeat', 'flip',
    'swapaxes', 'vstack', 'hstack', 'dstack', 'column_stack',
    'dsplit', 'hsplit', 'vsplit', 'fliplr', 'flipud', 'roll', 'rot90',
    'append',
    # 'copyto', 'rollaxis', 'block', 'array_split',
    # 'delete', 'insert', 'resize', 'trim_zeros', 'unique',

    # not defined in PyTorch
    # ----------------------
    # 'swapaxes', 'vstack', 'hstack', 'dstack', 'column_stack', 'append',
    # 'dsplit', 'hsplit', 'vsplit', 'fliplr', 'flipud', 'roll', 'rot90',
]

array_creation = [
    'empty', 'empty_like', 'ones', 'ones_like', 'zeros', 'zeros_like', 'full', 'full_like',
    'eye', 'identity', 'array', 'asarray', 'arange', 'linspace', 'logspace', 'meshgrid',
    'geomspace', 'asanyarray', 'ascontiguousarray', 'copy',
    # 'mgrid', 'ogrid', 'asmatrix', 'frombuffer', 'fromfile', 'fromfunction',
    # 'fromiter', 'fromstring', 'loadtxt',

    # not defined in PyTorch
    # ----------------------
    # 'geomspace', 'asanyarray', 'ascontiguousarray',
]

constants = ['e', 'pi', 'inf', 'newaxis']  # 'nan'

linear_algebra = []

data_types = [
    'bool_',
    'uint8', 'uint16', 'uint32', 'uint64',
    'int_', 'int8', 'int16', 'int32', 'int64',
    'float_', 'float16', 'float32', 'float64',
    'complex_', 'complex64', 'complex128',

    # not defined in PyTorch
    # ----------------------
    # 'uint16', 'uint32', 'uint64',
]

__all1 = ['astype', 'clip', 'squeeze', 'expand_dims',
          'cumsum', 'pad', 'sign', 'sqrt',
          'where', 'take_along_axis']

__all__ = []
for __ops in math_operations + trigonometric_functions + bitwise_functions + \
             comparison_functions + floating_functions + array_manipulation + \
             array_creation + data_types + constants:
    __all__.append(getattr(numpy, __ops))


def _reload(backend):
    global_vars = globals()

    if backend in ['numpy', 'numba']:
        for __ops in math_operations + trigonometric_functions + bitwise_functions + \
                     comparison_functions + floating_functions + array_manipulation + \
                     array_creation + data_types + constants:
            global_vars[__ops] = getattr(numpy, __ops)

    elif backend == 'jax':
        # https://jax.readthedocs.io/en/latest/jax.numpy.html
        jnp = import_module('jax.numpy')

        for __ops in math_operations + trigonometric_functions + bitwise_functions + \
                     comparison_functions + floating_functions + array_manipulation + \
                     array_creation + data_types + constants:
            global_vars[__ops] = getattr(jnp, __ops)

    elif backend == 'torch':
        from npbrain._numpy._backends import pytorch

        for __ops in math_operations + trigonometric_functions + bitwise_functions + \
                     comparison_functions + floating_functions + array_manipulation + \
                     array_creation + constants:
            try:
                global_vars[__ops] = getattr(pytorch, __ops)
            except AttributeError:
                global_vars[__ops] = None
        for __ops in data_types:
            try:
                global_vars[__ops] = getattr(pytorch, __ops)
            except AttributeError:
                global_vars[__ops] = None
            global_vars['complex_'] = getattr(pytorch, 'complex128')
            global_vars['float_'] = getattr(pytorch, 'float64')
            global_vars['int_'] = getattr(pytorch, 'int64')

    elif backend == 'tensorflow':
        # https://www.tensorflow.org/api_docs/python/tf/experimental/numpy
        tnp = import_module('tensorflow.experimental.numpy')
        from npbrain._numpy._backends import tensorflow

        for __ops in math_operations + trigonometric_functions + bitwise_functions + \
                     comparison_functions + floating_functions + array_manipulation + \
                     array_creation + data_types + constants:
            try:
                ops = getattr(tnp, __ops)
            except AttributeError:
                ops = getattr(tensorflow, __ops)
            global_vars[__ops] = ops

    else:
        raise ValueError(f'Unknown backend device: {backend}')
