# -*- coding: utf-8 -*-


import math

import numba
import numpy

from .. import profile


__all__ = [
    'func_in_numpy_or_math',
    'normal_like',
]


# Get functions in math
_functions_in_math = []
for key in dir(math):
    if not key.startswith('__'):
        _functions_in_math.append(getattr(math, key))


# Get functions in NumPy
_functions_in_numpy = []
for key in dir(numpy):
    if not key.startswith('__'):
        _functions_in_numpy.append(getattr(numpy, key))
for key in dir(numpy.random):
    if not key.startswith('__'):
        _functions_in_numpy.append(getattr(numpy.random, key))
for key in dir(numpy.linalg):
    if not key.startswith('__'):
        _functions_in_numpy.append(getattr(numpy.linalg, key))


def func_in_numpy_or_math(func):
    return func in _functions_in_math or func in _functions_in_numpy



@numba.generated_jit(**profile.get_numba_profile())
def normal_like(x):
    if isinstance(x, (numba.types.Integer, numba.types.Float)):
        return lambda x: numpy.random.normal()
    else:
        return lambda x: numpy.random.normal(0., 1.0, x.shape)

