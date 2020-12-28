# -*- coding: utf-8 -*-

import math
import numpy

from . import numba_cpu


# Get functions in math
# ---------------------
_functions_in_math = []
for key in dir(math):
    if not key.startswith('__'):
        _functions_in_math.append(getattr(math, key))

# Get functions in NumPy
# ----------------------
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


def normal_like(x):
    return numpy.random.normal(size=numpy.shape(x))

