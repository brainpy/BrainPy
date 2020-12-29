# -*- coding: utf-8 -*-

import math

import numba
import numpy
from numba.extending import overload

from .. import profile

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



@overload(numpy.cbrt)
def cbrt(x):
    return numpy.power(x, 1. / 3)


@overload(numpy.squeeze)
def squeeze(a, axis=None):
    if isinstance(axis, numba.types.NoneType):
        def squeeze_func(a, axis=None):
            shape = []
            for s in a.shape:
                if s != 1:
                    shape.append(s)
            return numpy.reshape(a, shape)

        return squeeze_func

    elif isinstance(axis, numba.types.Integer):
        def squeeze_func(a, axis=None):
            shape = []
            for i, s in enumerate(a.shape):
                if s != 1 or i != axis:
                    shape.append(s)
            return numpy.reshape(a, shape)

        return squeeze_func

    else:
        def squeeze_func(a, axis=None):
            shape = []
            for i, s in enumerate(a.shape):
                if s != 1 or i not in axis:
                    shape.append(s)
            return numpy.reshape(a, shape)

        return squeeze_func


@overload(numpy.float_power)
def float_power(x1, x2):
    return numpy.power(numpy.float(x1), x2)


@overload(numpy.heaviside)
def heaviside(x1, x2):
    return numpy.where(x1 == 0, x2, numpy.where(x1 > 0, 1, 0))


@overload(numpy.moveaxis)
def moveaxis(x, source, destination):
    shape = list(x.shape)
    s = shape.pop(source)
    shape.insert(destination, s)
    return numpy.transpose(x, tuple(shape))


@overload(numpy.swapaxes)
def swapaxes(x, axis1, axis2):
    shape = list(x.shape)
    s1 = shape[axis1]
    s2 = shape[axis2]
    shape[axis1] = s2
    shape[axis2] = s1
    return numpy.transpose(x, tuple(shape))


@overload(numpy.logspace)
def logspace(start, stop, num=50, endpoint=True, base=10.0, dtype=None):
    return numpy.power(base, numpy.linspace(start, stop, num=num, endpoint=endpoint)).astype(dtype)


@overload(numpy.inner)
def inner(a, b):
    return numpy.sum(a * b)


@overload(numpy.fix)
def fix(x):
    return numpy.where(x >= 0, x, -numpy.floor(-x))


@overload(numpy.clip)
def clip(x, x_min, x_max):
    x = numpy.maximum(x, x_min)
    x = numpy.minimum(x, x_max)
    return x


@overload(numpy.allclose)
def allclose(a, b, rtol=1e-05, atol=1e-08):
    return numpy.all(numpy.absolute(a - b) <= (atol + rtol * numpy.absolute(b)))


@overload(numpy.isclose)
def isclose(a, b, rtol=1e-05, atol=1e-08):
    return numpy.absolute(a - b) <= (atol + rtol * numpy.absolute(b))


@overload(numpy.average)
def average(a, axis=None, weights=None):
    if isinstance(weights, numba.types.NoneType):
        def func(a, axis=None, weights=None):
            return numpy.mean(a, axis)

        return func

    else:
        def func(a, axis=None, weights=None):
            return numpy.sum(a * weights, axis=axis) / sum(weights)

        return func


@numba.generated_jit(**profile.get_numba_profile())
def normal_like(x):
    if isinstance(x, (numba.types.Integer, numba.types.Float)):
        return lambda x: numpy.random.normal()
    else:
        return lambda x: numpy.random.normal(0., 1.0, x.shape)

