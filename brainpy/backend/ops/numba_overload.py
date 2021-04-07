# -*- coding: utf-8 -*-

import numpy

from brainpy import errors

try:
    import numba
except ModuleNotFoundError:
    raise errors.PackageMissingError(errors.backend_missing_msg.format(bk='numba'))


from numba.extending import overload


@overload(numpy.shape)
def shape_func(x):
    if isinstance(x, (numba.types.Integer, numba.types.Float)):
        def shape(x):
            return (1,)

        return shape
    else:
        return numpy.shape


@overload(numpy.cbrt)
def cbrt_func(x):
    def cbrt(x):
        return numpy.power(x, 1. / 3)

    return cbrt


@overload(numpy.squeeze)
def squeeze_func(a, axis=None):
    if isinstance(axis, numba.types.NoneType):
        def squeeze(a, axis=None):
            shape = []
            for s in a.shape:
                if s != 1:
                    shape.append(s)
            return numpy.reshape(a, shape)

        return squeeze

    elif isinstance(axis, numba.types.Integer):
        def squeeze(a, axis=None):
            shape = []
            for i, s in enumerate(a.shape):
                if s != 1 or i != axis:
                    shape.append(s)
            return numpy.reshape(a, shape)

        return squeeze

    else:
        def squeeze(a, axis=None):
            shape = []
            for i, s in enumerate(a.shape):
                if s != 1 or i not in axis:
                    shape.append(s)
            return numpy.reshape(a, shape)

        return squeeze


@overload(numpy.float_power)
def float_power_func(x1, x2):
    def float_power(x1, x2):
        return numpy.power(numpy.float(x1), x2)

    return float_power


@overload(numpy.heaviside)
def heaviside_func(x1, x2):
    def heaviside(x1, x2):
        return numpy.where(x1 == 0, x2, numpy.where(x1 > 0, 1, 0))

    return heaviside


@overload(numpy.moveaxis)
def moveaxis_func(x, source, destination):
    def moveaxis(x, source, destination):
        axes = list(range(len(x.shape)))
        if source < 0: source = axes[source]
        if destination < 0: destination = axes[destination]
        s = axes.pop(source)
        axes.insert(destination, s)
        return numpy.transpose(x, tuple(axes))

    return moveaxis


@overload(numpy.swapaxes)
def swapaxes_func(x, axis1, axis2):
    def swapaxes(x, axis1, axis2):
        shape = list(x.shape)
        s1 = shape[axis1]
        s2 = shape[axis2]
        shape[axis1] = s2
        shape[axis2] = s1
        return numpy.transpose(x, tuple(shape))

    return swapaxes


@overload(numpy.logspace)
def logspace_func(start, stop, num=50, endpoint=True, base=10.0, dtype=None):
    def logspace(start, stop, num=50, endpoint=True, base=10.0, dtype=None):
        return numpy.power(base, numpy.linspace(start, stop, num=num, endpoint=endpoint)).astype(dtype)

    return logspace


@overload(numpy.inner)
def inner_func(a, b):
    def inner(a, b):
        return numpy.sum(a * b)

    return inner


@overload(numpy.fix)
def fix_func(x):
    def fix(x):
        return numpy.where(x >= 0, x, -numpy.floor(-x))

    return fix


#
# @overload(numpy.clip)
# def clip_func(x, x_min, x_max):
#     def clip(x, x_min, x_max):
#         x = numpy.maximum(x, x_min)
#         x = numpy.minimum(x, x_max)
#         return x
#
#     return clip


@overload(numpy.allclose)
def allclose_func(a, b, rtol=1e-05, atol=1e-08):
    def allclose(a, b, rtol=1e-05, atol=1e-08):
        return numpy.all(numpy.absolute(a - b) <= (atol + rtol * numpy.absolute(b)))

    return allclose


@overload(numpy.isclose)
def isclose_func(a, b, rtol=1e-05, atol=1e-08):
    def isclose(a, b, rtol=1e-05, atol=1e-08):
        return numpy.absolute(a - b) <= (atol + rtol * numpy.absolute(b))

    return isclose


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
