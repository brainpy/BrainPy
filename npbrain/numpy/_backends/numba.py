# -*- coding: utf-8 -*-

import numba
import numpy

from npbrain import profile
from npbrain.tools import jit


def _reload():
    global_vars = globals()

    # Return the cube-root of an array, element-wise.
    global_vars['cbrt'] = jit(lambda x: numpy.power(x, 1. / 3))

    # First array elements raised to powers from second array, element-wise.
    global_vars['float_power'] = jit(lambda x1, x2: numpy.power(numpy.float(x1), x2))

    # Compute the Heaviside step function.
    global_vars['heaviside'] = jit(lambda x1, x2: numpy.where(x1 == 0, x2, numpy.where(x1 > 0, 1, 0)))

    @jit
    def func(x, source, destination):
        shape = list(x.shape)
        s = shape.pop(source)
        shape.insert(destination, s)
        return numpy.transpose(x, tuple(shape))

    # Move axes of an array to new positions.
    global_vars['moveaxis'] = func

    # Split an array into multiple sub-arrays as views into ary.
    global_vars['split'] = None
    global_vars['dsplit'] = None
    global_vars['hsplit'] = None
    global_vars['vsplit'] = None
    global_vars['tile'] = None

    @jit
    def func(x, axis1, axis2):
        shape = list(x.shape)
        s1 = shape[axis1]
        s2 = shape[axis2]
        shape[axis1] = s2
        shape[axis2] = s1
        return numpy.transpose(x, tuple(shape))

    # Interchange two axes of an array.
    global_vars['swapaxes'] = func

    # Rotate an array by 90 degrees in the plane specified by axes.
    global_vars['rot90'] = None

    # Return numbers spaced evenly on a log scale.
    global_vars['logspace'] = jit(
        lambda start, stop, num=50, endpoint=True, base=10.0, dtype=None:
        numpy.power(base, numpy.linspace(start, stop, num=num, endpoint=endpoint)).astype(dtype))

    @numba.generated_jit(**profile.get_numba_profile())
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

    # Remove single-dimensional entries from the shape of an array.
    global_vars['squeeze'] = func

    # Inner product of two arrays.
    global_vars['inner'] = jit(lambda a, b: numpy.sum(a * b))

    # Evenly round to the given number of decimals.
    global_vars['around'] = numpy.round_

    # Round to nearest integer towards zero.
    global_vars['fix'] = jit(lambda x: numpy.where(x >= 0, x, -numpy.floor(-x)))

    @jit
    def clip(x, x_min, x_max):
        x = numpy.maximum(x, x_min)
        x = numpy.minimum(x, x_max)
        return x

    # Clip (limit) the values in an array.
    global_vars['clip'] = clip

    # Returns True if two arrays are element-wise equal within a tolerance.
    global_vars['allclose'] = jit(
        lambda a, b, rtol=1e-05, atol=1e-08:
        numpy.all(numpy.absolute(a - b) <= (atol + rtol * numpy.absolute(b))))

    # Returns a boolean array where two arrays are element-wise equal within a tolerance.
    global_vars['isclose'] = jit(
        lambda a, b, rtol=1e-05, atol=1e-08:
        numpy.absolute(a - b) <= (atol + rtol * numpy.absolute(b)))

    @numba.generated_jit(**profile.get_numba_profile())
    def average(a, axis=None, weights=None):
        if isinstance(weights, numba.types.NoneType):
            def func(a, axis=None, weights=None):
                return numpy.mean(a, axis)

            return func

        else:
            def func(a, axis=None, weights=None):
                return numpy.sum(a * weights, axis=axis) / sum(weights)

            return func

    # Compute the weighted average along the specified axis.
    global_vars['average'] = average

    # set random seed
    global_vars['seed'] = numba.njit(lambda a: numpy.random.seed(a))

    @numba.generated_jit(**profile.get_numba_profile())
    def _normal_sample_(x):
        if isinstance(x, (numba.types.Integer, numba.types.Float)):
            return lambda x: numpy.random.normal()
        else:
            return lambda x: numpy.random.normal(0., 1.0, x.shape)

    global_vars['_normal_sample_'] = _normal_sample_


_reload()
