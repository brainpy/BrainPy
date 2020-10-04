# -*- coding: utf-8 -*-

import numba as nb
import numpy as np

from npbrain import profile
from npbrain.helper import autojit


def _reload():
    global_vars = globals()

    # Return the cube-root of an array, element-wise.
    global_vars['cbrt'] = autojit(lambda x: np.power(x, 1. / 3))

    # First array elements raised to powers from second array, element-wise.
    global_vars['float_power'] = autojit(lambda x1, x2: np.power(np.float(x1), x2))

    # Compute the Heaviside step function.
    global_vars['heaviside'] = autojit(lambda x1, x2: np.where(x1 == 0, x2, np.where(x1 > 0, 1, 0)))

    @autojit
    def func(x, source, destination):
        shape = list(x.shape)
        s = shape.pop(source)
        shape.insert(destination, s)
        return np.transpose(x, tuple(shape))

    # Move axes of an array to new positions.
    global_vars['moveaxis'] = func

    # Split an array into multiple sub-arrays as views into ary.
    global_vars['split'] = None
    global_vars['dsplit'] = None
    global_vars['hsplit'] = None
    global_vars['vsplit'] = None
    global_vars['tile'] = None

    @autojit
    def func(x, axis1, axis2):
        shape = list(x.shape)
        s1 = shape[axis1]
        s2 = shape[axis2]
        shape[axis1] = s2
        shape[axis2] = s1
        return np.transpose(x, tuple(shape))

    # Interchange two axes of an array.
    global_vars['swapaxes'] = func

    # Rotate an array by 90 degrees in the plane specified by axes.
    global_vars['rot90'] = None

    # Return numbers spaced evenly on a log scale.
    global_vars['logspace'] = autojit(
        lambda start, stop, num=50, endpoint=True, base=10.0, dtype=None:
        np.power(base, np.linspace(start, stop, num=num, endpoint=endpoint)).astype(dtype))

    @nb.generated_jit(**profile.get_numba_profile())
    def squeeze(a, axis=None):
        if isinstance(axis, nb.types.NoneType):
            def squeeze_func(a, axis=None):
                shape = []
                for s in a.shape:
                    if s != 1:
                        shape.append(s)
                return np.reshape(a, shape)

            return squeeze_func

        elif isinstance(axis, nb.types.Integer):
            def squeeze_func(a, axis=None):
                shape = []
                for i, s in enumerate(a.shape):
                    if s != 1 or i != axis:
                        shape.append(s)
                return np.reshape(a, shape)

            return squeeze_func

        else:
            def squeeze_func(a, axis=None):
                shape = []
                for i, s in enumerate(a.shape):
                    if s != 1 or i not in axis:
                        shape.append(s)
                return np.reshape(a, shape)

            return squeeze_func

    # Remove single-dimensional entries from the shape of an array.
    global_vars['squeeze'] = func

    # Inner product of two arrays.
    global_vars['inner'] = autojit(lambda a, b: np.sum(a * b))

    # Evenly round to the given number of decimals.
    global_vars['around'] = np.round_

    # Round to nearest integer towards zero.
    global_vars['fix'] = autojit(lambda x: np.where(x >= 0, x, -np.floor(-x)))

    # Clip (limit) the values in an array.
    global_vars['clip'] = autojit(lambda x, x_min, x_max: np.clip(x, x_min, x_max))

    # Returns True if two arrays are element-wise equal within a tolerance.
    global_vars['allclose'] = autojit(
        lambda a, b, rtol=1e-05, atol=1e-08:
        np.all(np.absolute(a - b) <= (atol + rtol * np.absolute(b))))

    # Returns a boolean array where two arrays are element-wise equal within a tolerance.
    global_vars['isclose'] = autojit(
        lambda a, b, rtol=1e-05, atol=1e-08:
        np.absolute(a - b) <= (atol + rtol * np.absolute(b)))

    @nb.generated_jit(**profile.get_numba_profile())
    def average(a, axis=None, weights=None):
        if isinstance(weights, nb.types.NoneType):
            def func(a, axis=None, weights=None):
                return np.mean(a, axis)

            return func

        else:
            def func(a, axis=None, weights=None):
                return np.sum(a * weights, axis=axis) / sum(weights)

            return func

    # Compute the weighted average along the specified axis.
    global_vars['average'] = average

    # set random seed
    global_vars['seed'] = nb.njit(lambda a: np.random.seed(a))


_reload()
