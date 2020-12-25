# -*- coding: utf-8 -*-

import numpy

from . import numba_cpu


def normal_like(x):
    return numpy.random.normal(size=numpy.shape(x))


def func_by_name(name):
    """Get backend function by its name.

    Parameters
    ----------
    name : str
        Function name.

    Returns
    -------
    func : callable
        Numpy function.
    """
    if hasattr(numpy.random, name):
        return getattr(numpy.random, name)
    elif hasattr(numpy.linalg, name):
        return getattr(numpy.linalg, name)
    elif hasattr(numpy, name):
        return getattr(numpy, name)
    else:
        return None
