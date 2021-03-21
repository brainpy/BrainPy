# -*- coding: utf-8 -*-

"""
In this script, we establish the unified and standard
functions for computation backends.
"""

import numpy as np


def sum(tensor, axis=None):
    """The sum operation. We expect "sum" function will behave like "numpy.sum"

    Parameters
    ----------
    tensor : array_like
        The data to sum.
    axis : None or int or tuple of ints, optional
        Axis or axes along which a sum is performed.  The default,
        axis=None, will sum all of the elements of the input array.  If
        axis is negative it counts from the last to the first axis. 
        
    Returns
    -------
    sum_along_axis : ndarray
        An array with the same shape as `a`, with the specified
        axis removed.   If `a` is a 0-d array, or if `axis` is None, a scalar
        is returned.  If an output array is specified, a reference to
        `out` is returned.
    
    Examples
    --------
    >>> sum([0.5, 1.5])
    2.0
    >>> sum([0.5, 0.7, 0.2, 1.5], dtype=np.int32)
    1
    >>> sum([[0, 1], [0, 5]])
    6
    >>> sum([[0, 1], [0, 5]], axis=0)
    array([0, 6])
    >>> sum([[0, 1], [0, 5]], axis=1)
    array([1, 5])
    >>> sum([[0, 1], [np.nan, 5]], where=[False, True], axis=1)
    array([1., 5.])

    If the accumulator is too small, overflow occurs:

    >>> np.ones(128, dtype=np.int8).sum(dtype=np.int8)
    -128

    You can also start the sum with a value other than zero:

    >>> sum([10], initial=5)
    15
    
    """
    pass


