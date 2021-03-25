# -*- coding: utf-8 -*-

"""
In this script, we establish the unified and standard
functions for computation backends.
"""

import numpy as np

__all__ = [
    # random function
    'normal',

    # arithmetic operation
    'sum',
    'exp',
    'matmul',

    # tensor creation
    'eye',
    'zeros',
    'ones',
    'arange',
    'as_tensor',

    # tensor manipulation
    'vstack',

    # others
    'shape',
    'reshape',
]


def normal(loc=0.0, scale=1.0, size=None):
    """The normal operation. We expect "normal" function will behave like "numpy.random.normal"

    Draw random samples from a normal (Gaussian) distribution.

    Parameters
    ----------
    loc : float or array_like of floats
        Mean ("centre") of the distribution.
    scale : float or array_like of floats
        Standard deviation (spread or "width") of the distribution. Must be
        non-negative.
    size : int or tuple of ints, optional
        Output shape.  If the given shape is, e.g., ``(m, n, k)``, then
        ``m * n * k`` samples are drawn.  If size is ``None`` (default),
        a single value is returned if ``loc`` and ``scale`` are both scalars.
        Otherwise, ``np.broadcast(loc, scale).size`` samples are drawn.

    Returns
    -------
    out : ndarray or scalar
        Drawn samples from the parameterized normal distribution.
    """
    pass


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


def exp(x):
    """The exp operation. We expect "exp" function will behave like "numpy.exp"

    Parameters
    ----------
    x : array_like
        Input values.

    Returns
    -------
    out : ndarray or scalar
        Output array, element-wise exponential of `x`.
        This is a scalar if `x` is a scalar.
    """
    pass


def eye(N, *args, **kwargs):
    """The eye operation. We expect "eye" function will behave like "numpy.eye".

    Return a 2-D array with ones on the diagonal and zeros elsewhere.

    Parameters
    ----------
    N : int
        Number of rows in the output.

    Returns
    -------
    I : tensor of shape (N,N)
        A tensor where all elements are equal to zero, except for the `k`-th
        diagonal, whose values are equal to one.
    """
    pass


def matmul(x1, x2, *args, **kwargs):
    """The matmul operation. We expect "matmul" function will behave like "numpy.matmul".

    Parameters
    ----------
    x1, x2 : array_like
        Input arrays, scalars not allowed.

    Returns
    -------
    y : tensor
        The matrix product of the inputs.
        This is a scalar only when both x1, x2 are 1-d vectors.
    """
    pass


def vstack(tup):
    """The vstack operation. We expect "vstack" function will behave like "numpy.vstack".

    Stack arrays in sequence vertically (row wise).

    Parameters
    ----------
    tup : sequence of tensors
        The arrays must have the same shape along all but the first axis.
        1-D arrays must have the same length.

    Returns
    -------
    stacked : tensor
        The tensor formed by stacking the given tensors, will be at least 2-D.

    Examples
    --------
    >>> a = np.array([1, 2, 3])
    >>> b = np.array([2, 3, 4])
    >>> np.vstack((a,b))
    array([[1, 2, 3],
           [2, 3, 4]])

    >>> a = np.array([[1], [2], [3]])
    >>> b = np.array([[2], [3], [4]])
    >>> np.vstack((a,b))
    array([[1],
           [2],
           [3],
           [2],
           [3],
           [4]])
    """
    pass


def zeros(shape, dtype=None):
    """The zeros operation. We expect "zeros" function will behave like "numpy.zeros".

    Return a new array of given shape and type, filled with zeros.

    Parameters
    ----------
    shape : shape : int or tuple of ints
            Shape of the new array, e.g., ``(2, 3)`` or ``2``.
    dtype : data-type, optional
        The desired data-type for the array, e.g., `int`.  Default is
        `float64`.

    Returns
    -------
    out : tensors
        Array of zeros with the given shape and dtype.
    """
    pass


def ones(shape, dtype=None):
    """The ones operation. We expect "ones" function will behave like "numpy.ones".

    Return a new array of given shape and type, filled with ones.

    Parameters
    ----------
    shape : shape : int or tuple of ints
            Shape of the new array, e.g., ``(2, 3)`` or ``2``.
    dtype : data-type, optional
        The desired data-type for the array, e.g., `int`.  Default is
        `float64`.

    Returns
    -------
    out : tensors
        Array of ones with the given shape and dtype.
    """
    pass


def arange(start=None, *args, **kwargs):
    """The arange operation. We expect "arange" function will behave like "numpy.arange".

    Return evenly spaced values within a given interval.

    Parameters
    ----------
    start : number, optional
        Start of interval.  The interval includes this value.  The default
        start value is 0.
    stop : number
        End of interval.  The interval does not include this value, except
        in some cases where `step` is not an integer and floating point
        round-off affects the length of `out`.
    step : number, optional
        Spacing between values.  For any output `out`, this is the distance
        between two adjacent values, ``out[i+1] - out[i]``.  The default
        step size is 1.  If `step` is specified as a position argument,
        `start` must also be given.
    dtype : dtype
        The type of the output array.  If `dtype` is not given, infer the data
        type from the other input arguments.

    Returns
    -------
    arange : ndarray
        Array of evenly spaced values.

        For floating point arguments, the length of the result is
        ``ceil((stop - start)/step)``.  Because of floating point overflow,
        this rule may result in the last element of `out` being greater
        than `stop`.
    """
    pass


def reshape(a, newshape):
    """The reshape operation. We expect "reshape" function will behave like "numpy.reshape".

    Gives a new shape to an array without changing its data.

    Parameters
    ----------
    a : array_like
        Array to be reshaped.
    newshape : int or tuple of ints
        The new shape should be compatible with the original shape. If
        an integer, then the result will be a 1-D array of that length.
        One shape dimension can be -1. In this case, the value is
        inferred from the length of the array and remaining dimensions.

    Returns
    -------
    reshaped_array : ndarray
        This will be a new view object if possible; otherwise, it will
        be a copy.  Note there is no guarantee of the *memory layout* (C- or
        Fortran- contiguous) of the returned array.
    """
    pass


def shape(a):
    """The shape operation. We expect "shape" function will behave like "numpy.shape".

    Parameters
    ----------
    a : array_like
        Input array.

    Returns
    -------
    shape : tuple of ints
        The elements of the shape tuple give the lengths of the
        corresponding array dimensions.
    """
    pass


def as_tensor(a, dtype=None):
    """The as_tensor operation. We expect "as_tensor" function will behave like "numpy.asarray".

    Parameters
    ----------
    a : array_like
        Input data, in any form that can be converted to an array.  This
        includes lists, lists of tuples, tuples, tuples of tuples, tuples
        of lists and ndarrays.
    dtype : data-type, optional
        By default, the data-type is inferred from the input data.

    Returns
    -------
    out : ndarray
        Array interpretation of `a`.  No copy is performed if the input
        is already an ndarray with matching dtype and order.  If `a` is a
        subclass of ndarray, a base class ndarray is returned.
    """
    pass
