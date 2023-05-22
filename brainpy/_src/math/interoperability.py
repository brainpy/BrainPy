# -*- coding: utf-8 -*-

import jax.numpy as jnp
import numpy as np

from .ndarray import Array


__all__ = [
  'as_device_array', 'as_jax', 'as_ndarray', 'as_numpy', 'as_variable',
]


def _as_jax_array_(obj):
  return obj.value if isinstance(obj, Array) else obj


def as_device_array(tensor, dtype=None):
  """Convert the input to a ``jax.numpy.DeviceArray``.

  Parameters
  ----------
  tensor: array_like
    Input data, in any form that can be converted to an array.  This
    includes lists, lists of tuples, tuples, tuples of tuples, tuples
    of lists, ArrayType.
  dtype: data-type, optional
    By default, the data-type is inferred from the input data.

  Returns
  -------
  out : ArrayType
    Array interpretation of `tensor`.  No copy is performed if the input
    is already an ndarray with matching dtype.
  """
  if isinstance(tensor, Array):
    return tensor.to_jax(dtype)
  elif isinstance(tensor, jnp.ndarray):
    return tensor if (dtype is None) else jnp.asarray(tensor, dtype=dtype)
  elif isinstance(tensor, np.ndarray):
    return jnp.asarray(tensor, dtype=dtype)
  else:
    return jnp.asarray(tensor, dtype=dtype)


as_jax = as_device_array


def as_ndarray(tensor, dtype=None):
  """Convert the input to a ``numpy.ndarray``.

  Parameters
  ----------
  tensor: array_like
    Input data, in any form that can be converted to an array.  This
    includes lists, lists of tuples, tuples, tuples of tuples, tuples
    of lists, ArrayType.
  dtype: data-type, optional
    By default, the data-type is inferred from the input data.

  Returns
  -------
  out : ndarray
    Array interpretation of `tensor`.  No copy is performed if the input
    is already an ndarray with matching dtype.
  """
  if isinstance(tensor, Array):
    return tensor.to_numpy(dtype=dtype)
  else:
    return np.asarray(tensor, dtype=dtype)


as_numpy = as_ndarray


def as_variable(tensor, dtype=None):
  """Convert the input to a ``brainpy.math.Variable``.

  Parameters
  ----------
  tensor: array_like
    Input data, in any form that can be converted to an array.  This
    includes lists, lists of tuples, tuples, tuples of tuples, tuples
    of lists, ArrayType.
  dtype: data-type, optional
    By default, the data-type is inferred from the input data.

  Returns
  -------
  out : ndarray
    Array interpretation of `tensor`.  No copy is performed if the input
    is already an ndarray with matching dtype.
  """
  from .object_transform.variables import Variable
  return Variable(tensor, dtype=dtype)
