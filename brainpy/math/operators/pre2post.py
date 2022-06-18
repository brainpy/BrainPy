# -*- coding: utf-8 -*-

import jax.numpy as jnp
from typing import Union, Tuple

from brainpy.errors import MathError
from brainpy.math.numpy_ops import as_device_array
from .utils import _check_brainpylib
from .pre2syn import pre2syn
from .syn2post import syn2post_mean
from brainpy.types import Tensor

try:
  import brainpylib
except ModuleNotFoundError:
  brainpylib = None

__all__ = [
  # pre-to-post
  'pre2post_sum',
  'pre2post_prod',
  'pre2post_max',
  'pre2post_min',
  'pre2post_mean',

  # pre-to-post event operator
  'pre2post_event_sum',
  'pre2post_event_prod',
]


def _raise_pre_ids_is_none(pre_ids):
  if pre_ids is None:
    raise MathError(f'pre2post synaptic computation needs "pre_ids" '
                    f'when providing heterogeneous "pre_values" '
                    f'(brainpy.math.ndim(pre_values) != 0).')


def pre2post_event_sum(events: Tensor,
                       pre2post: Tuple[Tensor, Tensor],
                       post_num: int,
                       values: Union[float, Tensor] = 1.):
  """The pre-to-post synaptic computation with event-driven summation.

  When ``values`` is a scalar, this function is equivalent to

  .. highlight:: python
  .. code-block:: python

    post_val = np.zeros(post_num)
    post_ids, idnptr = pre2post
    for i in range(pre_num):
      if events[i]:
        for j in range(idnptr[i], idnptr[i+1]):
          post_val[post_ids[i]] += values

  When ``values`` is a vector (with the length of ``len(post_ids)``),
  this function is equivalent to

  .. highlight:: python
  .. code-block:: python

    post_val = np.zeros(post_num)

    post_ids, idnptr = pre2post
    for i in range(pre_num):
      if events[i]:
        for j in range(idnptr[i], idnptr[i+1]):
          post_val[post_ids[i]] += values[j]


  Parameters
  ----------
  events: Tensor
    The events, must be bool.
  pre2post: tuple of Tensor, tuple of Tensor
    A tuple contains the connection information of pre-to-post.
  post_num: int
    The number of post-synaptic group.
  values: float, Tensor
    The value to make summation.

  Returns
  -------
  out: JaxArray, jax.numpy.ndarray
    A tensor with the shape of ``post_num``.
  """
  _check_brainpylib(pre2post_event_sum.__name__)
  indices, idnptr = pre2post
  events = as_device_array(events)
  indices = as_device_array(indices)
  idnptr = as_device_array(idnptr)
  values = as_device_array(values)
  return brainpylib.event_sum(events, (indices, idnptr), post_num, values)


def pre2post_event_prod(events, pre2post, post_num, values=1.):
  """The pre-to-post synaptic computation with event-driven production.

  When ``values`` is a scalar, this function is equivalent to

  .. highlight:: python
  .. code-block:: python

    post_val = np.ones(post_num)
    post_ids, idnptr = pre2post
    for i in range(pre_num):
      if events[i]:
        for j in range(idnptr[i], idnptr[i+1]):
          post_val[post_ids[i]] *= values

  When ``values`` is a vector (with the length of ``len(post_ids)``),
  this function is equivalent to

  .. highlight:: python
  .. code-block:: python

    post_val = np.ones(post_num)

    post_ids, idnptr = pre2post
    for i in range(pre_num):
      if events[i]:
        for j in range(idnptr[i], idnptr[i+1]):
          post_val[post_ids[i]] *= values[j]


  Parameters
  ----------
  events: JaxArray, jax.numpy.ndarray, Variable
    The events, must be bool.
  pre2post: tuple of JaxArray, tuple of jax.numpy.ndarray
    A tuple contains the connection information of pre-to-post.
  post_num: int
    The number of post-synaptic group.
  values: float, JaxArray, jax.numpy.ndarray
    The value to make summation.

  Returns
  -------
  out: JaxArray, jax.numpy.ndarray
    A tensor with the shape of ``post_num``.
  """
  _check_brainpylib(pre2post_event_prod.__name__)
  indices, idnptr = pre2post
  events = as_device_array(events)
  indices = as_device_array(indices)
  idnptr = as_device_array(idnptr)
  values = as_device_array(values)
  return brainpylib.event_prod(events, (indices, idnptr), post_num, values)


def pre2post_sum(pre_values, post_num, post_ids, pre_ids=None):
  """The pre-to-post synaptic summation.

  This function is equivalent to:

  .. highlight:: python
  .. code-block:: python

     post_val = np.zeros(post_num)
     for i, j in zip(pre_ids, post_ids):
       post_val[j] += pre_values[pre_ids[i]]

  Parameters
  ----------
  pre_values: float, jax.numpy.ndarray, JaxArray, Variable
    The pre-synaptic values.
  post_ids: jax.numpy.ndarray, JaxArray
    The connected post-synaptic neuron ids.
  post_num: int
    Output dimension. The number of post-synaptic neurons.
  pre_ids: optional, jax.numpy.ndarray, JaxArray
    The connected pre-synaptic neuron ids.

  Returns
  -------
  post_val: jax.numpy.ndarray, JaxArray
    The value with the size of post-synaptic neurons.
  """
  out = jnp.zeros(post_num)
  pre_values = as_device_array(pre_values)
  post_ids = as_device_array(post_ids)
  if jnp.ndim(pre_values) != 0:
    _raise_pre_ids_is_none(pre_ids)
    pre_ids = as_device_array(pre_ids)
    pre_values = pre_values[pre_ids]
  return out.at[post_ids].add(pre_values)


def pre2post_prod(pre_values, post_num, post_ids, pre_ids=None):
  """The pre-to-post synaptic production.

  This function is equivalent to:

  .. highlight:: python
  .. code-block:: python

     post_val = np.zeros(post_num)
     for i, j in zip(pre_ids, post_ids):
       post_val[j] *= pre_values[pre_ids[i]]

  Parameters
  ----------
  pre_values: float, jax.numpy.ndarray, JaxArray, Variable
    The pre-synaptic values.
  pre_ids: jax.numpy.ndarray, JaxArray
    The connected pre-synaptic neuron ids.
  post_ids: jax.numpy.ndarray, JaxArray
    The connected post-synaptic neuron ids.
  post_num: int
    Output dimension. The number of post-synaptic neurons.

  Returns
  -------
  post_val: jax.numpy.ndarray, JaxArray
    The value with the size of post-synaptic neurons.
  """
  out = jnp.zeros(post_num)
  pre_values = as_device_array(pre_values)
  post_ids = as_device_array(post_ids)
  if jnp.ndim(pre_values) != 0:
    _raise_pre_ids_is_none(pre_ids)
    pre_ids = as_device_array(pre_ids)
    pre_values = pre_values[pre_ids]
  return out.at[post_ids].multiply(pre_values)


def pre2post_min(pre_values, post_num, post_ids, pre_ids=None):
  """The pre-to-post synaptic minimization.

  This function is equivalent to:

  .. highlight:: python
  .. code-block:: python

     post_val = np.zeros(post_num)
     for i, j in zip(pre_ids, post_ids):
       post_val[j] = np.minimum(post_val[j], pre_values[pre_ids[i]])

  Parameters
  ----------
  pre_values: float, jax.numpy.ndarray, JaxArray
    The pre-synaptic values.
  pre_ids: jax.numpy.ndarray, JaxArray
    The connected pre-synaptic neuron ids.
  post_ids: jax.numpy.ndarray, JaxArray
    The connected post-synaptic neuron ids.
  post_num: int
    Output dimension. The number of post-synaptic neurons.

  Returns
  -------
  post_val: jax.numpy.ndarray, JaxArray
    The value with the size of post-synaptic neurons.
  """
  out = jnp.zeros(post_num)
  pre_values = as_device_array(pre_values)
  post_ids = as_device_array(post_ids)
  if jnp.ndim(pre_values) != 0:
    _raise_pre_ids_is_none(pre_ids)
    pre_ids = as_device_array(pre_ids)
    pre_values = pre_values[pre_ids]
  return out.at[post_ids].min(pre_values)


def pre2post_max(pre_values, post_num, post_ids, pre_ids=None):
  """The pre-to-post synaptic maximization.

  This function is equivalent to:

  .. highlight:: python
  .. code-block:: python

     post_val = np.zeros(post_num)
     for i, j in zip(pre_ids, post_ids):
       post_val[j] = np.maximum(post_val[j], pre_values[pre_ids[i]])

  Parameters
  ----------
  pre_values: float, jax.numpy.ndarray, JaxArray, Variable
    The pre-synaptic values.
  pre_ids: jax.numpy.ndarray, JaxArray
    The connected pre-synaptic neuron ids.
  post_ids: jax.numpy.ndarray, JaxArray
    The connected post-synaptic neuron ids.
  post_num: int
    Output dimension. The number of post-synaptic neurons.

  Returns
  -------
  post_val: jax.numpy.ndarray, JaxArray
    The value with the size of post-synaptic neurons.
  """
  out = jnp.zeros(post_num)
  pre_values = as_device_array(pre_values)
  post_ids = as_device_array(post_ids)
  if jnp.ndim(pre_values) != 0:
    _raise_pre_ids_is_none(pre_ids)
    pre_ids = as_device_array(pre_ids)
    pre_values = pre_values[pre_ids]
  return out.at[post_ids].max(pre_values)


def pre2post_mean(pre_values, post_num, post_ids, pre_ids=None):
  """The pre-to-post synaptic mean computation.

  Parameters
  ----------
  pre_values: float, jax.numpy.ndarray, JaxArray, Variable
    The pre-synaptic values.
  pre_ids: jax.numpy.ndarray, JaxArray
    The connected pre-synaptic neuron ids.
  post_ids: jax.numpy.ndarray, JaxArray
    The connected post-synaptic neuron ids.
  post_num: int
    Output dimension. The number of post-synaptic neurons.

  Returns
  -------
  post_val: jax.numpy.ndarray, JaxArray
    The value with the size of post-synaptic neurons.
  """
  out = jnp.zeros(post_num)
  pre_values = as_device_array(pre_values)
  post_ids = as_device_array(post_ids)
  if jnp.ndim(pre_values) == 0:
    # return out.at[post_ids].set(pre_values)
    return out.at[jnp.unique(post_ids)].set(pre_values)
  else:
    _raise_pre_ids_is_none(pre_ids)
    pre_ids = as_device_array(pre_ids)
    pre_values = pre2syn(pre_values, pre_ids)
    return syn2post_mean(pre_values, post_ids, post_num)
