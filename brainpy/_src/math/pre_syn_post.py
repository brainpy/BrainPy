# -*- coding: utf-8 -*-


import jax.numpy as jnp
from jax import vmap, jit, ops as jops

from brainpy._src.math.interoperability import as_jax
from brainpy._src.math import event
from brainpy.errors import MathError
from brainpy._src import tools

__all__ = [
  # pre-to-post
  'pre2post_sum',
  'pre2post_prod',
  'pre2post_max',
  'pre2post_min',
  'pre2post_mean',

  # pre-to-post event operator
  'pre2post_event_sum',
  'pre2post_csr_event_sum',
  'pre2post_coo_event_sum',

  # pre-to-syn
  'pre2syn',

  # syn-to-post
  'syn2post_sum', 'syn2post',
  'syn2post_prod',
  'syn2post_max',
  'syn2post_min',
  'syn2post_mean',
  'syn2post_softmax',
]


def _raise_pre_ids_is_none(pre_ids):
  if pre_ids is None:
    raise MathError(f'pre2post synaptic computation needs "pre_ids" '
                    f'when providing heterogeneous "pre_values" '
                    f'(brainpy.math.ndim(pre_values) != 0).')


def pre2post_event_sum(events,
                       pre2post,
                       post_num: int,
                       values=1.):
  """The pre-to-post event-driven synaptic summation with `CSR` synapse structure.

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
  events: ArrayType
    The events, must be bool.
  pre2post: tuple of ArrayType, tuple of ArrayType
    A tuple contains the connection information of pre-to-post.
  post_num: int
    The number of post-synaptic group.
  values: float, ArrayType
    The value to make summation.

  Returns
  -------
  out: ArrayType
    A tensor with the shape of ``post_num``.
  """
  indices, idnptr = pre2post
  events = as_jax(events)
  indices = as_jax(indices)
  idnptr = as_jax(idnptr)
  values = as_jax(values)
  return event.csrmv(values, indices, idnptr, events,
                     shape=(events.shape[0], post_num),
                     transpose=True)


pre2post_csr_event_sum = pre2post_event_sum


def pre2post_coo_event_sum(events,
                           pre_ids,
                           post_ids,
                           post_num: int,
                           values=1.):
  """The pre-to-post synaptic computation with event-driven summation.

  Parameters
  ----------
  events: ArrayType
    The events, must be bool.
  pre_ids: ArrayType
    Pre-synaptic ids.
  post_ids: ArrayType
    Post-synaptic ids.
  post_num: int
    The number of post-synaptic group.
  values: float, ArrayType
    The value to make summation.

  Returns
  -------
  out: ArrayType
    A tensor with the shape of ``post_num``.
  """
  events = as_jax(events)
  post_ids = as_jax(post_ids)
  pre_ids = as_jax(pre_ids)
  values = as_jax(values)
  bl = tools.import_brainpylib()
  return bl.compat.coo_event_sum(events, pre_ids, post_ids, post_num, values)


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
  pre_values: float, ArrayType
    The pre-synaptic values.
  post_ids: ArrayType
    The connected post-synaptic neuron ids.
  post_num: int
    Output dimension. The number of post-synaptic neurons.
  pre_ids: optional, ArrayType
    The connected pre-synaptic neuron ids.

  Returns
  -------
  post_val: ArrayType
    The value with the size of post-synaptic neurons.
  """
  out = jnp.zeros(post_num)
  pre_values = as_jax(pre_values)
  post_ids = as_jax(post_ids)
  if jnp.ndim(pre_values) != 0:
    _raise_pre_ids_is_none(pre_ids)
    pre_ids = as_jax(pre_ids)
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
  pre_values: float, ArrayType
    The pre-synaptic values.
  pre_ids: ArrayType
    The connected pre-synaptic neuron ids.
  post_ids: ArrayType
    The connected post-synaptic neuron ids.
  post_num: int
    Output dimension. The number of post-synaptic neurons.

  Returns
  -------
  post_val: ArrayType
    The value with the size of post-synaptic neurons.
  """
  out = jnp.zeros(post_num)
  pre_values = as_jax(pre_values)
  post_ids = as_jax(post_ids)
  if jnp.ndim(pre_values) != 0:
    _raise_pre_ids_is_none(pre_ids)
    pre_ids = as_jax(pre_ids)
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
  pre_values: float, ArrayType
    The pre-synaptic values.
  pre_ids: ArrayType
    The connected pre-synaptic neuron ids.
  post_ids: ArrayType
    The connected post-synaptic neuron ids.
  post_num: int
    Output dimension. The number of post-synaptic neurons.

  Returns
  -------
  post_val: ArrayType
    The value with the size of post-synaptic neurons.
  """
  out = jnp.zeros(post_num)
  pre_values = as_jax(pre_values)
  post_ids = as_jax(post_ids)
  if jnp.ndim(pre_values) != 0:
    _raise_pre_ids_is_none(pre_ids)
    pre_ids = as_jax(pre_ids)
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
  pre_values: float, ArrayType
    The pre-synaptic values.
  pre_ids: ArrayType
    The connected pre-synaptic neuron ids.
  post_ids: ArrayType
    The connected post-synaptic neuron ids.
  post_num: int
    Output dimension. The number of post-synaptic neurons.

  Returns
  -------
  post_val: ArrayType
    The value with the size of post-synaptic neurons.
  """
  out = jnp.zeros(post_num)
  pre_values = as_jax(pre_values)
  post_ids = as_jax(post_ids)
  if jnp.ndim(pre_values) != 0:
    _raise_pre_ids_is_none(pre_ids)
    pre_ids = as_jax(pre_ids)
    pre_values = pre_values[pre_ids]
  return out.at[post_ids].max(pre_values)


def pre2post_mean(pre_values, post_num, post_ids, pre_ids=None):
  """The pre-to-post synaptic mean computation.

  Parameters
  ----------
  pre_values: float, ArrayType
    The pre-synaptic values.
  pre_ids: ArrayType
    The connected pre-synaptic neuron ids.
  post_ids: ArrayType
    The connected post-synaptic neuron ids.
  post_num: int
    Output dimension. The number of post-synaptic neurons.

  Returns
  -------
  post_val: ArrayType
    The value with the size of post-synaptic neurons.
  """
  out = jnp.zeros(post_num)
  pre_values = as_jax(pre_values)
  post_ids = as_jax(post_ids)
  if jnp.ndim(pre_values) == 0:
    return out.at[post_ids].set(pre_values)
    # return out.at[jnp.unique(post_ids)].set(pre_values)
  else:
    _raise_pre_ids_is_none(pre_ids)
    pre_ids = as_jax(pre_ids)
    pre_values = pre2syn(pre_values, pre_ids)
    return syn2post_mean(pre_values, post_ids, post_num)


_pre2syn = vmap(lambda pre_id, pre_vs: pre_vs[pre_id], in_axes=(0, None))


def pre2syn(pre_values, pre_ids):
  """The pre-to-syn computation.

  Change the pre-synaptic data to the data with the dimension of synapses.

  This function is equivalent to:

  .. highlight:: python
  .. code-block:: python

    syn_val = np.zeros(len(pre_ids))
    for syn_i, pre_i in enumerate(pre_ids):
      syn_val[i] = pre_values[pre_i]

  Parameters
  ----------
  pre_values: float, ArrayType
    The pre-synaptic value.
  pre_ids: ArrayType
    The pre-synaptic neuron index.

  Returns
  -------
  syn_val: ArrayType
    The synaptic value.
  """
  pre_values = as_jax(pre_values)
  pre_ids = as_jax(pre_ids)
  if jnp.ndim(pre_values) == 0:
    return jnp.ones(len(pre_ids), dtype=pre_values.dtype) * pre_values
  else:
    return _pre2syn(pre_ids, pre_values)


_jit_seg_sum = jit(jops.segment_sum, static_argnums=(2, 3))
_jit_seg_prod = jit(jops.segment_prod, static_argnums=(2, 3))
_jit_seg_max = jit(jops.segment_max, static_argnums=(2, 3))
_jit_seg_min = jit(jops.segment_min, static_argnums=(2, 3))


def syn2post_sum(syn_values, post_ids, post_num: int, indices_are_sorted=False):
  """The syn-to-post summation computation.

  This function is equivalent to:

  .. highlight:: python
  .. code-block:: python

    post_val = np.zeros(post_num)
    for syn_i, post_i in enumerate(post_ids):
      post_val[post_i] += syn_values[syn_i]

  Parameters
  ----------
  syn_values: ArrayType
    The synaptic values.
  post_ids: ArrayType
    The post-synaptic neuron ids.
  post_num: int
    The number of the post-synaptic neurons.

  Returns
  -------
  post_val: ArrayType
    The post-synaptic value.
  """
  post_ids = as_jax(post_ids)
  syn_values = as_jax(syn_values)
  if syn_values.dtype == jnp.bool_:
    syn_values = jnp.asarray(syn_values, dtype=jnp.int32)
  return _jit_seg_sum(syn_values, post_ids, post_num, indices_are_sorted)


syn2post = syn2post_sum


def syn2post_prod(syn_values, post_ids, post_num: int, indices_are_sorted=False):
  """The syn-to-post product computation.

  This function is equivalent to:

  .. highlight:: python
  .. code-block:: python

    post_val = np.zeros(post_num)
    for syn_i, post_i in enumerate(post_ids):
      post_val[post_i] *= syn_values[syn_i]

  Parameters
  ----------
  syn_values: ArrayType
    The synaptic values.
  post_ids: ArrayType
    The post-synaptic neuron ids. If ``post_ids`` is generated by
    ``brainpy.conn.TwoEndConnector``, then it has sorted indices.
    Otherwise, this function cannot guarantee indices are sorted.
    You's better set ``indices_are_sorted=False``.
  post_num: int
    The number of the post-synaptic neurons.
  indices_are_sorted: whether ``post_ids`` is known to be sorted.

  Returns
  -------
  post_val: ArrayType
    The post-synaptic value.
  """
  post_ids = as_jax(post_ids)
  syn_values = as_jax(syn_values)
  if syn_values.dtype == jnp.bool_:
    syn_values = jnp.asarray(syn_values, dtype=jnp.int32)
  return _jit_seg_prod(syn_values, post_ids, post_num, indices_are_sorted)


def syn2post_max(syn_values, post_ids, post_num: int, indices_are_sorted=False):
  """The syn-to-post maximum computation.

  This function is equivalent to:

  .. highlight:: python
  .. code-block:: python

    post_val = np.zeros(post_num)
    for syn_i, post_i in enumerate(post_ids):
      post_val[post_i] = np.maximum(post_val[post_i], syn_values[syn_i])

  Parameters
  ----------
  syn_values: ArrayType
    The synaptic values.
  post_ids: ArrayType
    The post-synaptic neuron ids. If ``post_ids`` is generated by
    ``brainpy.conn.TwoEndConnector``, then it has sorted indices.
    Otherwise, this function cannot guarantee indices are sorted.
    You's better set ``indices_are_sorted=False``.
  post_num: int
    The number of the post-synaptic neurons.
  indices_are_sorted: whether ``post_ids`` is known to be sorted.

  Returns
  -------
  post_val: ArrayType
    The post-synaptic value.
  """
  post_ids = as_jax(post_ids)
  syn_values = as_jax(syn_values)
  if syn_values.dtype == jnp.bool_:
    syn_values = jnp.asarray(syn_values, dtype=jnp.int32)
  return _jit_seg_max(syn_values, post_ids, post_num, indices_are_sorted)


def syn2post_min(syn_values, post_ids, post_num: int, indices_are_sorted=False):
  """The syn-to-post minimization computation.

  This function is equivalent to:

  .. highlight:: python
  .. code-block:: python

    post_val = np.zeros(post_num)
    for syn_i, post_i in enumerate(post_ids):
      post_val[post_i] = np.minimum(post_val[post_i], syn_values[syn_i])

  Parameters
  ----------
  syn_values: ArrayType
    The synaptic values.
  post_ids: ArrayType
    The post-synaptic neuron ids. If ``post_ids`` is generated by
    ``brainpy.conn.TwoEndConnector``, then it has sorted indices.
    Otherwise, this function cannot guarantee indices are sorted.
    You's better set ``indices_are_sorted=False``.
  post_num: int
    The number of the post-synaptic neurons.
  indices_are_sorted: whether ``post_ids`` is known to be sorted.

  Returns
  -------
  post_val: ArrayType
    The post-synaptic value.
  """
  post_ids = as_jax(post_ids)
  syn_values = as_jax(syn_values)
  if syn_values.dtype == jnp.bool_:
    syn_values = jnp.asarray(syn_values, dtype=jnp.int32)
  return _jit_seg_min(syn_values, post_ids, post_num, indices_are_sorted)


def syn2post_mean(syn_values, post_ids, post_num: int, indices_are_sorted=False):
  """The syn-to-post mean computation.

  Parameters
  ----------
  syn_values: ArrayType
    The synaptic values.
  post_ids: ArrayType
    The post-synaptic neuron ids. If ``post_ids`` is generated by
    ``brainpy.conn.TwoEndConnector``, then it has sorted indices.
    Otherwise, this function cannot guarantee indices are sorted.
    You's better set ``indices_are_sorted=False``.
  post_num: int
    The number of the post-synaptic neurons.
  indices_are_sorted: whether ``post_ids`` is known to be sorted.

  Returns
  -------
  post_val: ArrayType
    The post-synaptic value.
  """
  post_ids = as_jax(post_ids)
  syn_values = as_jax(syn_values)
  if syn_values.dtype == jnp.bool_:
    syn_values = jnp.asarray(syn_values, dtype=jnp.int32)
  nominator = _jit_seg_sum(syn_values, post_ids, post_num, indices_are_sorted)
  denominator = _jit_seg_sum(jnp.ones_like(syn_values), post_ids, post_num, indices_are_sorted)
  return jnp.nan_to_num(nominator / denominator)


def syn2post_softmax(syn_values, post_ids, post_num: int, indices_are_sorted=False):
  """The syn-to-post softmax computation.

  Parameters
  ----------
  syn_values: ArrayType
    The synaptic values.
  post_ids: ArrayType
    The post-synaptic neuron ids. If ``post_ids`` is generated by
    ``brainpy.conn.TwoEndConnector``, then it has sorted indices.
    Otherwise, this function cannot guarantee indices are sorted.
    You's better set ``indices_are_sorted=False``.
  post_num: int
    The number of the post-synaptic neurons.
  indices_are_sorted: whether ``post_ids`` is known to be sorted.

  Returns
  -------
  post_val: ArrayType
    The post-synaptic value.
  """
  post_ids = as_jax(post_ids)
  syn_values = as_jax(syn_values)
  if syn_values.dtype == jnp.bool_:
    syn_values = jnp.asarray(syn_values, dtype=jnp.int32)
  syn_maxs = _jit_seg_max(syn_values, post_ids, post_num, indices_are_sorted)
  syn_values = syn_values - syn_maxs[post_ids]
  syn_values = jnp.exp(syn_values)
  normalizers = _jit_seg_sum(syn_values, post_ids, post_num, indices_are_sorted)
  softmax = syn_values / normalizers[post_ids]
  return jnp.nan_to_num(softmax)
