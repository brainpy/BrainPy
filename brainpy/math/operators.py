# -*- coding: utf-8 -*-

try:
  import brainpylib
except (ImportError, ModuleNotFoundError):
  brainpylib = None
from jax import jit, vmap
from jax import ops as jops

from brainpy.errors import PackageMissingError
from brainpy.math.jaxarray import JaxArray
from brainpy.math.numpy_ops import append, zeros_like

__all__ = [
  'event_add',
  'pre2syn', 'pre2post',
  'syn2post',
  'syn2post_sum',
  'syn2post_prod',
  'syn2post_max',
  'syn2post_min',
]

_pre2post = vmap(lambda pre_ids, pre_vs: pre_vs[pre_ids].sum(), in_axes=(0, None))
_pre2syn = vmap(lambda pre_id, pre_vs: pre_vs[pre_id], in_axes=(0, None))
_syn2post = jit(jops.segment_sum, static_argnums=2)
_jit_seg_sum = jit(jops.segment_sum, static_argnums=(2, 3, 4, 5))
_jit_seg_prod = jit(jops.segment_prod, static_argnums=(2, 3, 4, 5))
_jit_seg_max = jit(jops.segment_max, static_argnums=(2, 3, 4, 5))
_jit_seg_min = jit(jops.segment_min, static_argnums=(2, 3, 4, 5))


def event_add(events, pre2post, post_num, values):
  """Event add.

  Parameters
  ----------
  events: JaxArray, jnp.ndarray
  pre2post: tuple of JaxArray, tuple of jnp.ndarray
  post_num: int
  values: float, JaxArray, jnp.ndarray

  Returns
  -------
  out: JaxArray, jnp.ndarray
  """
  if brainpylib is None:
    raise PackageMissingError(
      '"brainpylib" must be installed when the user wants to use "event_add" operator. \n'
      'Please install "brainpylib" through:\n\n'
      '>>> pip install brainpylib')

  indices, idnptr = pre2post
  events = events.value if isinstance(events, JaxArray) else events
  indices = indices.value if isinstance(indices, JaxArray) else indices
  idnptr = idnptr.value if isinstance(idnptr, JaxArray) else idnptr
  values = values.value if isinstance(values, JaxArray) else values
  return brainpylib.event_add(events, (indices, idnptr), post_num, values)


def pre2post(pre_values, post2pre_conn):
  pre_values = append(pre_values, zeros_like(pre_values[0]))
  pre_values = pre_values.value
  post2pre_conn = post2pre_conn.value if isinstance(post2pre_conn, JaxArray) else post2pre_conn
  return _pre2post(post2pre_conn, pre_values)


def pre2syn(pre_values, pre_ids):
  pre_values = pre_values.value if isinstance(pre_values, JaxArray) else pre_values
  pre_ids = pre_ids.value if isinstance(pre_ids, JaxArray) else pre_ids
  return _pre2syn(pre_ids, pre_values)


def syn2post(syn_values, post_ids, post_num):
  """Syn to post

  Parameters
  ----------
  syn_values
  post_ids
  post_num

  Returns
  -------

  """
  syn_values = syn_values.value if isinstance(syn_values, JaxArray) else syn_values
  post_ids = post_ids.value if isinstance(post_ids, JaxArray) else post_ids
  return _syn2post(syn_values, post_ids, post_num)


def syn2post_sum(data, segment_ids, num_segments: int, indices_are_sorted: bool = False,
                 unique_indices: bool = False, bucket_size: int = None):
  """Computes the sum within segments of an array.

  Parameters
  ----------
  data
  segment_ids
  num_segments
  indices_are_sorted
  unique_indices
  bucket_size

  Returns
  -------

  """
  data = data.value if isinstance(data, JaxArray) else data
  segment_ids = segment_ids.value if isinstance(segment_ids, JaxArray) else segment_ids
  return _jit_seg_sum(data, segment_ids, num_segments,
                      indices_are_sorted, unique_indices, bucket_size)


def syn2post_prod(data, segment_ids, num_segments: int, indices_are_sorted: bool = False,
                  unique_indices: bool = False, bucket_size: int = None):
  """Computes the product within segments of an array.

  Parameters
  ----------
  data
  segment_ids
  num_segments
  indices_are_sorted
  unique_indices
  bucket_size

  Returns
  -------

  """
  data = data.value if isinstance(data, JaxArray) else data
  segment_ids = segment_ids.value if isinstance(segment_ids, JaxArray) else segment_ids
  return _jit_seg_prod(data, segment_ids, num_segments,
                       indices_are_sorted, unique_indices, bucket_size)


def syn2post_max(data, segment_ids, num_segments: int, indices_are_sorted: bool = False,
                 unique_indices: bool = False, bucket_size: int = None):
  """Computes the product within segments of an array.

  Parameters
  ----------
  data
  segment_ids
  num_segments
  indices_are_sorted
  unique_indices
  bucket_size

  Returns
  -------

  """
  data = data.value if isinstance(data, JaxArray) else data
  segment_ids = segment_ids.value if isinstance(segment_ids, JaxArray) else segment_ids
  return _jit_seg_max(data, segment_ids, num_segments,
                      indices_are_sorted, unique_indices, bucket_size)


def syn2post_min(data, segment_ids, num_segments: int, indices_are_sorted: bool = False,
                 unique_indices: bool = False, bucket_size: int = None):
  """Computes the product within segments of an array.

  Parameters
  ----------
  data
  segment_ids
  num_segments
  indices_are_sorted
  unique_indices
  bucket_size

  Returns
  -------

  """
  data = data.value if isinstance(data, JaxArray) else data
  segment_ids = segment_ids.value if isinstance(segment_ids, JaxArray) else segment_ids
  return _jit_seg_min(data, segment_ids, num_segments,
                      indices_are_sorted, unique_indices, bucket_size)
