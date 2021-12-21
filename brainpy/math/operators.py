# -*- coding: utf-8 -*-


import jax.numpy as jnp
from jax import jit, vmap
from jax import ops as jops

from brainpy.errors import PackageMissingError
from brainpy.math.jaxarray import JaxArray
from brainpy.math.numpy_ops import asarray

try:
  import brainpylib
except ModuleNotFoundError:
  brainpylib = None

__all__ = [
  'pre2post_event_sum',
  'pre2post_event_sum2',
  'pre2post_sum',
  'pre2syn',
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


def _check_brainpylib(ops_name):
  if brainpylib is not None:
    return
  raise PackageMissingError(
    f'"brainpylib" must be installed when the user '
    f'wants to use "{ops_name}" operator. \n'
    f'Please install "brainpylib" through:\n\n'
    f'>>> pip install brainpylib'
  )


def pre2post_event_sum(events, pre2post, post_num, values):
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
  _check_brainpylib(pre2post_event_sum.__name__)
  indices, idnptr = pre2post
  events = events.value if isinstance(events, JaxArray) else events
  indices = indices.value if isinstance(indices, JaxArray) else indices
  idnptr = idnptr.value if isinstance(idnptr, JaxArray) else idnptr
  values = values.value if isinstance(values, JaxArray) else values
  return brainpylib.event_sum(events, (indices, idnptr), post_num, values)


def pre2post_event_sum2(events, pre_ids, post_ids, post_num, values):
  _check_brainpylib(pre2post_event_sum2.__name__)
  events = events.value if isinstance(events, JaxArray) else events
  pre_ids = pre_ids.value if isinstance(pre_ids, JaxArray) else pre_ids
  post_ids = post_ids.value if isinstance(post_ids, JaxArray) else post_ids
  values = values.value if isinstance(values, JaxArray) else values
  return brainpylib.event_sum2(events, pre_ids, post_ids, post_num, values)


def pre2post_sum(pre_values, pre_ids, post_ids, post_num):
  _check_brainpylib(pre2post_sum.__name__)
  pre_values = asarray(pre_values, dtype=jnp.float_)
  pre_values = pre_values.value if isinstance(pre_values, JaxArray) else pre_values
  pre_ids = pre_ids.value if isinstance(pre_ids, JaxArray) else pre_ids
  post_ids = post_ids.value if isinstance(post_ids, JaxArray) else post_ids
  post_num = post_num.value if isinstance(post_num, JaxArray) else post_num
  return brainpylib.atomic_sum(pre_values, pre_ids, post_ids, post_num)


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
  if syn_values.dtype == jnp.bool_:
    syn_values = jnp.asarray(syn_values, dtype=jnp.int_)
  post_ids = post_ids.value if isinstance(post_ids, JaxArray) else post_ids
  return _syn2post(syn_values, post_ids, post_num)


def syn2post_sum(syn_values, post_ids, post_num: int,
                 indices_are_sorted: bool = False,
                 unique_indices: bool = False,
                 bucket_size: int = None):
  """Computes the sum within segments of an array.

  Parameters
  ----------
  syn_values
  post_ids
  post_num
  indices_are_sorted
  unique_indices
  bucket_size

  Returns
  -------

  """
  syn_values = syn_values.value if isinstance(syn_values, JaxArray) else syn_values
  post_ids = post_ids.value if isinstance(post_ids, JaxArray) else post_ids
  return _jit_seg_sum(syn_values, post_ids, post_num,
                      indices_are_sorted, unique_indices, bucket_size)


def syn2post_prod(syn_values, post_ids, post_num: int,
                  indices_are_sorted: bool = False,
                  unique_indices: bool = False,
                  bucket_size: int = None):
  """Computes the product within segments of an array.

  Parameters
  ----------
  syn_values
  post_ids
  post_num
  indices_are_sorted
  unique_indices
  bucket_size

  Returns
  -------

  """
  syn_values = syn_values.value if isinstance(syn_values, JaxArray) else syn_values
  post_ids = post_ids.value if isinstance(post_ids, JaxArray) else post_ids
  return _jit_seg_prod(syn_values, post_ids, post_num,
                       indices_are_sorted, unique_indices, bucket_size)


def syn2post_max(syn_values, post_ids, post_num: int,
                 indices_are_sorted: bool = False,
                 unique_indices: bool = False,
                 bucket_size: int = None):
  """Computes the product within segments of an array.

  Parameters
  ----------
  syn_values
  post_ids
  post_num
  indices_are_sorted
  unique_indices
  bucket_size

  Returns
  -------

  """
  syn_values = syn_values.value if isinstance(syn_values, JaxArray) else syn_values
  post_ids = post_ids.value if isinstance(post_ids, JaxArray) else post_ids
  return _jit_seg_max(syn_values, post_ids, post_num,
                      indices_are_sorted, unique_indices, bucket_size)


def syn2post_min(syn_values, post_ids, post_num: int,
                 indices_are_sorted: bool = False,
                 unique_indices: bool = False,
                 bucket_size: int = None):
  """Computes the product within segments of an array.

  Parameters
  ----------
  syn_values
  post_ids
  post_num
  indices_are_sorted
  unique_indices
  bucket_size

  Returns
  -------

  """
  syn_values = syn_values.value if isinstance(syn_values, JaxArray) else syn_values
  post_ids = post_ids.value if isinstance(post_ids, JaxArray) else post_ids
  return _jit_seg_min(syn_values, post_ids, post_num,
                      indices_are_sorted, unique_indices, bucket_size)
