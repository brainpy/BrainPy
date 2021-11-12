# -*- coding: utf-8 -*-


from jax import jit, vmap
from jax import ops as jops

from brainpy.math.jax.jaxarray import JaxArray

__all__ = [
  'pre2syn', 'syn2post',
  'segment_sum', 'segment_prod', 'segment_max', 'segment_min',
]

_pre2syn = vmap(lambda pre_id, pre_vs: pre_vs[pre_id], in_axes=(0, None))
_syn2post = jit(jops.segment_sum, static_argnums=2)
_jit_seg_sum = jit(jops.segment_sum, static_argnums=(2, 3, 4, 5))
_jit_seg_prod = jit(jops.segment_prod, static_argnums=(2, 3, 4, 5))
_jit_seg_max = jit(jops.segment_max, static_argnums=(2, 3, 4, 5))
_jit_seg_min = jit(jops.segment_min, static_argnums=(2, 3, 4, 5))


def pre2syn(pre_values, pre_ids):
  pre_values = pre_values.value if isinstance(pre_values, JaxArray) else pre_values
  pre_ids = pre_ids.value if isinstance(pre_ids, JaxArray) else pre_ids
  return _pre2syn(pre_ids, pre_values)


def syn2post(syn_values, post_ids, post_num):
  syn_values = syn_values.value if isinstance(syn_values, JaxArray) else syn_values
  post_ids = post_ids.value if isinstance(post_ids, JaxArray) else post_ids
  return _syn2post(syn_values, post_ids, post_num)


def segment_sum(data, segment_ids, num_segments: int, indices_are_sorted: bool = False,
                unique_indices: bool = False, bucket_size: int = None):
  """Computes the sum within segments of an array."""
  data = data.value if isinstance(data, JaxArray) else data
  segment_ids = segment_ids.value if isinstance(segment_ids, JaxArray) else segment_ids
  return _jit_seg_sum(data, segment_ids, num_segments,
                      indices_are_sorted, unique_indices, bucket_size)


def segment_prod(data, segment_ids, num_segments: int, indices_are_sorted: bool = False,
                 unique_indices: bool = False, bucket_size: int = None):
  """Computes the product within segments of an array."""
  data = data.value if isinstance(data, JaxArray) else data
  segment_ids = segment_ids.value if isinstance(segment_ids, JaxArray) else segment_ids
  return _jit_seg_prod(data, segment_ids, num_segments,
                       indices_are_sorted, unique_indices, bucket_size)


def segment_max(data, segment_ids, num_segments: int, indices_are_sorted: bool = False,
                unique_indices: bool = False, bucket_size: int = None):
  """Computes the product within segments of an array."""
  data = data.value if isinstance(data, JaxArray) else data
  segment_ids = segment_ids.value if isinstance(segment_ids, JaxArray) else segment_ids
  return _jit_seg_max(data, segment_ids, num_segments,
                      indices_are_sorted, unique_indices, bucket_size)


def segment_min(data, segment_ids, num_segments: int, indices_are_sorted: bool = False,
                unique_indices: bool = False, bucket_size: int = None):
  """Computes the product within segments of an array."""
  data = data.value if isinstance(data, JaxArray) else data
  segment_ids = segment_ids.value if isinstance(segment_ids, JaxArray) else segment_ids
  return _jit_seg_min(data, segment_ids, num_segments,
                      indices_are_sorted, unique_indices, bucket_size)
