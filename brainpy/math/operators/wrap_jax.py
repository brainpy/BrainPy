# -*- coding: utf-8 -*-


from typing import Union, Optional

import jax.numpy as jnp
from jax import lax
from jax import ops as jops

from brainpy.math.jaxarray import JaxArray

__all__ = [
  'segment_sum',
  'segment_prod',
  'segment_max',
  'segment_min',
]


def segment_sum(data: Union[JaxArray, jnp.ndarray],
                segment_ids: Union[JaxArray, jnp.ndarray],
                num_segments: Optional[int] = None,
                indices_are_sorted: bool = False,
                unique_indices: bool = False,
                bucket_size: Optional[int] = None,
                mode: Optional[lax.GatherScatterMode] = None) -> JaxArray:
  return JaxArray(jops.segment_sum(data.value if isinstance(data, JaxArray) else data,
                                   segment_ids.value if isinstance(segment_ids, JaxArray) else segment_ids,
                                   num_segments,
                                   indices_are_sorted,
                                   unique_indices,
                                   bucket_size, mode))


def segment_prod(data: Union[JaxArray, jnp.ndarray],
                 segment_ids: Union[JaxArray, jnp.ndarray],
                 num_segments: Optional[int] = None,
                 indices_are_sorted: bool = False,
                 unique_indices: bool = False,
                 bucket_size: Optional[int] = None,
                 mode: Optional[lax.GatherScatterMode] = None) -> JaxArray:
  return JaxArray(jops.segment_prod(data.value if isinstance(data, JaxArray) else data,
                                    segment_ids.value if isinstance(segment_ids, JaxArray) else segment_ids,
                                    num_segments,
                                    indices_are_sorted,
                                    unique_indices,
                                    bucket_size, mode))


def segment_max(data: Union[JaxArray, jnp.ndarray],
                segment_ids: Union[JaxArray, jnp.ndarray],
                num_segments: Optional[int] = None,
                indices_are_sorted: bool = False,
                unique_indices: bool = False,
                bucket_size: Optional[int] = None,
                mode: Optional[lax.GatherScatterMode] = None) -> JaxArray:
  return JaxArray(jops.segment_max(data.value if isinstance(data, JaxArray) else data,
                                   segment_ids.value if isinstance(segment_ids, JaxArray) else segment_ids,
                                   num_segments,
                                   indices_are_sorted,
                                   unique_indices,
                                   bucket_size, mode))


def segment_min(data: Union[JaxArray, jnp.ndarray],
                segment_ids: Union[JaxArray, jnp.ndarray],
                num_segments: Optional[int] = None,
                indices_are_sorted: bool = False,
                unique_indices: bool = False,
                bucket_size: Optional[int] = None,
                mode: Optional[lax.GatherScatterMode] = None) -> JaxArray:
  return JaxArray(jops.segment_min(data.value if isinstance(data, JaxArray) else data,
                                   segment_ids.value if isinstance(segment_ids, JaxArray) else segment_ids,
                                   num_segments,
                                   indices_are_sorted,
                                   unique_indices,
                                   bucket_size, mode))
