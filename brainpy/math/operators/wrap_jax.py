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
  """``segment_sum`` operator for brainpy `JaxArray` and `Variable`.

  Parameters
  ----------
  data: Array
    An array with the values to be reduced.
  segment_ids: Array
    An array with integer dtype that indicates the segments of
    `data` (along its leading axis) to be summed. Values can be repeated and
    need not be sorted.
  num_segments: Optional, int
    An int with nonnegative value indicating the number
    of segments. The default is set to be the minimum number of segments that
    would support all indices in ``segment_ids``, calculated as
    ``max(segment_ids) + 1``.
    Since `num_segments` determines the size of the output, a static value
    must be provided to use ``segment_sum`` in a ``jit``-compiled function.
  indices_are_sorted: bool
    whether ``segment_ids`` is known to be sorted.
  unique_indices: bool
    whether `segment_ids` is known to be free of duplicates.
  bucket_size: int
    Size of bucket to group indices into. ``segment_sum`` is
    performed on each bucket separately to improve numerical stability of
    addition. Default ``None`` means no bucketing.

  Returns
  -------
  output: Array
    An array with shape :code:`(num_segments,) + data.shape[1:]` representing the
    segment sums.
  """
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
  """``segment_prod`` operator for brainpy `JaxArray` and `Variable`.

  Parameters
  ----------
  data: Array
    An array with the values to be reduced.
  segment_ids: Array
    An array with integer dtype that indicates the segments of
    `data` (along its leading axis) to be summed. Values can be repeated and
    need not be sorted.
  num_segments: Optional, int
    An int with nonnegative value indicating the number
    of segments. The default is set to be the minimum number of segments that
    would support all indices in ``segment_ids``, calculated as
    ``max(segment_ids) + 1``.
    Since `num_segments` determines the size of the output, a static value
    must be provided to use ``segment_sum`` in a ``jit``-compiled function.
  indices_are_sorted: bool
    whether ``segment_ids`` is known to be sorted.
  unique_indices: bool
    whether `segment_ids` is known to be free of duplicates.
  bucket_size: int
    Size of bucket to group indices into. ``segment_sum`` is
    performed on each bucket separately to improve numerical stability of
    addition. Default ``None`` means no bucketing.

  Returns
  -------
  output: Array
    An array with shape :code:`(num_segments,) + data.shape[1:]` representing the
    segment sums.
  """
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
  """``segment_max`` operator for brainpy `JaxArray` and `Variable`.

  Parameters
  ----------
  data: Array
    An array with the values to be reduced.
  segment_ids: Array
    An array with integer dtype that indicates the segments of
    `data` (along its leading axis) to be summed. Values can be repeated and
    need not be sorted.
  num_segments: Optional, int
    An int with nonnegative value indicating the number
    of segments. The default is set to be the minimum number of segments that
    would support all indices in ``segment_ids``, calculated as
    ``max(segment_ids) + 1``.
    Since `num_segments` determines the size of the output, a static value
    must be provided to use ``segment_sum`` in a ``jit``-compiled function.
  indices_are_sorted: bool
    whether ``segment_ids`` is known to be sorted.
  unique_indices: bool
    whether `segment_ids` is known to be free of duplicates.
  bucket_size: int
    Size of bucket to group indices into. ``segment_sum`` is
    performed on each bucket separately to improve numerical stability of
    addition. Default ``None`` means no bucketing.

  Returns
  -------
  output: Array
    An array with shape :code:`(num_segments,) + data.shape[1:]` representing the
    segment sums.
  """
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
  """``segment_min`` operator for brainpy `JaxArray` and `Variable`.

  Parameters
  ----------
  data: Array
    An array with the values to be reduced.
  segment_ids: Array
    An array with integer dtype that indicates the segments of
    `data` (along its leading axis) to be summed. Values can be repeated and
    need not be sorted.
  num_segments: Optional, int
    An int with nonnegative value indicating the number
    of segments. The default is set to be the minimum number of segments that
    would support all indices in ``segment_ids``, calculated as
    ``max(segment_ids) + 1``.
    Since `num_segments` determines the size of the output, a static value
    must be provided to use ``segment_sum`` in a ``jit``-compiled function.
  indices_are_sorted: bool
    whether ``segment_ids`` is known to be sorted.
  unique_indices: bool
    whether `segment_ids` is known to be free of duplicates.
  bucket_size: int
    Size of bucket to group indices into. ``segment_sum`` is
    performed on each bucket separately to improve numerical stability of
    addition. Default ``None`` means no bucketing.

  Returns
  -------
  output: Array
    An array with shape :code:`(num_segments,) + data.shape[1:]` representing the
    segment sums.
  """
  return JaxArray(jops.segment_min(data.value if isinstance(data, JaxArray) else data,
                                   segment_ids.value if isinstance(segment_ids, JaxArray) else segment_ids,
                                   num_segments,
                                   indices_are_sorted,
                                   unique_indices,
                                   bucket_size, mode))
