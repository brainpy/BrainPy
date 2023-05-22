from typing import Union, Optional

import jax.numpy as jnp
import jax.ops
from jax import lax

from brainpy._src.math.interoperability import as_jax
from brainpy._src.math.ndarray import Array
from .compat_numpy import (
  prod, min, sum, all, any, mean, std, var, concatenate, clip,
  asarray,
)
from .ndarray import _return, _as_jax_array_

__all__ = [
  'concat',
  'reduce_sum', 'reduce_max', 'reduce_min', 'reduce_mean', 'reduce_all', 'reduce_any',
  'reduce_logsumexp', 'reduce_prod', 'reduce_std', 'reduce_variance', 'reduce_euclidean_norm',
  'unsorted_segment_sqrt_n', 'segment_mean', 'unsorted_segment_sum', 'unsorted_segment_prod',
  'unsorted_segment_max', 'unsorted_segment_min', 'unsorted_segment_mean',
  'segment_sum', 'segment_prod', 'segment_max', 'segment_min',
  'clip_by_value', 'cast',
]

reduce_prod = prod
reduce_sum = sum
reduce_all = all
reduce_any = any
reduce_min = min
reduce_mean = mean
reduce_std = std
reduce_variance = var
concat = concatenate
clip_by_value = clip


def reduce_logsumexp(input_tensor, axis=None, keepdims=False):
  """Computes log(sum(exp(elements across dimensions of a tensor))).

  Reduces `input_tensor` along the dimensions given in `axis`.

  Unless `keepdims` is true, the rank of the tensor is reduced by 1 for each
  of the entries in `axis`, which must be unique. If `keepdims` is true, the
  reduced dimensions are retained with length 1.

  If `axis` has no entries, all dimensions are reduced, and a
  tensor with a single element is returned.

  This function is more numerically stable than log(sum(exp(input))). It avoids
  overflows caused by taking the exp of large inputs and underflows caused by
  taking the log of small inputs.

  Args:
    input_tensor: The tensor to reduce. Should have numeric type.
    axis: The dimensions to reduce. If `None` (the default), reduces all
      dimensions. Must be in the range `[-rank(input_tensor),
      rank(input_tensor))`.
    keepdims: If true, retains reduced dimensions with length 1.

  Returns:
    The reduced tensor.
  """
  r = jnp.log(jnp.sum(jnp.exp(_as_jax_array_(input_tensor)), axis=axis, keepdims=keepdims))
  return _return(r)


def reduce_euclidean_norm(input_tensor, axis=None, keepdims=False):
  """Computes the Euclidean norm of elements across dimensions of a tensor.
  Reduces `input_tensor` along the dimensions given in `axis`.

  Unless `keepdims` is true, the rank of the tensor is reduced by 1 for each
  of the entries in `axis`, which must be unique. If `keepdims` is true, the
  reduced dimensions are retained with length 1.

  If `axis` is None, all dimensions are reduced, and a
  tensor with a single element is returned.

  Args:
    input_tensor: The tensor to reduce. Should have numeric type.
    axis: The dimensions to reduce. If `None` (the default), reduces all
      dimensions. Must be in the range `[-rank(input_tensor),
      rank(input_tensor))`.
    keepdims: If true, retains reduced dimensions with length 1.

  Returns:
    The reduced tensor, of the same dtype as the input_tensor.
  """
  r = jnp.linalg.norm(_as_jax_array_(input_tensor), axis=axis, keepdims=keepdims)
  return _return(r)


def reduce_max(input_tensor, axis=None, keepdims=False):
  """Computes `maximum` of elements across dimensions of a tensor.
  
  This is the reduction operation for the elementwise `maximum` op.
  Reduces `input_tensor` along the dimensions given in `axis`.

  Unless `keepdims` is true, the rank of the tensor is reduced by 1 for each
  of the entries in `axis`, which must be unique. If `keepdims` is true, the
  reduced dimensions are retained with length 1.

  If `axis` is None, all dimensions are reduced, and a
  tensor with a single element is returned.

  Args:
    input_tensor: The tensor to reduce. Should have real numeric type.
    axis: The dimensions to reduce. If `None` (the default), reduces all
      dimensions. Must be in the range `[-rank(input_tensor),
      rank(input_tensor))`.
    keepdims: If true, retains reduced dimensions with length 1.

  Returns:
    The reduced tensor.
  """
  return _return(jnp.max(_as_jax_array_(input_tensor), axis=axis, keepdims=keepdims))


def segment_mean(data, segment_ids):
  """Computes the average along segments of a tensor.

  See https://tensorflow.google.cn/api_docs/python/tf/math/segment_mean

  """
  r = jax.ops.segment_sum(_as_jax_array_(data),
                          _as_jax_array_(segment_ids),
                          indices_are_sorted=False)
  d = jax.ops.segment_sum(jnp.ones_like(data),
                          _as_jax_array_(segment_ids),
                          indices_are_sorted=False)
  return _return(jnp.nan_to_num(r / d))


def unsorted_segment_sum(data, segment_ids, num_segments):
  """Computes the sum along segments of a tensor.

  See https://tensorflow.google.cn/api_docs/python/tf/math/unsorted_segment_sum

  """
  r = jax.ops.segment_sum(_as_jax_array_(data),
                          _as_jax_array_(segment_ids),
                          num_segments=num_segments,
                          indices_are_sorted=False)
  return _return(r)


def unsorted_segment_prod(data, segment_ids, num_segments):
  """Computes the product along segments of a tensor.

  See https://tensorflow.google.cn/api_docs/python/tf/math/unsorted_segment_prod

  """
  r = jax.ops.segment_prod(_as_jax_array_(data),
                           _as_jax_array_(segment_ids),
                           num_segments=num_segments,
                           indices_are_sorted=False)
  return _return(r)


def unsorted_segment_max(data, segment_ids, num_segments):
  """Computes the maximum along segments of a tensor.

  See https://tensorflow.google.cn/api_docs/python/tf/math/unsorted_segment_max

  """
  r = jax.ops.segment_max(_as_jax_array_(data),
                          _as_jax_array_(segment_ids),
                          num_segments=num_segments,
                          indices_are_sorted=False)
  return _return(r)


def unsorted_segment_min(data, segment_ids, num_segments):
  """Computes the minimum along segments of a tensor.

  See https://tensorflow.google.cn/api_docs/python/tf/math/unsorted_segment_min

  """
  r = jax.ops.segment_min(_as_jax_array_(data),
                          _as_jax_array_(segment_ids),
                          num_segments=num_segments,
                          indices_are_sorted=False)
  return _return(r)


def unsorted_segment_sqrt_n(data, segment_ids, num_segments):
  """Computes the sum along segments of a tensor divided by the sqrt(N).

  See https://tensorflow.google.cn/api_docs/python/tf/math/unsorted_segment_sqrt_n

  """
  r = jax.ops.segment_sum(_as_jax_array_(data),
                          _as_jax_array_(segment_ids),
                          num_segments=num_segments,
                          indices_are_sorted=False)
  d = jax.ops.segment_sum(jnp.ones_like(data),
                          _as_jax_array_(segment_ids),
                          num_segments=num_segments,
                          indices_are_sorted=False)
  return _return(jnp.nan_to_num(r / jnp.sqrt(d)))


def unsorted_segment_mean(data, segment_ids, num_segments):
  """Computes the average along segments of a tensor.

  See https://tensorflow.google.cn/api_docs/python/tf/math/unsorted_segment_mean

  """
  r = jax.ops.segment_sum(_as_jax_array_(data),
                          _as_jax_array_(segment_ids),
                          num_segments=num_segments,
                          indices_are_sorted=False)
  d = jax.ops.segment_sum(jnp.ones_like(data),
                          _as_jax_array_(segment_ids),
                          num_segments=num_segments,
                          indices_are_sorted=False)
  return _return(jnp.nan_to_num(r / d))


def segment_sum(data: Union[Array, jnp.ndarray],
                segment_ids: Union[Array, jnp.ndarray],
                num_segments: Optional[int] = None,
                indices_are_sorted: bool = False,
                unique_indices: bool = False,
                bucket_size: Optional[int] = None,
                mode: Optional[lax.GatherScatterMode] = None) -> Array:
  """``segment_sum`` operator for brainpy `Array` and `Variable`.

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
  mode: lax.GatherScatterMode
    A :class:`jax.lax.GatherScatterMode` value describing how
    out-of-bounds indices should be handled. By default, values outside of the
    range [0, num_segments) are dropped and do not contribute to the sum.

  Returns
  -------
  output: Array
    An array with shape :code:`(num_segments,) + data.shape[1:]` representing the
    segment sums.
  """
  return Array(jax.ops.segment_sum(as_jax(data),
                                   as_jax(segment_ids),
                                   num_segments,
                                   indices_are_sorted,
                                   unique_indices,
                                   bucket_size,
                                   mode))


def segment_prod(data: Union[Array, jnp.ndarray],
                 segment_ids: Union[Array, jnp.ndarray],
                 num_segments: Optional[int] = None,
                 indices_are_sorted: bool = False,
                 unique_indices: bool = False,
                 bucket_size: Optional[int] = None,
                 mode: Optional[lax.GatherScatterMode] = None) -> Array:
  """``segment_prod`` operator for brainpy `Array` and `Variable`.

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
  mode: lax.GatherScatterMode
    A :class:`jax.lax.GatherScatterMode` value describing how
    out-of-bounds indices should be handled. By default, values outside of the
    range [0, num_segments) are dropped and do not contribute to the sum.

  Returns
  -------
  output: Array
    An array with shape :code:`(num_segments,) + data.shape[1:]` representing the
    segment sums.
  """
  return Array(jax.ops.segment_prod(as_jax(data),
                                    as_jax(segment_ids),
                                    num_segments,
                                    indices_are_sorted,
                                    unique_indices,
                                    bucket_size,
                                    mode))


def segment_max(data: Union[Array, jnp.ndarray],
                segment_ids: Union[Array, jnp.ndarray],
                num_segments: Optional[int] = None,
                indices_are_sorted: bool = False,
                unique_indices: bool = False,
                bucket_size: Optional[int] = None,
                mode: Optional[lax.GatherScatterMode] = None) -> Array:
  """``segment_max`` operator for brainpy `Array` and `Variable`.

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
  mode: lax.GatherScatterMode
    A :class:`jax.lax.GatherScatterMode` value describing how
    out-of-bounds indices should be handled. By default, values outside of the
    range [0, num_segments) are dropped and do not contribute to the sum.

  Returns
  -------
  output: Array
    An array with shape :code:`(num_segments,) + data.shape[1:]` representing the
    segment sums.
  """
  return Array(jax.ops.segment_max(as_jax(data),
                                   as_jax(segment_ids),
                                   num_segments,
                                   indices_are_sorted,
                                   unique_indices,
                                   bucket_size,
                                   mode))


def segment_min(data: Union[Array, jnp.ndarray],
                segment_ids: Union[Array, jnp.ndarray],
                num_segments: Optional[int] = None,
                indices_are_sorted: bool = False,
                unique_indices: bool = False,
                bucket_size: Optional[int] = None,
                mode: Optional[lax.GatherScatterMode] = None) -> Array:
  """``segment_min`` operator for brainpy `Array` and `Variable`.

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
  mode: lax.GatherScatterMode
    A :class:`jax.lax.GatherScatterMode` value describing how
    out-of-bounds indices should be handled. By default, values outside of the
    range [0, num_segments) are dropped and do not contribute to the sum.

  Returns
  -------
  output: Array
    An array with shape :code:`(num_segments,) + data.shape[1:]` representing the
    segment sums.
  """
  return Array(jax.ops.segment_min(as_jax(data),
                                   as_jax(segment_ids),
                                   num_segments,
                                   indices_are_sorted,
                                   unique_indices,
                                   bucket_size,
                                   mode))


def cast(x, dtype):
  """Casts a tensor to a new type.

  The operation casts `x` (in case of `Tensor`) or `x.values`
  (in case of `SparseTensor` or `IndexedSlices`) to `dtype`.

  The operation supports data types (for `x` and `dtype`) of
  `uint8`, `uint16`, `uint32`, `uint64`, `int8`, `int16`, `int32`, `int64`,
  `float16`, `float32`, `float64`, `complex64`, `complex128`, `bfloat16`.
  In case of casting from complex types (`complex64`, `complex128`) to real
  types, only the real part of `x` is returned. In case of casting from real
  types to complex types (`complex64`, `complex128`), the imaginary part of the
  returned value is set to `0`. The handling of complex types here matches the
  behavior of numpy.

  Note casting nan and inf values to integral types has undefined behavior.

  Args:
    x: A `Array`. It could be `uint8`, `uint16`, `uint32`, `uint64`, `int8`, `int16`, `int32`,
      `int64`, `float16`, `float32`, `float64`, `complex64`, `complex128`,
      `bfloat16`.
    dtype: The destination type. The list of supported dtypes is the same as
      `x`.
  Returns:
    A `Array` with same shape as `x` and same type as `dtype`.

  """
  return asarray(x, dtype=dtype)
