import jax.numpy as jnp
import jax.ops

from .ndarray import _return, _as_jax_array_
from .compat_numpy import (
  prod, min, sum, all, any, mean, std, var, concatenate, clip
)

__all__ = [
  'concat',
  'reduce_sum', 'reduce_max', 'reduce_min', 'reduce_mean', 'reduce_all', 'reduce_any',
  'reduce_logsumexp', 'reduce_prod', 'reduce_std', 'reduce_variance', 'reduce_euclidean_norm',
  'unsorted_segment_sqrt_n', 'segment_mean', 'unsorted_segment_sum', 'unsorted_segment_prod',
  'unsorted_segment_max', 'unsorted_segment_min', 'unsorted_segment_mean',
  'clip_by_value',
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
                          indices_are_sorted=True)
  d = jax.ops.segment_sum(jnp.ones_like(data),
                          _as_jax_array_(segment_ids),
                          indices_are_sorted=True)
  return _return(jnp.nan_to_num(r / d))


def unsorted_segment_sum(data, segment_ids, num_segments):
  """Computes the sum along segments of a tensor.

  See https://tensorflow.google.cn/api_docs/python/tf/math/unsorted_segment_sum

  """
  r = jax.ops.segment_sum(_as_jax_array_(data),
                          _as_jax_array_(segment_ids),
                          num_segments=num_segments,
                          indices_are_sorted=True)
  return _return(r)


def unsorted_segment_prod(data, segment_ids, num_segments):
  """Computes the product along segments of a tensor.

  See https://tensorflow.google.cn/api_docs/python/tf/math/unsorted_segment_prod

  """
  r = jax.ops.segment_prod(_as_jax_array_(data),
                           _as_jax_array_(segment_ids),
                           num_segments=num_segments,
                           indices_are_sorted=True)
  return _return(r)


def unsorted_segment_max(data, segment_ids, num_segments):
  """Computes the maximum along segments of a tensor.

  See https://tensorflow.google.cn/api_docs/python/tf/math/unsorted_segment_max

  """
  r = jax.ops.segment_max(_as_jax_array_(data),
                          _as_jax_array_(segment_ids),
                          num_segments=num_segments,
                          indices_are_sorted=True)
  return _return(r)


def unsorted_segment_min(data, segment_ids, num_segments):
  """Computes the minimum along segments of a tensor.

  See https://tensorflow.google.cn/api_docs/python/tf/math/unsorted_segment_min

  """
  r = jax.ops.segment_min(_as_jax_array_(data),
                          _as_jax_array_(segment_ids),
                          num_segments=num_segments,
                          indices_are_sorted=True)
  return _return(r)


def unsorted_segment_sqrt_n(data, segment_ids, num_segments):
  """Computes the sum along segments of a tensor divided by the sqrt(N).

  See https://tensorflow.google.cn/api_docs/python/tf/math/unsorted_segment_sqrt_n

  """
  r = jax.ops.segment_sum(_as_jax_array_(data),
                          _as_jax_array_(segment_ids),
                          num_segments=num_segments,
                          indices_are_sorted=True)
  d = jax.ops.segment_sum(jnp.ones_like(data),
                          _as_jax_array_(segment_ids),
                          num_segments=num_segments,
                          indices_are_sorted=True)
  return _return(jnp.nan_to_num(r / jnp.sqrt(d)))


def unsorted_segment_mean(data, segment_ids, num_segments):
  """Computes the average along segments of a tensor.

  See https://tensorflow.google.cn/api_docs/python/tf/math/unsorted_segment_mean

  """
  r = jax.ops.segment_sum(_as_jax_array_(data),
                          _as_jax_array_(segment_ids),
                          num_segments=num_segments,
                          indices_are_sorted=True)
  d = jax.ops.segment_sum(jnp.ones_like(data),
                          _as_jax_array_(segment_ids),
                          num_segments=num_segments,
                          indices_are_sorted=True)
  return _return(jnp.nan_to_num(r / d))
