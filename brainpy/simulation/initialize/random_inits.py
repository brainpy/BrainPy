# -*- coding: utf-8 -*-

import numpy as np

from brainpy import math
from .base import Initializer

__all__ = [
  'Normal',
  'Uniform',
  'Orthogonal',
  'KaimingNormal',
  'KaimingNormalTruncated',
  'XavierNormal',
  'XavierNormalTruncated',
  'TruncatedNormal',
]


class Normal(Initializer):
  """Initialize weights with normal distribution.

  Parameters
  ----------
  gain : float
    The gain of the derivation of the normal distribution.

  """
  def __init__(self, gain=1.):
    super(Normal, self).__init__()
    self.gain = gain

  def __call__(self, shape, dtype=None):
    weights = math.random.normal(size=shape, scale=self.gain * np.sqrt(1 / np.prod(shape)))
    return math.asarray(weights, dtype=dtype)


class Uniform(Initializer):
  """Initialize weights with uniform distribution.

  Parameters
  ----------
  min_val : float
    The lower limit of the uniform distribution.
  max_val : float
    The upper limit of the uniform distribution.

  """
  def __init__(self, min_val=0., max_val=1.):
    super(Uniform, self).__init__()
    self.min_val = min_val
    self.max_val = max_val

  def __call__(self, shape, dtype=None):
    r = math.random.uniform(low=self.min_val, high=self.max_val, size=shape)
    return math.asarray(r, dtype=dtype)


class Orthogonal(Initializer):
  """Returns a uniformly distributed orthogonal tensor from
    `Exact solutions to the nonlinear dynamics of learning in deep linear neural networks
    <https://openreview.net/forum?id=_wzZwKpTDF_9C>`_.

    Args:
        shape: shape of the output tensor.
        gain: optional scaling factor.
        axis: the orthogonalizarion axis

    Returns:
        An orthogonally initialized tensor.
        These tensors will be row-orthonormal along the access specified by
        ``axis``. If the rank of the weight is greater than 2, the shape will be
        flattened in all other dimensions and then will be row-orthonormal along the
        final dimension. Note that this only works if the ``axis`` dimension is
        larger, otherwise the tensor will be transposed (equivalently, it will be
        column orthonormal instead of row orthonormal).
        If the shape is not square, the matrices will have orthonormal rows or
        columns depending on which side is smaller.
    """

  def __init__(self, gain=1., axis=-1):
    super(Orthogonal, self).__init__()
    self.gain = gain
    self.axis = axis

  def __call__(self, shape, dtype=None):
    n_rows = shape[self.axis]
    n_cols = np.prod(shape) // n_rows
    matrix_shape = (n_rows, n_cols) if n_rows > n_cols else (n_cols, n_rows)
    norm_dst = math.random.normal(size=matrix_shape)
    q_mat, r_mat = np.linalg.qr(norm_dst)
    # Enforce Q is uniformly distributed
    q_mat *= np.sign(np.diag(r_mat))
    if n_rows < n_cols: q_mat = q_mat.T
    q_mat = np.reshape(q_mat, (n_rows,) + tuple(np.delete(shape, self.axis)))
    q_mat = np.moveaxis(q_mat, 0, self.axis)
    return self.gain * math.asarray(q_mat, dtype=dtype)


class KaimingNormal(Initializer):
  """Returns a tensor with values assigned using Kaiming He normal initializer from
    `Delving Deep into Rectifiers: Surpassing Human-Level Performance on ImageNet Classification
    <https://arxiv.org/abs/1502.01852>`_.

    Args:
        shape: shape of the output tensor.
        gain: optional scaling factor.

    Returns:
        Tensor initialized with normal random variables with standard deviation (gain * kaiming_normal_gain).
    """

  def __init__(self, gain=1.):
    self.gain = gain
    super(KaimingNormal, self).__init__()

  def __call__(self, shape, dtype=None):
    gain = np.sqrt(1 / np.prod(shape[:-1]))
    res = math.random.normal(size=shape, scale=self.gain * gain)
    return math.asarray(res, dtype=dtype)


class KaimingNormalTruncated(Initializer):
  """Returns a tensor with values assigned using Kaiming He truncated normal initializer from
    `Delving Deep into Rectifiers: Surpassing Human-Level Performance on ImageNet Classification
    <https://arxiv.org/abs/1502.01852>`_.

    Args:
        shape: shape of the output tensor.
        lower: lower truncation of the normal.
        upper: upper truncation of the normal.
        gain: optional scaling factor.

    Returns:
        Tensor initialized with truncated normal random variables with standard
        deviation (gain * kaiming_normal_gain) and support [lower, upper].
    """

  def __init__(self, lower=-2., upper=2., gain=1.):
    self.lower = lower
    self.upper = upper
    self.gain = gain
    super(KaimingNormalTruncated, self).__init__()

  def __call__(self, shape, dtype=None):
    truncated_std = scipy.stats.truncnorm.std(a=self.lower,
                                              b=self.upper,
                                              loc=0.,
                                              scale=1.)
    stddev = self.gain * np.sqrt(1 / np.prod(shape[:-1])) / truncated_std
    res = math.random.truncated_normal(size=shape,
                                       scale=stddev,
                                       lower=self.lower,
                                       upper=self.upper)
    return math.asarray(res, dtype=dtype)


class XavierNormal(Initializer):
  """Returns a tensor with values assigned using Xavier Glorot normal initializer from
    `Understanding the difficulty of training deep feedforward neural networks
    <http://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf>`_.

    Args:
        shape: shape of the output tensor.
        gain: optional scaling factor.

    Returns:
        Tensor initialized with normal random variables with standard deviation (gain * xavier_normal_gain).
    """

  def __init__(self, gain=1.):
    super(XavierNormal, self).__init__()
    self.gain = gain

  def __call__(self, shape, dtype=None):
    fan_in, fan_out = np.prod(shape[:-1]), shape[-1]
    gain = np.sqrt(2 / (fan_in + fan_out))
    res = math.random.normal(size=shape, scale=self.gain * gain)
    return math.asarray(res, dtype=dtype)


class XavierNormalTruncated(Initializer):
  """Returns a tensor with values assigned using Xavier Glorot truncated normal initializer from
    `Understanding the difficulty of training deep feedforward neural networks
    <http://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf>`_.

    Args:
        shape: shape of the output tensor.
        lower: lower truncation of the normal.
        upper: upper truncation of the normal.
        gain: optional scaling factor.

    Returns:
        Tensor initialized with truncated normal random variables with standard
        deviation (gain * xavier_normal_gain) and support [lower, upper].
    """

  def __init__(self, lower=-2., upper=2., gain=1.):
    self.lower = lower
    self.upper = upper
    self.gain = gain
    super(XavierNormalTruncated, self).__init__()

  def __call__(self, shape, dtype=None):
    truncated_std = scipy.stats.truncnorm.std(a=self.lower, b=self.upper, loc=0., scale=1)
    fan_in, fan_out = np.prod(shape[:-1]), shape[-1]
    gain = np.sqrt(2 / (fan_in + fan_out))
    stddev = self.gain * gain / truncated_std
    res = math.random.truncated_normal(size=shape,
                                       scale=stddev,
                                       lower=self.lower,
                                       upper=self.upper)
    return math.asarray(res, dtype=dtype)


class TruncatedNormal(Initializer):
  """Returns a tensor with values assigned using truncated normal initialization.

    Args:
        shape: shape of the output tensor.
        lower: lower truncation of the normal.
        upper: upper truncation of the normal.
        stddev: expected standard deviation.

    Returns:
        Tensor initialized with truncated normal random variables with standard
        deviation stddev and support [lower, upper].
    """

  def __init__(self, lower=-2., upper=2., scale=1.):
    self.lower = lower
    self.upper = upper
    self.scale = scale
    super(TruncatedNormal, self).__init__()

  def __call__(self, shape, dtype=None):
    truncated_std = scipy.stats.truncnorm.std(a=self.lower, b=self.upper, loc=0., scale=1)
    res = math.random.truncated_normal(size=shape,
                                       scale=self.scale / truncated_std,
                                       lower=self.lower,
                                       upper=self.upper)
    return math.asarray(res, dtype=dtype)
