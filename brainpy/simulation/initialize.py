# -*- coding: utf-8 -*-

import abc

import numpy as np

from brainpy import math

__all__ = [
  'Initializer',
  'ZeroInit',
  'OneInit',
  'Identity',
  'Orthogonal',
  'KaimingNormal',
  'KaimingTruncatedNormal',
  'XavierNormal',
  'XavierTruncatedNormal',
  'TruncatedNormal',
]


class Initializer(abc.ABC):
  def __init__(self, dtype):
    self.dtype = math.float_ if dtype is None else dtype

  @abc.abstractmethod
  def __call__(self, *args, **kwargs):
    pass


def _gain_leaky_relu(relu_slope=0.1):
  """The recommended gain value for leaky_relu.

  Args:
      relu_slope: negative slope of leaky_relu.

  Returns:
      The recommended gain value for leaky_relu.
  """
  return np.sqrt(2 / (1 + relu_slope ** 2))


class ZeroInit(Initializer):
  def __call__(self, shape):
    return math.zeros(shape, dtype=self.dtype)


class OneInit(Initializer):
  def __init__(self, value=1., dtype=None):
    self.value = value
    super(OneInit, self).__init__(dtype=dtype)

  def __call__(self, shape):
    return math.ones(shape, dtype=self.dtype)


class Identity(Initializer):
  """Returns the identity matrix.

    This initializer was proposed in
    `A Simple Way to Initialize Recurrent Networks of Rectified Linear Units
    <https://arxiv.org/abs/1504.00941>`_.

    Args:
        shape: Shape of the tensor. It should have exactly rank 2.
        gain: optional scaling factor.

    Returns:
        Tensor initialized to the identity matrix.
    """

  def __init__(self, gain=1.):
    self.gain = gain
    super(Identity, self).__init__(dtype=self.dtype)

  def __call__(self, shape):
    return self.gain * math.eye(*shape, dtype=self.dtype)


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

  def __init__(self, gain=1., axis=-1, dtype=None):
    self.gain = gain
    self.axis = axis
    super(Orthogonal, self).__init__(dtype=dtype)

  def __call__(self, shape):
    n_rows = shape[self.axis]
    n_cols = np.prod(shape) // n_rows
    matrix_shape = (n_rows, n_cols) if n_rows > n_cols else (n_cols, n_rows)
    norm_dst = math.random.normal(size=matrix_shape)
    q_mat, r_mat = np.linalg.qr(norm_dst)
    # Enforce Q is uniformly distributed
    q_mat *= np.sign(np.diag(r_mat))
    if n_rows < n_cols:
      q_mat = q_mat.T
    q_mat = np.reshape(q_mat, (n_rows,) + tuple(np.delete(shape, self.axis)))
    q_mat = np.moveaxis(q_mat, 0, self.axis)
    return self.gain * math.asarray(q_mat, dtype=self.dtype)


class Normal(Initializer):
  def __init__(self, gain=1., dtype=None):
    self.gain = gain
    super(Normal, self).__init__(dtype=dtype)

  def __call__(self, shape):
    gain = np.sqrt(1 / np.prod(shape))
    res = math.random.normal(size=shape, scale=self.gain * gain)
    return math.asarray(res, dtype=res)


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

  def __init__(self, gain=1., dtype=None):
    self.gain = gain
    super(KaimingNormal, self).__init__(dtype=dtype)

  def __call__(self, shape, dtype=None):
    gain = np.sqrt(1 / np.prod(shape[:-1]))
    res = math.random.normal(size=shape, scale=self.gain * gain)
    return math.asarray(res, dtype=self.dtype)


class KaimingTruncatedNormal(Initializer):
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

  def __init__(self, lower=-2., upper=2., gain=1., dtype=None):
    self.lower = lower
    self.upper = upper
    self.gain = gain
    super(KaimingTruncatedNormal, self).__init__(dtype)

  def __call__(self, shape):
    truncated_std = scipy.stats.truncnorm.std(a=self.lower,
                                              b=self.upper,
                                              loc=0.,
                                              scale=1.)
    stddev = self.gain * np.sqrt(1 / np.prod(shape[:-1])) / truncated_std
    res = math.random.truncated_normal(size=shape,
                                       scale=stddev,
                                       lower=self.lower,
                                       upper=self.upper)
    return math.asarray(res, dtype=self.dtype)


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

  def __init__(self, gain=1., dtype=None):
    self.gain = gain
    super(XavierNormal, self).__init__(dtype=dtype)

  def __call__(self, shape, dtype=None):
    fan_in, fan_out = np.prod(shape[:-1]), shape[-1]
    gain = np.sqrt(2 / (fan_in + fan_out))
    res = math.random.normal(size=shape, scale=self.gain * gain)
    return math.asarray(res, dtype=self.dtype)


class XavierTruncatedNormal(Initializer):
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

  def __init__(self, lower=-2., upper=2., gain=1., dtype=None):
    self.lower = lower
    self.upper = upper
    self.gain = gain
    super(XavierTruncatedNormal, self).__init__(dtype=dtype)

  def __call__(self, shape, dtype=None):
    truncated_std = scipy.stats.truncnorm.std(a=self.lower, b=self.upper, loc=0., scale=1)
    fan_in, fan_out = np.prod(shape[:-1]), shape[-1]
    gain = np.sqrt(2 / (fan_in + fan_out))
    stddev = self.gain * gain / truncated_std
    res = math.random.truncated_normal(size=shape,
                                       scale=stddev,
                                       lower=self.lower,
                                       upper=self.upper)
    return math.asarray(res, dtype=self.dtype)


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

  def __init__(self, lower=-2., upper=2., scale=1., dtype=None):
    self.lower = lower
    self.upper = upper
    self.scale = scale
    super(TruncatedNormal, self).__init__(dtype=dtype)

  def __call__(self, shape, dtype=None):
    truncated_std = scipy.stats.truncnorm.std(a=self.lower, b=self.upper, loc=0., scale=1)
    res = math.random.truncated_normal(size=shape,
                                       scale=self.scale / truncated_std,
                                       lower=self.lower,
                                       upper=self.upper)
    return math.asarray(res, dtype=self.dtype)

# from scipy.stats import truncnorm
