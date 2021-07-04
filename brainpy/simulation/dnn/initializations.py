# -*- coding: utf-8 -*-

import numpy as np
import scipy.stats

from brainpy.simulation.dnn.imports import random, jnp, ndarray

__all__ = [
  'identity',
  'orthogonal',
  'kaiming_normal',
  'kaiming_truncated_normal',
  'xavier_normal',
  'xavier_truncated_normal',
  'truncated_normal',
]


def _gain_leaky_relu(relu_slope=0.1):
  """The recommended gain value for leaky_relu.

  Args:
      relu_slope: negative slope of leaky_relu.

  Returns:
      The recommended gain value for leaky_relu.
  """
  return np.sqrt(2 / (1 + relu_slope ** 2))


def _kaiming_normal_gain(shape):
  """Returns Kaiming He gain from
  `Delving Deep into Rectifiers: Surpassing Human-Level Performance on ImageNet Classification
  <https://arxiv.org/abs/1502.01852>`_.

  Args:
      shape: shape of the output tensor.

  Returns:
      Scalar, the standard deviation gain.
  """
  fan_in = np.prod(shape[:-1])
  return np.sqrt(1 / fan_in)


def _xavier_normal_gain(shape):
  """Returns Xavier Glorot gain from
  `Understanding the difficulty of training deep feedforward neural networks
  <http://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf>`_.

  Args:
      shape: shape of the output tensor.

  Returns:
      Scalar, the standard deviation gain.
  """
  fan_in, fan_out = np.prod(shape[:-1]), shape[-1]
  return np.sqrt(2 / (fan_in + fan_out))


def identity(shape, gain=1):
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
  assert len(shape) == 2
  return ndarray(gain * jnp.eye(*shape))


def orthogonal(shape, gain=1, axis=-1):
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
  n_rows = shape[axis]
  n_cols = np.prod(shape) // n_rows
  matrix_shape = (n_rows, n_cols) if n_rows > n_cols else (n_cols, n_rows)
  norm_dst = random.normal(matrix_shape)
  q_mat, r_mat = np.linalg.qr(norm_dst)
  # Enforce Q is uniformly distributed
  q_mat *= np.sign(np.diag(r_mat))
  if n_rows < n_cols:
    q_mat = q_mat.T
  q_mat = np.reshape(q_mat, (n_rows,) + tuple(np.delete(shape, axis)))
  q_mat = np.moveaxis(q_mat, 0, axis)
  return ndarray(gain * jnp.array(q_mat))


def kaiming_normal(shape, gain=1):
  """Returns a tensor with values assigned using Kaiming He normal initializer from
  `Delving Deep into Rectifiers: Surpassing Human-Level Performance on ImageNet Classification
  <https://arxiv.org/abs/1502.01852>`_.

  Args:
      shape: shape of the output tensor.
      gain: optional scaling factor.

  Returns:
      Tensor initialized with normal random variables with standard deviation (gain * kaiming_normal_gain).
  """
  return random.normal(shape, scale=gain * _kaiming_normal_gain(shape))


def kaiming_truncated_normal(shape, lower=-2, upper=2, gain=1):
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
  truncated_std = scipy.stats.truncnorm.std(a=lower, b=upper, loc=0., scale=1)
  stddev = gain * _kaiming_normal_gain(shape) / truncated_std
  return random.truncated_normal(size=shape, scale=stddev, lower=lower, upper=upper)


def xavier_normal(shape, gain=1.):
  """Returns a tensor with values assigned using Xavier Glorot normal initializer from
  `Understanding the difficulty of training deep feedforward neural networks
  <http://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf>`_.

  Args:
      shape: shape of the output tensor.
      gain: optional scaling factor.

  Returns:
      Tensor initialized with normal random variables with standard deviation (gain * xavier_normal_gain).
  """
  return random.normal(shape, scale=gain * _xavier_normal_gain(shape))


def xavier_truncated_normal(shape, lower=-2, upper=2, gain=1):
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
  truncated_std = scipy.stats.truncnorm.std(a=lower, b=upper, loc=0., scale=1)
  stddev = gain * _xavier_normal_gain(shape) / truncated_std
  return random.truncated_normal(size=shape, scale=stddev, lower=lower, upper=upper)


def truncated_normal(shape, lower=-2, upper=2, stddev=1):
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
  truncated_std = scipy.stats.truncnorm.std(a=lower, b=upper, loc=0., scale=1)
  stddev /= truncated_std
  return random.truncated_normal(size=shape, scale=stddev, lower=lower, upper=upper)
