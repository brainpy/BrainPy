# -*- coding: utf-8 -*-

import numpy as np

from brainpy import math
from .base import Initializer

__all__ = [
  'Normal',
  'Uniform',
  'VarianceScaling',
  'KaimingUniform',
  'KaimingNormal',
  'XavierUniform',
  'XavierNormal',
  'LecunUniform',
  'LecunNormal',
  'Orthogonal',
  'DeltaOrthogonal',
]


def _compute_fans(shape, in_axis=-2, out_axis=-1):
  receptive_field_size = np.prod(shape) / shape[in_axis] / shape[out_axis]
  fan_in = shape[in_axis] * receptive_field_size
  fan_out = shape[out_axis] * receptive_field_size
  return fan_in, fan_out


class Normal(Initializer):
  """Initialize weights with normal distribution.

  Parameters
  ----------
  scale : float
    The gain of the derivation of the normal distribution.

  """

  def __init__(self, scale=1., seed=None):
    super(Normal, self).__init__()
    self.scale = scale
    self.rng = math.random.RandomState(seed=seed)

  def __call__(self, shape, dtype=None):
    weights = self.rng.normal(size=shape, scale=self.scale)
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

  def __init__(self, min_val=0., max_val=1., seed=None):
    super(Uniform, self).__init__()
    self.min_val = min_val
    self.max_val = max_val
    self.rng = math.random.RandomState(seed=seed)

  def __call__(self, shape, dtype=None):
    r = self.rng.uniform(low=self.min_val, high=self.max_val, size=shape)
    return math.asarray(r, dtype=dtype)


class VarianceScaling(Initializer):
  def __init__(self, scale, mode, distribution, in_axis=-2, out_axis=-1, seed=None):
    self.scale = scale
    self.mode = mode
    self.in_axis = in_axis
    self.out_axis = out_axis
    self.distribution = distribution
    self.rng = math.random.RandomState(seed=seed)

  def __call__(self, shape, dtype=None):
    fan_in, fan_out = _compute_fans(shape, in_axis=self.in_axis, out_axis=self.out_axis)
    if self.mode == "fan_in":
      denominator = fan_in
    elif self.mode == "fan_out":
      denominator = fan_out
    elif self.mode == "fan_avg":
      denominator = (fan_in + fan_out) / 2
    else:
      raise ValueError("invalid mode for variance scaling initializer: {}".format(self.mode))
    variance = math.array(self.scale / denominator, dtype=dtype)
    if self.distribution == "truncated_normal":
      # constant is stddev of standard normal truncated to (-2, 2)
      stddev = math.sqrt(variance) / math.array(.87962566103423978, dtype)
      res = self.rng.truncated_normal(-2, 2, shape) * stddev
      return math.asarray(res, dtype=dtype)
    elif self.distribution == "normal":
      res = self.rng.normal(size=shape) * math.sqrt(variance)
      return math.asarray(res, dtype=dtype)
    elif self.distribution == "uniform":
      res = self.rng.uniform(low=-1, high=1, size=shape) * math.sqrt(3 * variance)
      return math.asarray(res, dtype=dtype)
    else:
      raise ValueError("invalid distribution for variance scaling initializer")


class KaimingUniform(VarianceScaling):
  def __init__(self, scale=2.0, mode="fan_in",
               distribution="uniform",
               in_axis=-2, out_axis=-1,
               seed=None):
    super(KaimingUniform, self).__init__(scale, mode, distribution,
                                         in_axis=in_axis, out_axis=out_axis,
                                         seed=seed)


class KaimingNormal(VarianceScaling):
  def __init__(self, scale=2.0, mode="fan_in",
               distribution="truncated_normal",
               in_axis=-2, out_axis=-1,
               seed=None):
    super(KaimingNormal, self).__init__(scale, mode, distribution,
                                        in_axis=in_axis, out_axis=out_axis,
                                        seed=seed)


class XavierUniform(VarianceScaling):
  def __init__(self, scale=1.0, mode="fan_avg",
               distribution="uniform",
               in_axis=-2, out_axis=-1,
               seed=None):
    super(XavierUniform, self).__init__(scale, mode, distribution,
                                        in_axis=in_axis, out_axis=out_axis,
                                        seed=seed)


class XavierNormal(VarianceScaling):
  def __init__(self, scale=1.0, mode="fan_avg",
               distribution="truncated_normal",
               in_axis=-2, out_axis=-1,
               seed=None):
    super(XavierNormal, self).__init__(scale, mode, distribution,
                                       in_axis=in_axis, out_axis=out_axis,
                                       seed=seed)


class LecunUniform(VarianceScaling):
  def __init__(self, scale=1.0, mode="fan_in",
               distribution="uniform",
               in_axis=-2, out_axis=-1,
               seed=None):
    super(LecunUniform, self).__init__(scale, mode, distribution,
                                       in_axis=in_axis,out_axis=out_axis,
                                       seed=seed)


class LecunNormal(VarianceScaling):
  def __init__(self, scale=1.0, mode="fan_in",
               distribution="truncated_normal",
               in_axis=-2, out_axis=-1,
               seed=None):
    super(LecunNormal, self).__init__(scale, mode, distribution,
                                      in_axis=in_axis, out_axis=out_axis,
                                      seed=seed)


class Orthogonal(Initializer):
  """
  Construct an initializer for uniformly distributed orthogonal matrices.

  If the shape is not square, the matrices will have orthonormal rows or columns
  depending on which side is smaller.
  """

  def __init__(self, scale=1., axis=-1, seed=None):
    super(Orthogonal, self).__init__()
    self.scale = scale
    self.axis = axis
    self.rng = math.random.RandomState(seed=seed)

  def __call__(self, shape, dtype=None):
    n_rows = shape[self.axis]
    n_cols = np.prod(shape) // n_rows
    matrix_shape = (n_rows, n_cols) if n_rows > n_cols else (n_cols, n_rows)
    norm_dst = self.rng.normal(size=matrix_shape)
    q_mat, r_mat = np.linalg.qr(norm_dst)
    # Enforce Q is uniformly distributed
    q_mat *= np.sign(np.diag(r_mat))
    if n_rows < n_cols: q_mat = q_mat.T
    q_mat = np.reshape(q_mat, (n_rows,) + tuple(np.delete(shape, self.axis)))
    q_mat = np.moveaxis(q_mat, 0, self.axis)
    return self.scale * math.asarray(q_mat, dtype=dtype)


class DeltaOrthogonal(Initializer):
  """
  Construct an initializer for delta orthogonal kernels; see arXiv:1806.05393.

  The shape must be 3D, 4D or 5D.
  """

  def __init__(self, scale=1.0, axis=-1, ):
    super(DeltaOrthogonal, self).__init__()
    self.scale = scale
    self.axis = axis

  def __call__(self, shape, dtype=None):
    if len(shape) not in [3, 4, 5]:
      raise ValueError("Delta orthogonal initializer requires a 3D, 4D or 5D shape.")
    if shape[-1] < shape[-2]:
      raise ValueError("`fan_in` must be less or equal than `fan_out`. ")
    ortho_init = Orthogonal(scale=self.scale, axis=self.axis)
    ortho_matrix = ortho_init(shape[-2:], dtype=dtype)
    W = math.zeros(shape, dtype=dtype)
    if len(shape) == 3:
      k = shape[0]
      W[(k - 1) // 2, ...] = ortho_matrix
    elif len(shape) == 4:
      k1, k2 = shape[:2]
      W[(k1 - 1) // 2, (k2 - 1) // 2, ...] = ortho_matrix
    else:
      k1, k2, k3 = shape[:3]
      W[(k1 - 1) // 2, (k2 - 1) // 2, (k3 - 1) // 2, ...] = ortho_matrix
    return W
