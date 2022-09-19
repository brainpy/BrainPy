# -*- coding: utf-8 -*-

import numpy as np

from brainpy import math as bm, tools
from .base import InterLayerInitializer


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


class Normal(InterLayerInitializer):
  """Initialize weights with normal distribution.

  Parameters
  ----------
  scale : float
    The gain of the derivation of the normal distribution.

  """

  def __init__(self, mean=0., scale=1., seed=None):
    super(Normal, self).__init__()
    self.scale = scale
    self.mean = mean
    self.seed = tools.format_seed(seed)
    self.rng = np.random.RandomState(seed=seed)

  def __call__(self, shape, dtype=None):
    shape = [tools.size2num(d) for d in shape]
    weights = self.rng.normal(size=shape, loc=self.mean, scale=self.scale)
    return bm.asarray(weights, dtype=dtype)

  def __repr__(self):
    return f'{self.__class__.__name__}(scale={self.scale}, seed={self.seed})'


class Uniform(InterLayerInitializer):
  """Initialize weights with uniform distribution.

  Parameters
  ----------
  min_val : float
    The lower limit of the uniform distribution.
  max_val : float
    The upper limit of the uniform distribution.
  """

  def __init__(self, min_val: float = 0., max_val: float = 1., seed=None):
    super(Uniform, self).__init__()
    self.min_val = min_val
    self.max_val = max_val
    self.seed = tools.format_seed(seed)
    self.rng = np.random.RandomState(seed=self.seed)

  def __call__(self, shape, dtype=None):
    shape = [tools.size2num(d) for d in shape]
    r = self.rng.uniform(low=self.min_val, high=self.max_val, size=shape)
    return bm.asarray(r, dtype=dtype)

  def __repr__(self):
    return (f'{self.__class__.__name__}(min_val={self.min_val}, '
            f'max_val={self.max_val}, seed={self.seed})')


class VarianceScaling(InterLayerInitializer):
  def __init__(
      self,
      scale: float,
      mode: str,
      distribution: str,
      in_axis: int = -2,
      out_axis: int = -1,
      seed: int = None
  ):
    self.scale = scale
    self.mode = mode
    self.in_axis = in_axis
    self.out_axis = out_axis
    self.distribution = distribution
    self.seed = tools.format_seed(seed)
    self.rng = np.random.RandomState(seed=self.seed)

  def __call__(self, shape, dtype=None):
    shape = [tools.size2num(d) for d in shape]
    fan_in, fan_out = _compute_fans(shape, in_axis=self.in_axis, out_axis=self.out_axis)
    if self.mode == "fan_in":
      denominator = fan_in
    elif self.mode == "fan_out":
      denominator = fan_out
    elif self.mode == "fan_avg":
      denominator = (fan_in + fan_out) / 2
    else:
      raise ValueError("invalid mode for variance scaling initializer: {}".format(self.mode))
    variance = np.array(self.scale / denominator, dtype=dtype)
    if self.distribution == "truncated_normal":
      stddev = bm.array(np.sqrt(variance) / .87962566103423978, dtype)
      rng = bm.random.RandomState(self.rng.randint(0, int(1e7)))
      return rng.truncated_normal(-2, 2, shape, dtype) * stddev
    elif self.distribution == "normal":
      res = self.rng.randn(*shape) * np.sqrt(variance)
    elif self.distribution == "uniform":
      res = self.rng.uniform(low=-1, high=1, size=shape) * np.sqrt(3 * variance)
    else:
      raise ValueError("invalid distribution for variance scaling initializer")
    return bm.asarray(res, dtype=dtype)

  def __repr__(self):
    name = self.__class__.__name__
    blank = ' ' * len(name)
    return (f'{name}(scale={self.scale}, mode={self.mode}, in_axis={self.in_axis}, \n'
            f'{blank}out_axis={self.out_axis}, distribution={self.distribution}, seed={self.seed})')


class KaimingUniform(VarianceScaling):
  def __init__(
      self,
      scale: float = 2.0,
      mode: str = "fan_in",
      distribution: str = "uniform",
      in_axis: int = -2,
      out_axis: int = -1,
      seed: int = None
  ):
    super(KaimingUniform, self).__init__(scale,
                                         mode,
                                         distribution,
                                         in_axis=in_axis,
                                         out_axis=out_axis,
                                         seed=seed)


class KaimingNormal(VarianceScaling):
  def __init__(
      self,
      scale: float = 2.0,
      mode: str = "fan_in",
      distribution: str = "truncated_normal",
      in_axis: int = -2,
      out_axis: int = -1,
      seed: int = None
  ):
    super(KaimingNormal, self).__init__(scale,
                                        mode,
                                        distribution,
                                        in_axis=in_axis,
                                        out_axis=out_axis,
                                        seed=seed)


class XavierUniform(VarianceScaling):
  def __init__(
      self,
      scale: float = 1.0,
      mode: str = "fan_avg",
      distribution: str = "uniform",
      in_axis: int = -2,
      out_axis: int = -1,
      seed: int = None
  ):
    super(XavierUniform, self).__init__(scale,
                                        mode,
                                        distribution,
                                        in_axis=in_axis,
                                        out_axis=out_axis,
                                        seed=seed)


class XavierNormal(VarianceScaling):
  def __init__(
      self,
      scale: float = 1.0,
      mode: str = "fan_avg",
      distribution: str = "truncated_normal",
      in_axis: int = -2,
      out_axis: int = -1,
      seed: int = None
  ):
    super(XavierNormal, self).__init__(scale,
                                       mode,
                                       distribution,
                                       in_axis=in_axis,
                                       out_axis=out_axis,
                                       seed=seed)


class LecunUniform(VarianceScaling):
  def __init__(
      self,
      scale: float = 1.0,
      mode: str = "fan_in",
      distribution: str = "uniform",
      in_axis: int = -2,
      out_axis: int = -1,
      seed: int = None
  ):
    super(LecunUniform, self).__init__(scale,
                                       mode,
                                       distribution,
                                       in_axis=in_axis,
                                       out_axis=out_axis,
                                       seed=seed)


class LecunNormal(VarianceScaling):
  def __init__(
      self,
      scale: float = 1.0,
      mode: str = "fan_in",
      distribution: str = "truncated_normal",
      in_axis: int = -2,
      out_axis: int = -1,
      seed: int = None
  ):
    super(LecunNormal, self).__init__(scale,
                                      mode,
                                      distribution,
                                      in_axis=in_axis,
                                      out_axis=out_axis,
                                      seed=seed)


class Orthogonal(InterLayerInitializer):
  """
  Construct an initializer for uniformly distributed orthogonal matrices.

  If the shape is not square, the matrix will have orthonormal rows or columns
  depending on which side is smaller.
  """

  def __init__(
      self,
      scale: float = 1.,
      axis: int = -1,
      seed: int = None
  ):
    super(Orthogonal, self).__init__()
    self.scale = scale
    self.axis = axis
    self.seed = tools.format_seed(seed)
    self.rng = np.random.RandomState(seed=self.seed)

  def __call__(self, shape, dtype=None):
    shape = [tools.size2num(d) for d in shape]
    n_rows = shape[self.axis]
    n_cols = np.prod(shape) // n_rows
    matrix_shape = (n_rows, n_cols) if n_rows > n_cols else (n_cols, n_rows)
    norm_dst = self.rng.normal(size=matrix_shape)
    q_mat, r_mat = np.linalg.qr(norm_dst)
    # Enforce Q is uniformly distributed
    q_mat *= np.sign(np.diag(r_mat))
    if n_rows < n_cols:
      q_mat = q_mat.T
    q_mat = np.reshape(q_mat, (n_rows,) + tuple(np.delete(shape, self.axis)))
    q_mat = np.moveaxis(q_mat, 0, self.axis)
    return self.scale * bm.asarray(q_mat, dtype=dtype)

  def __repr__(self):
    return f'{self.__class__.__name__}(scale={self.scale}, axis={self.axis}, seed={self.seed})'


class DeltaOrthogonal(InterLayerInitializer):
  """
  Construct an initializer for delta orthogonal kernels; see arXiv:1806.05393.

  The shape must be 3D, 4D or 5D.
  """

  def __init__(self, scale=1.0, axis=-1, ):
    super(DeltaOrthogonal, self).__init__()
    self.scale = scale
    self.axis = axis

  def __call__(self, shape, dtype=None):
    shape = [tools.size2num(d) for d in shape]
    if len(shape) not in [3, 4, 5]:
      raise ValueError("Delta orthogonal initializer requires a 3D, 4D or 5D shape.")
    if shape[-1] < shape[-2]:
      raise ValueError("`fan_in` must be less or equal than `fan_out`. ")
    ortho_init = Orthogonal(scale=self.scale, axis=self.axis)
    ortho_matrix = ortho_init(shape[-2:], dtype=dtype)
    W = bm.zeros(shape, dtype=dtype)
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

  def __repr__(self):
    return f'{self.__class__.__name__}(scale={self.scale}, axis={self.axis})'
