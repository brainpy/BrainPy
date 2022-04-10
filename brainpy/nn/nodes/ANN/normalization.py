# -*- coding: utf-8 -*-

from typing import Union

import jax.nn
import jax.numpy as jnp

import brainpy.math as bm
import brainpy
from brainpy.initialize import ZeroInit, OneInit, Initializer
from brainpy.nn.base import Node


__all__ = [
  'BatchNorm',
  'BatchNorm1d',
  'BatchNorm2d',
  'BatchNorm3d',
  'LayerNorm'
]


class BatchNorm(Node):
  """Batch Normalization node.
  Most commonly, the first axis of the data is the batch, and the last is
  the channel. However, users can specify the axes to be normalized.

  adapted from jax.example_libraries.stax.BatchNorm
  https://jax.readthedocs.io/en/latest/_modules/jax/example_libraries/stax.html#BatchNorm

  Parameters
  ----------
  axis: int, tuple, list
    axes where the data will be normalized. The axis of channels should be excluded.
  epsilon: float
    a value added to the denominator for numerical stability. Default: 1e-5
  use_bias: bool
    whether to translate data in refactoring
  use_scale: bool
    whether to scale data in refactoring
  beta_init: brainpy.init.Initializer
    an initializer generating the original translation matrix
  gamma_init: brainpy.init.Initializer
    an initializer generating the original scaling matrix
  """

  def __init__(self,
               axis: Union[int, tuple, list],
               epsilon: float = 1e-5,
               use_bias: bool = True,
               use_scale: bool = True,
               beta_init: Initializer = ZeroInit(),
               gamma_init: Initializer = OneInit(),
               **kwargs):
    super(BatchNorm, self).__init__(**kwargs)
    self.epsilon = epsilon
    self.bias = use_bias
    self.scale = use_scale
    self.beta_init = beta_init if use_bias else ()
    self.gamma_init = gamma_init if use_scale else ()
    self.axis = (axis,) if jnp.isscalar(axis) else axis

  def _check_input_dim(self):
    pass

  def init_ff_conn(self):
    self._check_input_dim()

    input_shape = tuple(d for i, d in enumerate(self.feedforward_shapes) if i not in self.axis)
    self.beta = bm.TrainVar(self.beta_init(input_shape)) if self.bias else None
    self.gamma = bm.TrainVar(self.gamma_init(input_shape)) if self.scale else None
    self.set_output_shape(self.feedforward_shapes)

  def forward(self, ff, **shared_kwargs):
    ed = tuple(None if i in self.axis else slice(None) for i in range(jnp.ndim(ff)))
    output = jax.nn.normalize(bm.as_device_array(ff), self.axis, epsilon=self.epsilon)
    if self.bias and self.scale: return self.gamma[ed] * output + self.beta[ed]
    if self.bias: return output + self.beta[ed]
    if self.scale: return self.gamma[ed] * output
    return output


class BatchNorm1d(BatchNorm):
  """1-D batch normalization.
  The data should be of `(b, l, c)`, where `b` is the batch dimension,
  `l` is the layer dimension, and `c` is the channel dimension, or of
  '(b, c)'.

  Parameters
  ----------
  axis: int, tuple, list
    axes where the data will be normalized. The axis of channels should be excluded.
  epsilon: float
    a value added to the denominator for numerical stability. Default: 1e-5
  use_bias: bool
    whether to translate data in refactoring
  use_scale: bool
    whether to scale data in refactoring
  beta_init: brainpy.init.Initializer
    an initializer generating the original translation matrix
  gamma_init: brainpy.init.Initializer
    an initializer generating the original scaling matrix
  """
  def __init__(self, axis=(0, 1), **kwargs):
    super(BatchNorm1d, self).__init__(axis=axis, **kwargs)

  def _check_input_dim(self):
    ndim = len(self.feedforward_shapes)
    if ndim != 2 and ndim != 3:
      raise ValueError(
        "expected 2D or 3D input (got {}D input)".format(ndim)
      )
    if ndim == 2 and len(self.axis) == 2:
      self.axis = (0,)


class BatchNorm2d(BatchNorm):
  """2-D batch normalization.
    The data should be of `(b, h, w, c)`, where `b` is the batch dimension,
    `h` is the height dimension, `w` is the width dimension, and `c` is the
    channel dimension.

    Parameters
    ----------
    axis: int, tuple, list
      axes where the data will be normalized. The axis of channels should be excluded.
    epsilon: float
      a value added to the denominator for numerical stability. Default: 1e-5
    use_bias: bool
      whether to translate data in refactoring
    use_scale: bool
      whether to scale data in refactoring
    beta_init: brainpy.init.Initializer
      an initializer generating the original translation matrix
    gamma_init: brainpy.init.Initializer
      an initializer generating the original scaling matrix
    """
  def __init__(self, axis=(0, 1, 2), **kwargs):
    super(BatchNorm2d, self).__init__(axis=axis, **kwargs)

  def _check_input_dim(self):
    ndim = len(self.feedforward_shapes)
    if ndim != 4:
      raise ValueError(
        "expected 4D input (got {}D input)".format(ndim)
      )


class BatchNorm3d(BatchNorm):
  """3-D batch normalization.
    The data should be of `(b, h, w, d, c)`, where `b` is the batch dimension,
    `h` is the height dimension, `w` is the width dimension, `d` is the depth
    dimension, and `c` is the channel dimension.

    Parameters
    ----------
    axis: int, tuple, list
      axes where the data will be normalized. The axis of channels should be excluded.
    epsilon: float
      a value added to the denominator for numerical stability. Default: 1e-5
    use_bias: bool
      whether to translate data in refactoring
    use_scale: bool
      whether to scale data in refactoring
    beta_init: brainpy.init.Initializer
      an initializer generating the original translation matrix
    gamma_init: brainpy.init.Initializer
      an initializer generating the original scaling matrix
      """
  def __init__(self, axis=(0, 1, 2, 3), **kwargs):
    super(BatchNorm3d, self).__init__(axis=axis, **kwargs)

  def _check_input_dim(self):
    ndim = len(self.feedforward_shapes)
    if ndim != 5:
      raise ValueError(
        "expected 5D input (got {}D input)".format(ndim)
      )


class LayerNorm(Node):
  def __init__(self,
               epsilon: float = 1e-6,
               use_bias: bool = True,
               use_scale: bool = True,
               beta_init: Initializer = ZeroInit(),
               gamma_init: Initializer = ZeroInit(),
               **kwargs):
    super(LayerNorm, self).__init__(**kwargs)
    self.epsilon = epsilon
    self.bias = use_bias
    self.scale = use_scale
    self.beta_init = beta_init if use_bias else ()
    self.gamma_init = gamma_init if use_scale else ()

  def init_ff_conn(self):
    self.axis = tuple(i for i in range(len(self.feedforward_shapes)) if i != 0)
    # todo: what if elementwise_affine = False?
    input_shape = tuple(d for i, d in enumerate(self.feedforward_shapes) if i in self.axis)
    self.beta = bm.TrainVar(self.beta_init(input_shape)) if self.bias else None
    self.gamma = bm.TrainVar(self.gamma_init(input_shape)) if self.scale else None
    self.set_output_shape(self.feedforward_shapes)

  def forward(self, ff, **shared_kwargs):
    # todo: wrong output shape
    ed = tuple(None if i in self.axis else slice(None) for i in range(jnp.ndim(ff)))
    output = jax.nn.normalize(bm.as_device_array(ff), self.axis, epsilon=self.epsilon)
    if self.bias and self.scale: return self.gamma[ed] * output + self.beta[ed]
    if self.bias: return output + self.beta[ed]
    if self.scale: return self.gamma[ed] * output
    return output
