# -*- coding: utf-8 -*-

from typing import Union

import jax.nn
import jax.numpy as jnp
import jax.lax

import brainpy.math as bm
from brainpy.initialize import ZeroInit, OneInit, Initializer, parameter
from brainpy.dyn.base import DynamicalSystem
from brainpy.modes import Mode, TrainingMode, NormalMode, training, check_mode

__all__ = [
  'BatchNorm',
  'BatchNorm1d',
  'BatchNorm2d',
  'BatchNorm3d',
  'GroupNorm',
  'LayerNorm',
  'InstanceNorm',
]


class BatchNorm(DynamicalSystem):
  """Batch Normalization node.
  This layer aims to reduce the internal covariant shift of data. It
  normalizes a batch of data by fixing the mean and variance of inputs
  on each feature (channel). Most commonly, the first axis of the data
  is the batch, and the last is the channel. However, users can specify
  the axes to be normalized.

  adapted from jax.example_libraries.stax.BatchNorm
  https://jax.readthedocs.io/en/latest/_modules/jax/example_libraries/stax.html#BatchNorm

  Parameters
  ----------
  axis: int, tuple, list
    axes where the data will be normalized. The feature (channel) axis should be excluded.
  epsilon: float
    a value added to the denominator for numerical stability. Default: 1e-5
  use_bias: bool
    whether to translate data in refactoring. Default: True
  use_scale: bool
    whether to scale data in refactoring. Default: True
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
               mode: Mode = training,
               name: str = None,
               **kwargs):
    super(BatchNorm, self).__init__(name=name, mode=mode)
    self.epsilon = epsilon
    self.bias = use_bias
    self.scale = use_scale
    self.beta_init = beta_init if use_bias else ()
    self.gamma_init = gamma_init if use_scale else ()
    self.axis = (axis,) if jnp.isscalar(axis) else axis

  def _check_input_dim(self, x):
    pass

  def update(self, sha, x):
    self._check_input_dim(x)

    input_shape = tuple(d for i, d in enumerate(x.shape) if i not in self.axis)
    self.beta = parameter(self.beta_init, input_shape) if self.bias else None
    self.gamma = parameter(self.gamma_init, input_shape) if self.scale else None
    if isinstance(self.mode, TrainingMode):
      self.beta = bm.TrainVar(self.beta)
      self.gamma = bm.TrainVar(self.gamma)

    ed = tuple(None if i in self.axis else slice(None) for i in range(jnp.ndim(x)))
    # output = bm.normalize(x, self.axis, epsilon=self.epsilon)
    print(x)
    output = jax.nn.standardize(x.value, self.axis, epsilon=self.epsilon)
    print(output)
    if self.bias and self.scale: return self.gamma[ed] * output + self.beta[ed]
    if self.bias: return output + self.beta[ed]
    if self.scale: return self.gamma[ed] * output
    return output

  def reset_state(self, batch_size=None):
    pass


class BatchNorm1d(BatchNorm):
  """1-D batch normalization.
  The data should be of `(b, l, c)`, where `b` is the batch dimension,
  `l` is the layer dimension, and `c` is the channel dimension, or of
  '(b, c)'.

  Parameters
  ----------
  axis: int, tuple, list
    axes where the data will be normalized. The feature (channel) axis should be excluded.
  epsilon: float
    a value added to the denominator for numerical stability. Default: 1e-5
  use_bias: bool
    whether to translate data in refactoring. Default: True
  use_scale: bool
    whether to scale data in refactoring. Default: True
  beta_init: brainpy.init.Initializer
    an initializer generating the original translation matrix
  gamma_init: brainpy.init.Initializer
    an initializer generating the original scaling matrix
  """
  def __init__(self, axis=(0, 1), **kwargs):
    super(BatchNorm1d, self).__init__(axis=axis, **kwargs)

  def _check_input_dim(self, x):
    ndim = len(x.shape)
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
      axes where the data will be normalized. The feature (channel) axis should be excluded.
    epsilon: float
      a value added to the denominator for numerical stability. Default: 1e-5
    use_bias: bool
      whether to translate data in refactoring. Default: True
    use_scale: bool
      whether to scale data in refactoring. Default: True
    beta_init: brainpy.init.Initializer
      an initializer generating the original translation matrix
    gamma_init: brainpy.init.Initializer
      an initializer generating the original scaling matrix
    """
  def __init__(self, axis=(0, 1, 2), **kwargs):
    super(BatchNorm2d, self).__init__(axis=axis, **kwargs)

  def _check_input_dim(self, x):
    ndim = len(x.shape)
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
      axes where the data will be normalized. The feature (channel) axis should be excluded.
    epsilon: float
      a value added to the denominator for numerical stability. Default: 1e-5
    use_bias: bool
      whether to translate data in refactoring. Default: True
    use_scale: bool
      whether to scale data in refactoring. Default: True
    beta_init: brainpy.init.Initializer
      an initializer generating the original translation matrix
    gamma_init: brainpy.init.Initializer
      an initializer generating the original scaling matrix
      """
  def __init__(self, axis=(0, 1, 2, 3), **kwargs):
    super(BatchNorm3d, self).__init__(axis=axis, **kwargs)

  def _check_input_dim(self, x):
    ndim = len(x.shape)
    if ndim != 5:
      raise ValueError(
        "expected 5D input (got {}D input)".format(ndim)
      )


class LayerNorm(DynamicalSystem):
  """Layer normalization (https://arxiv.org/abs/1607.06450).

  This layer normalizes data on each example, independently of the batch. More
  specifically, it normalizes data of shape (b, d1, d2, ..., c) on the axes of
  the data dimensions and the channel (d1, d2, ..., c). Different from batch
  normalization, gamma and beta are assigned to each position (elementwise
  operation) instead of the whole channel. If users want to assign a single
  gamma and beta to a whole example/whole channel, please use GroupNorm/
  InstanceNorm.

  Parameters
  ----------
  epsilon: float
    a value added to the denominator for numerical stability. Default: 1e-5
  use_bias: bool
    whether to translate data in refactoring. Default: True
  use_scale: bool
    whether to scale data in refactoring. Default: True
  beta_init: brainpy.init.Initializer
    an initializer generating the original translation matrix
  gamma_init: brainpy.init.Initializer
    an initializer generating the original scaling matrix
  axis: int, tuple, list
    axes where the data will be normalized. The batch axis should be excluded.
  """
  def __init__(self,
               epsilon: float = 1e-5,
               use_bias: bool = True,
               use_scale: bool = True,
               beta_init: Initializer = ZeroInit(),
               gamma_init: Initializer = OneInit(),
               axis: Union[int, tuple] = None,
               mode: Mode = training,
               name: str = None,
               **kwargs):
    super(LayerNorm, self).__init__(name=name, mode=mode)
    self.epsilon = epsilon
    self.bias = use_bias
    self.scale = use_scale
    self.beta_init = beta_init if use_bias else ()
    self.gamma_init = gamma_init if use_scale else ()
    self.axis = (axis,) if jnp.isscalar(axis) else axis

  def default_axis(self, x):
    # default: the first axis (batch dim) is excluded
    return tuple(i for i in range(1, len(x.shape)))

  def update(self, sha, x):
    if self.axis is None:
      self.axis = self.default_axis(x)
    # todo: what if elementwise_affine = False?
    input_shape = tuple(d for i, d in enumerate(x.shape) if i in self.axis)
    self.beta = parameter(self.beta_init, input_shape) if self.bias else None
    self.gamma = parameter(self.gamma_init, input_shape) if self.scale else None
    if isinstance(self.mode, TrainingMode):
      self.beta = bm.TrainVar(self.beta)
      self.gamma = bm.TrainVar(self.gamma)

    ed = tuple(None if i not in self.axis else slice(None) for i in range(jnp.ndim(x)))
    output = bm.normalize(x, self.axis, epsilon=self.epsilon)
    if self.bias and self.scale: return self.gamma[ed] * output + self.beta[ed]
    if self.bias: return output + self.beta[ed]
    if self.scale: return self.gamma[ed] * output
    return output

  def reset_state(self, batch_size=None):
    pass


class GroupNorm(DynamicalSystem):
  """Group normalization layer.

  This layer divides channels into groups and normalizes the features within each
  group. Its computation is also independent of the batch size. The feature size
  must be multiple of the group size.

  The shape of the data should be (b, d1, d2, ..., c), where `d` denotes the batch
  size and `c` denotes the feature (channel) size. The `d` and `c` axis should be
  excluded in parameter `axis`.

  Parameters
  ----------
  num_groups: int
    the number of groups. It should be a factor of the number of features.
  group_size: int
    the group size. It should equal to int(num_features / num_groups).
    Either `num_groups` or `group_size` should be specified.
  epsilon: float
    a value added to the denominator for numerical stability. Default: 1e-5
  use_bias: bool
    whether to translate data in refactoring. Default: True
  use_scale: bool
    whether to scale data in refactoring. Default: True
  beta_init: brainpy.init.Initializer
    an initializer generating the original translation matrix
  gamma_init: brainpy.init.Initializer
    an initializer generating the original scaling matrix
  axis: int, tuple, list
    axes where the data will be normalized. Besides the batch axis, the channel
    axis should be also excluded, since it will be automatically added to `axis`.
  """
  def __init__(self,
               num_groups: int = None,
               group_size: int = None,
               epsilon: float = 1e-5,
               use_bias: bool = True,
               use_scale: bool = True,
               beta_init: Initializer = ZeroInit(),
               gamma_init: Initializer = OneInit(),
               axis: Union[int, tuple] = None,
               mode: Mode = training,
               name: str = None,
               **kwargs):
    super(GroupNorm, self).__init__(name=name, mode=mode)
    self.num_groups = num_groups
    self.group_size = group_size
    self.epsilon = epsilon
    self.bias = use_bias
    self.scale = use_scale
    self.beta_init = beta_init if use_bias else ()
    self.gamma_init = gamma_init if use_scale else ()
    self.norm_axis = (axis,) if jnp.isscalar(axis) else axis

  def update(self, sha, x):
    num_channels = x.shape[-1]
    self.ndim = len(x)

    # compute num_groups and group_size
    if ((self.num_groups is None and self.group_size is None) or
        (self.num_groups is not None and self.group_size is not None)):
      raise ValueError('Either `num_groups` or `group_size` should be specified. '
                       'Once one is specified, the other will be automatically '
                       'computed.')

    if self.num_groups is None:
      assert self.group_size > 0, '`group_size` should be a positive integer.'
      if num_channels % self.group_size != 0:
        raise ValueError('The number of channels ({}) is not multiple of the '
                         'group size ({}).'.format(num_channels, self.group_size))
      else:
        self.num_groups = num_channels // self.group_size
    else:  # self.num_groups is not None:
      assert self.num_groups > 0, '`num_groups` should be a positive integer.'
      if num_channels % self.num_groups != 0:
        raise ValueError('The number of channels ({}) is not multiple of the '
                         'number of groups ({}).'.format(num_channels, self.num_groups))
      else:
        self.group_size = num_channels // self.num_groups

    # axes for normalization
    if self.norm_axis is None:
      # default: the first axis (batch dim) and the second-last axis (num_group dim) are excluded
      self.norm_axis = tuple(i for i in range(1, len(x.shape) - 1)) + (self.ndim,)

    group_shape = x.shape[:-1] + (self.num_groups, self.group_size)
    input_shape = tuple(d for i, d in enumerate(group_shape) if i in self.norm_axis)
    self.beta = parameter(self.beta_init, input_shape) if self.bias else None
    self.gamma = parameter(self.gamma_init, input_shape) if self.scale else None
    if isinstance(self.mode, TrainingMode):
      self.beta = bm.TrainVar(self.beta)
      self.gamma = bm.TrainVar(self.gamma)

    group_shape = x.shape[:-1] + (self.num_groups, self.group_size)
    ff_reshape = x.reshape(group_shape)
    ed = tuple(None if i not in self.norm_axis else slice(None) for i in range(jnp.ndim(ff_reshape)))
    output = bm.normalize(ff_reshape, self.norm_axis, epsilon=self.epsilon)
    if self.bias and self.scale:
      output = self.gamma[ed] * output + self.beta[ed]
    elif self.bias:
      output = output + self.beta[ed]
    elif self.scale:
      output = self.gamma[ed] * output
    return output.reshape(x.shape)


class InstanceNorm(GroupNorm):
  """Instance normalization layer.

  This layer normalizes the data within each feature. It can be regarded as
  a group normalization layer in which `group_size` equals to 1.

  Parameters
  ----------
  epsilon: float
    a value added to the denominator for numerical stability. Default: 1e-5
  use_bias: bool
    whether to translate data in refactoring. Default: True
  use_scale: bool
    whether to scale data in refactoring. Default: True
  beta_init: brainpy.init.Initializer
    an initializer generating the original translation matrix
  gamma_init: brainpy.init.Initializer
    an initializer generating the original scaling matrix
  axis: int, tuple, list
    axes where the data will be normalized. The batch and channel axes
    should be excluded.
  """
  def __init__(self,
               epsilon: float = 1e-5,
               use_bias: bool = True,
               use_scale: bool = True,
               beta_init: Initializer = ZeroInit(),
               gamma_init: Initializer = OneInit(),
               axis: Union[int, tuple] = None,
               **kwargs):
    super(InstanceNorm, self).__init__(group_size=1, epsilon=epsilon, use_bias=use_bias,
                                       use_scale=use_scale, beta_init=beta_init,
                                       gamma_init=gamma_init, axis=axis, **kwargs)