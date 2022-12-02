# -*- coding: utf-8 -*-

from typing import Union, Optional, Sequence

from jax import lax, numpy as jnp

import brainpy.math as bm
from brainpy.dyn.base import DynamicalSystem
from brainpy.initialize import ZeroInit, OneInit, Initializer, parameter
from brainpy.modes import Mode, TrainingMode, training

__all__ = [
  'BatchNorm1D',
  'BatchNorm2D',
  'BatchNorm3D',

  'LayerNorm',
  'GroupNorm',
  'InstanceNorm',
]


def _abs_sq(x):
  """Computes the elementwise square of the absolute value |x|^2."""
  if jnp.iscomplexobj(x):
    return lax.square(lax.real(x)) + lax.square(lax.imag(x))
  else:
    return lax.square(x)


class BatchNorm(DynamicalSystem):
  """Batch Normalization layer.

  This layer aims to reduce the internal covariant shift of data. It
  normalizes a batch of data by fixing the mean and variance of inputs
  on each feature (channel). Most commonly, the first axis of the data
  is the batch, and the last is the channel. However, users can specify
  the axes to be normalized.

  Parameters
  ----------
  num_features: int
    ``C`` from an expected input of size ``(..., C)``.
  axis: int, tuple, list
    Axes where the data will be normalized. The feature (channel) axis should be excluded.
  epsilon: float
    A value added to the denominator for numerical stability. Default: 1e-5
  affine: bool
    A boolean value that when set to ``True``, this module has
    learnable affine parameters. Default: ``True``
  bias_init: Initializer
    An initializer generating the original translation matrix
  scale_init: Initializer
    An initializer generating the original scaling matrix
  """

  def __init__(
      self,
      num_features: int,
      axis: Union[int, Sequence[int]],
      epsilon: float = 1e-5,
      momentum: Optional[float] = 0.99,
      affine: bool = True,
      bias_init: Initializer = ZeroInit(),
      scale_init: Initializer = OneInit(),
      mode: Mode = training,
      name: str = None,
  ):
    super(BatchNorm, self).__init__(name=name, mode=mode)

    # parameters
    self.num_features = num_features
    self.epsilon = epsilon
    self.momentum = momentum
    self.affine = affine
    self.bias_init = bias_init
    self.scale_init = scale_init
    self.axis = (axis,) if jnp.isscalar(axis) else axis

    # variables
    self.running_mean = bm.Variable(bm.zeros(self.num_features))
    self.running_var = bm.Variable(bm.ones(self.num_features))
    if self.affine:
      assert isinstance(self.mode, TrainingMode)
      self.bias = bm.TrainVar(parameter(self.bias_init, self.num_features))
      self.scale = bm.TrainVar(parameter(self.scale_init, self.num_features))

  def _check_input_dim(self, x):
    raise NotImplementedError

  def update(self, sha, x):
    self._check_input_dim(x)

    if sha['fit']:
      mean = bm.mean(x, self.axis)
      mean2 = bm.mean(_abs_sq(x), self.axis)
      var = jnp.maximum(0., mean2 - _abs_sq(mean))
      self.running_mean.value = (self.momentum * self.running_mean.value +
                                 (1 - self.momentum) * mean)
      self.running_var.value = (self.momentum * self.running_var.value +
                                (1 - self.momentum) * var)
    else:
      mean = self.running_mean.value
      var = self.running_var.value
    stats_shape = [(1 if i in self.axis else x.shape[i]) for i in range(x.ndim)]
    mean = mean.reshape(stats_shape)
    var = var.reshape(stats_shape)

    y = x - mean
    mul = lax.rsqrt(var + lax.convert_element_type(self.epsilon, x.dtype))
    if self.affine:
      mul *= self.scale
    y *= mul
    if self.affine:
      y += self.bias
    return y

  def reset_state(self, batch_size=None):
    pass


class BatchNorm1D(BatchNorm):
  """1-D batch normalization.

  The data should be of `(b, l, c)`, where `b` is the batch dimension,
  `l` is the layer dimension, and `c` is the channel dimension.

  Parameters
  ----------
  num_features: int
    ``C`` from an expected input of size ``(B, L, C)``.
  axis: int, tuple, list
    axes where the data will be normalized. The feature (channel) axis should be excluded.
  epsilon: float
    A value added to the denominator for numerical stability. Default: 1e-5
  affine: bool
    A boolean value that when set to ``True``, this module has
    learnable affine parameters. Default: ``True``
  bias_init: Initializer
    an initializer generating the original translation matrix
  scale_init: Initializer
    an initializer generating the original scaling matrix
  """

  def __init__(
      self,
      num_features: int,
      axis: Union[int, Sequence[int]] = (0, 1),
      epsilon: float = 1e-5,
      momentum: Optional[float] = 0.99,
      affine: bool = True,
      bias_init: Initializer = ZeroInit(),
      scale_init: Initializer = OneInit(),
      mode: Mode = training,
      name: str = None,
  ):
    super(BatchNorm1D, self).__init__(num_features=num_features,
                                      axis=axis,
                                      epsilon=epsilon,
                                      momentum=momentum,
                                      affine=affine,
                                      bias_init=bias_init,
                                      scale_init=scale_init,
                                      mode=mode,
                                      name=name)

  def _check_input_dim(self, x):
    if x.ndim != 3:
      raise ValueError(f"expected 3D input (got {x.ndim}D input)")
    assert x.shape[-1] == self.num_features


class BatchNorm2D(BatchNorm):
  """2-D batch normalization.

  The data should be of `(b, h, w, c)`, where `b` is the batch dimension,
  `h` is the height dimension, `w` is the width dimension, and `c` is the
  channel dimension.

  Parameters
  ----------
  num_features: int
    ``C`` from an expected input of size ``(B, H, W, C)``.
  axis: int, tuple, list
    axes where the data will be normalized. The feature (channel) axis should be excluded.
  epsilon: float
    a value added to the denominator for numerical stability. Default: 1e-5
  affine: bool
    A boolean value that when set to ``True``, this module has
    learnable affine parameters. Default: ``True``
  bias_init: Initializer
    an initializer generating the original translation matrix
  scale_init: Initializer
    an initializer generating the original scaling matrix
  """

  def __init__(
      self,
      num_features: int,
      axis: Union[int, Sequence[int]] = (0, 1, 2),
      epsilon: float = 1e-5,
      momentum: Optional[float] = 0.99,
      affine: bool = True,
      bias_init: Initializer = ZeroInit(),
      scale_init: Initializer = OneInit(),
      mode: Mode = training,
      name: str = None,
  ):
    super(BatchNorm2D, self).__init__(num_features=num_features,
                                      axis=axis,
                                      epsilon=epsilon,
                                      momentum=momentum,
                                      affine=affine,
                                      bias_init=bias_init,
                                      scale_init=scale_init,
                                      mode=mode,
                                      name=name)

  def _check_input_dim(self, x):
    if x.ndim != 4:
      raise ValueError(f"expected 4D input (got {x.ndim}D input)")
    assert x.shape[-1] == self.num_features


class BatchNorm3D(BatchNorm):
  """3-D batch normalization.

  The data should be of `(b, h, w, d, c)`, where `b` is the batch dimension,
  `h` is the height dimension, `w` is the width dimension, `d` is the depth
  dimension, and `c` is the channel dimension.

  Parameters
  ----------
  num_features: int
    ``C`` from an expected input of size ``(B, H, W, D, C)``.
  axis: int, tuple, list
    axes where the data will be normalized. The feature (channel) axis should be excluded.
  epsilon: float
    a value added to the denominator for numerical stability. Default: 1e-5
  affine: bool
    A boolean value that when set to ``True``, this module has
    learnable affine parameters. Default: ``True``
  bias_init: Initializer
    an initializer generating the original translation matrix
  scale_init: Initializer
    an initializer generating the original scaling matrix
  """

  def __init__(
      self,
      num_features: int,
      axis: Union[int, Sequence[int]] = (0, 1, 2, 3),
      epsilon: float = 1e-5,
      momentum: Optional[float] = 0.99,
      affine: bool = True,
      bias_init: Initializer = ZeroInit(),
      scale_init: Initializer = OneInit(),
      mode: Mode = training,
      name: str = None,
  ):
    super(BatchNorm3D, self).__init__(num_features=num_features,
                                      axis=axis,
                                      epsilon=epsilon,
                                      momentum=momentum,
                                      affine=affine,
                                      bias_init=bias_init,
                                      scale_init=scale_init,
                                      mode=mode,
                                      name=name)

  def _check_input_dim(self, x):
    if x.ndim != 5:
      raise ValueError(f"expected 5D input (got {x.ndim}D input)")
    assert x.shape[-1] == self.num_features


class LayerNorm(DynamicalSystem):
  """Layer normalization (https://arxiv.org/abs/1607.06450).

  .. math::
      y = \frac{x - \mathrm{E}[x]}{ \sqrt{\mathrm{Var}[x] + \epsilon}} * \gamma + \beta

  This layer normalizes data on each example, independently of the batch. More
  specifically, it normalizes data of shape (b, d1, d2, ..., c) on the axes of
  the data dimensions and the channel (d1, d2, ..., c). Different from batch
  normalization, scale and bias are assigned to each position (elementwise
  operation) instead of the whole channel. If users want to assign a single
  scale and bias to a whole example/whole channel, please use GroupNorm/
  InstanceNorm.

  Parameters
  ----------
  normalized_shape: int, sequence of int
    The input shape from an expected input of size

    .. math::
        [* \times \text{normalized\_shape}[0] \times \text{normalized\_shape}[1]
            \times \ldots \times \text{normalized\_shape}[-1]]

    If a single integer is used, it is treated as a singleton list, and this module will
    normalize over the last dimension which is expected to be of that specific size.
  epsilon: float
    a value added to the denominator for numerical stability. Default: 1e-5
  bias_init: Initializer
    an initializer generating the original translation matrix
  scale_init: Initializer
    an initializer generating the original scaling matrix
  elementwise_affine: bool
    A boolean value that when set to ``True``, this module
    has learnable per-element affine parameters initialized to ones (for weights)
    and zeros (for biases). Default: ``True``.

  Examples
  --------
  >>> import brainpy as bp
  >>> import brainpy.math as bm
  >>>
  >>> # NLP Example
  >>> batch, sentence_length, embedding_dim = 20, 5, 10
  >>> embedding = bm.random.randn(batch, sentence_length, embedding_dim)
  >>> layer_norm = bp.layers.LayerNorm(embedding_dim)
  >>> # Activate module
  >>> layer_norm(embedding)
  >>>
  >>> # Image Example
  >>> N, C, H, W = 20, 5, 10, 10
  >>> input = bm.random.randn(N, H, W, C)
  >>> # Normalize over the last three dimensions (i.e. the channel and spatial dimensions)
  >>> # as shown in the image below
  >>> layer_norm = bp.layers.LayerNorm([H, W, C])
  >>> output = layer_norm(input)

  """

  def __init__(
      self,
      normalized_shape: Union[int, Sequence[int]],
      epsilon: float = 1e-5,
      bias_init: Initializer = ZeroInit(),
      scale_init: Initializer = OneInit(),
      elementwise_affine: bool = True,
      mode: Mode = training,
      name: str = None
  ):
    super(LayerNorm, self).__init__(name=name, mode=mode)

    self.epsilon = epsilon
    self.bias_init = bias_init
    self.scale_init = scale_init
    if isinstance(normalized_shape, int):
      normalized_shape = (normalized_shape, )
    self.normalized_shape = tuple(normalized_shape)
    assert all([isinstance(s, int) for s in normalized_shape]), 'Must be a sequence of integer.'
    self.elementwise_affine = elementwise_affine
    if self.elementwise_affine:
      assert isinstance(self.mode, TrainingMode)
      self.bias = bm.TrainVar(parameter(self.bias_init, self.normalized_shape))
      self.scale = bm.TrainVar(parameter(self.scale_init, self.normalized_shape))

  def update(self, sha, x):
    if x.shape[-len(self.normalized_shape):] != self.normalized_shape:
      raise ValueError(f'Expect the input shape should be (..., {", ".join(self.normalized_shape)}), '
                       f'but we got {x.shape}')
    axis = tuple(range(0, x.ndim - len(self.normalized_shape)))
    mean = jnp.mean(bm.as_jax(x), axis=axis, keepdims=True)
    variance = jnp.var(bm.as_jax(x), axis=axis, keepdims=True)
    inv = lax.rsqrt(variance + lax.convert_element_type(self.epsilon, x.dtype))
    out = (x - mean) * inv
    if self.elementwise_affine:
      out = self.scale * out + self.bias
    return out

  def reset_state(self, batch_size=None):
    pass


class GroupNorm(DynamicalSystem):
  """Group normalization layer.

  .. math::
    y = \frac{x - \mathrm{E}[x]}{ \sqrt{\mathrm{Var}[x] + \epsilon}} * \gamma + \beta


  This layer divides channels into groups and normalizes the features within each
  group. Its computation is also independent of the batch size. The feature size
  must be multiple of the group size.

  The shape of the data should be (b, d1, d2, ..., c), where `d` denotes the batch
  size and `c` denotes the feature (channel) size.

  Parameters
  ----------
  num_groups: int
    The number of groups. It should be a factor of the number of channels.
  num_channels: int
    The number of channels expected in input.
  epsilon: float
    a value added to the denominator for numerical stability. Default: 1e-5
  affine: bool
    A boolean value that when set to ``True``, this module
    has learnable per-channel affine parameters initialized to ones (for weights)
    and zeros (for biases). Default: ``True``.
  bias_init: Initializer
    An initializer generating the original translation matrix
  scale_init: Initializer
    An initializer generating the original scaling matrix

  Examples
  --------
  >>> import brainpy as bp
  >>> import brainpy.math as bm
  >>> input = bm.random.randn(20, 10, 10, 6)
  >>> # Separate 6 channels into 3 groups
  >>> m = bp.layers.GroupNorm(3, 6)
  >>> # Separate 6 channels into 6 groups (equivalent with InstanceNorm)
  >>> m = bp.layers.GroupNorm(6, 6)
  >>> # Put all 6 channels into a single group (equivalent with LayerNorm)
  >>> m = bp.layers.GroupNorm(1, 6)
  >>> # Activating the module
  >>> output = m(input)
  """

  def __init__(
      self,
      num_groups: int,
      num_channels: int,
      epsilon: float = 1e-5,
      affine: bool = True,
      bias_init: Initializer = ZeroInit(),
      scale_init: Initializer = OneInit(),
      mode: Mode = training,
      name: str = None,
  ):
    super(GroupNorm, self).__init__(name=name, mode=mode)
    if num_channels % num_groups != 0:
      raise ValueError('num_channels must be divisible by num_groups')
    self.num_groups = num_groups
    self.num_channels = num_channels
    self.epsilon = epsilon
    self.affine = affine
    self.bias_init = bias_init
    self.scale_init = scale_init
    if self.affine:
      assert isinstance(self.mode, TrainingMode)
      self.bias = bm.TrainVar(parameter(self.bias_init, self.num_channels))
      self.scale = bm.TrainVar(parameter(self.scale_init, self.num_channels))

  def update(self, sha, x):
    assert x.shape[-1] == self.num_channels
    origin_shape, origin_dim = x.shape, x.ndim
    group_shape = (-1,) + x.shape[1:-1] + (self.num_groups, self.num_channels // self.num_groups)
    x = bm.as_jax(x.reshape(group_shape))
    reduction_axes = tuple(range(1, x.ndim - 1)) + (-1,)
    mean = jnp.mean(x, reduction_axes, keepdims=True)
    var = jnp.var(x, reduction_axes, keepdims=True)
    x = (x - mean) * lax.rsqrt(var + lax.convert_element_type(self.epsilon, x.dtype))
    x = x.reshape(origin_shape)
    if self.affine:
      x = x * lax.broadcast_to_rank(self.scale, origin_dim)
      x = x + lax.broadcast_to_rank(self.bias, origin_dim)
    return x


class InstanceNorm(GroupNorm):
  """Instance normalization layer.

  This layer normalizes the data within each feature. It can be regarded as
  a group normalization layer in which `group_size` equals to 1.

  Parameters
  ----------
  num_channels: int
    The number of channels expected in input.
  epsilon: float
    a value added to the denominator for numerical stability. Default: 1e-5
  affine: bool
    A boolean value that when set to ``True``, this module
    has learnable per-channel affine parameters initialized to ones (for weights)
    and zeros (for biases). Default: ``True``.
  bias_init: Initializer
    an initializer generating the original translation matrix
  scale_init: Initializer
    an initializer generating the original scaling matrix
  """

  def __init__(
      self,
      num_channels: int,
      epsilon: float = 1e-5,
      affine: bool = True,
      bias_init: Initializer = ZeroInit(),
      scale_init: Initializer = OneInit(),
      mode: Mode = training,
      name: str = None,
  ):
    super(InstanceNorm, self).__init__(num_channels=num_channels,
                                       num_groups=num_channels,
                                       epsilon=epsilon,
                                       affine=affine,
                                       bias_init=bias_init,
                                       scale_init=scale_init,
                                       mode=mode,
                                       name=name)
