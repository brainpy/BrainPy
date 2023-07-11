# -*- coding: utf-8 -*-

from typing import Union, Optional, Sequence, Callable

from jax import lax, numpy as jnp

from brainpy._src.context import share
from brainpy import math as bm, check
from brainpy.initialize import ZeroInit, OneInit, Initializer, parameter
from brainpy.types import ArrayType
from brainpy._src.dnn.base import Layer

__all__ = [
  'BatchNorm1d',
  'BatchNorm2d',
  'BatchNorm3d',
  'BatchNorm1D',
  'BatchNorm2D',
  'BatchNorm3D',

  'LayerNorm',
  'GroupNorm',
  'InstanceNorm',
]


def _square(x):
  """Computes the elementwise square of the absolute value |x|^2."""
  if jnp.iscomplexobj(x):
    return lax.square(lax.real(x)) + lax.square(lax.imag(x))
  else:
    return lax.square(x)


class BatchNorm(Layer):
  r"""Batch Normalization layer [1]_.

  This layer aims to reduce the internal covariant shift of data. It
  normalizes a batch of data by fixing the mean and variance of inputs
  on each feature (channel). Most commonly, the first axis of the data
  is the batch, and the last is the channel. However, users can specify
  the axes to be normalized.

  .. math::
     y=\frac{x-\mathrm{E}[x]}{\sqrt{\operatorname{Var}[x]+\epsilon}} * \gamma+\beta

  .. note::
      This :attr:`momentum` argument is different from one used in optimizer
      classes and the conventional notion of momentum. Mathematically, the
      update rule for running statistics here is
      :math:`\hat{x}_\text{new} = \text{momentum} \times \hat{x} + (1-\text{momentum}) \times x_t`,
      where :math:`\hat{x}` is the estimated statistic and :math:`x_t` is the
      new observed value.

  Parameters
  ----------
  num_features: int
    ``C`` from an expected input of size ``(..., C)``.
  axis: int, tuple, list
    Axes where the data will be normalized. The feature (channel) axis should be excluded.
  momentum: float
    The value used for the ``running_mean`` and ``running_var`` computation. Default: 0.99
  epsilon: float
    A value added to the denominator for numerical stability. Default: 1e-5
  affine: bool
    A boolean value that when set to ``True``, this module has
    learnable affine parameters. Default: ``True``
  bias_initializer: Initializer, ArrayType, Callable
    An initializer generating the original translation matrix
  scale_initializer: Initializer, ArrayType, Callable
    An initializer generating the original scaling matrix
  axis_name: optional, str, sequence of str
    If not ``None``, it should be a string (or sequence of
    strings) representing the axis name(s) over which this module is being
    run within a jax map (e.g. ``jax.pmap`` or ``jax.vmap``). Supplying this
    argument means that batch statistics are calculated across all replicas
    on the named axes.
  axis_index_groups: optional, sequence
    Specifies how devices are grouped. Valid
    only within ``jax.pmap`` collectives.

  References
  ----------
  .. [1] Ioffe, Sergey and Christian Szegedy. “Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift.” ArXiv abs/1502.03167 (2015): n. pag.

  """

  def __init__(
      self,
      num_features: int,
      axis: Union[int, Sequence[int]],
      epsilon: float = 1e-5,
      momentum: float = 0.99,
      affine: bool = True,
      bias_initializer: Union[Initializer, ArrayType, Callable] = ZeroInit(),
      scale_initializer: Union[Initializer, ArrayType, Callable] = OneInit(),
      axis_name: Optional[Union[str, Sequence[str]]] = None,
      axis_index_groups: Optional[Sequence[Sequence[int]]] = None,
      mode: Optional[bm.Mode] = None,
      name: Optional[str] = None,
  ):
    super(BatchNorm, self).__init__(name=name, mode=mode)
    check.is_subclass(self.mode, (bm.BatchingMode, bm.TrainingMode), self.name)

    # parameters
    self.num_features = num_features
    self.epsilon = epsilon
    self.momentum = momentum
    self.affine = affine
    self.bias_initializer = bias_initializer
    self.scale_initializer = scale_initializer
    self.axis = (axis,) if jnp.isscalar(axis) else axis
    self.axis_name = axis_name
    self.axis_index_groups = axis_index_groups

    # variables
    self.running_mean = bm.Variable(jnp.zeros(self.num_features))
    self.running_var = bm.Variable(jnp.ones(self.num_features))
    if self.affine:
      assert isinstance(self.mode, bm.TrainingMode)
      self.bias = bm.TrainVar(parameter(self.bias_initializer, self.num_features))
      self.scale = bm.TrainVar(parameter(self.scale_initializer, self.num_features))

  def _check_input_dim(self, x):
    raise NotImplementedError

  def update(self, x):
    self._check_input_dim(x)

    x = bm.as_jax(x)

    if share.load('fit'):
      mean = jnp.mean(x, self.axis)
      mean_of_square = jnp.mean(_square(x), self.axis)
      if self.axis_name is not None:
        mean, mean_of_square = jnp.split(
          lax.pmean(jnp.concatenate([mean, mean_of_square]),
                    axis_name=self.axis_name,
                    axis_index_groups=self.axis_index_groups),
          2
        )
      var = jnp.maximum(0., mean_of_square - _square(mean))
      self.running_mean.value = (self.momentum * self.running_mean + (1 - self.momentum) * mean)
      self.running_var.value = (self.momentum * self.running_var + (1 - self.momentum) * var)
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


class BatchNorm1d(BatchNorm):
  r"""1-D batch normalization [1]_.

  The data should be of `(b, l, c)`, where `b` is the batch dimension,
  `l` is the layer dimension, and `c` is the channel dimension.

  .. math::
     y=\frac{x-\mathrm{E}[x]}{\sqrt{\operatorname{Var}[x]+\epsilon}} * \gamma+\beta

  .. note::
      This :attr:`momentum` argument is different from one used in optimizer
      classes and the conventional notion of momentum. Mathematically, the
      update rule for running statistics here is
      :math:`\hat{x}_\text{new} = \text{momentum} \times \hat{x} + (1-\text{momentum}) \times x_t`,
      where :math:`\hat{x}` is the estimated statistic and :math:`x_t` is the
      new observed value.

  Parameters
  ----------
  num_features: int
    ``C`` from an expected input of size ``(B, L, C)``.
  axis: int, tuple, list
    axes where the data will be normalized. The feature (channel) axis should be excluded.
  epsilon: float
    A value added to the denominator for numerical stability. Default: 1e-5
  momentum: float
    The value used for the ``running_mean`` and ``running_var`` computation. Default: 0.99
  affine: bool
    A boolean value that when set to ``True``, this module has
    learnable affine parameters. Default: ``True``
  bias_initializer: Initializer, ArrayType, Callable
    an initializer generating the original translation matrix
  scale_initializer: Initializer, ArrayType, Callable
    an initializer generating the original scaling matrix
  axis_name: optional, str, sequence of str
    If not ``None``, it should be a string (or sequence of
    strings) representing the axis name(s) over which this module is being
    run within a jax map (e.g. ``jax.pmap`` or ``jax.vmap``). Supplying this
    argument means that batch statistics are calculated across all replicas
    on the named axes.
  axis_index_groups: optional, sequence
    Specifies how devices are grouped. Valid
    only within ``jax.pmap`` collectives.

  References
  ----------
  .. [1] Ioffe, Sergey and Christian Szegedy. “Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift.” ArXiv abs/1502.03167 (2015): n. pag.

  """

  def __init__(
      self,
      num_features: int,
      axis: Union[int, Sequence[int]] = (0, 1),
      epsilon: float = 1e-5,
      momentum: float = 0.99,
      affine: bool = True,
      bias_initializer: Union[Initializer, ArrayType, Callable] = ZeroInit(),
      scale_initializer: Union[Initializer, ArrayType, Callable] = OneInit(),
      axis_name: Optional[Union[str, Sequence[str]]] = None,
      axis_index_groups: Optional[Sequence[Sequence[int]]] = None,
      mode: Optional[bm.Mode] = None,
      name: Optional[str] = None,
  ):
    super(BatchNorm1d, self).__init__(num_features=num_features,
                                      axis=axis,
                                      epsilon=epsilon,
                                      momentum=momentum,
                                      affine=affine,
                                      bias_initializer=bias_initializer,
                                      scale_initializer=scale_initializer,
                                      axis_name=axis_name,
                                      axis_index_groups=axis_index_groups,
                                      mode=mode,
                                      name=name)

  def _check_input_dim(self, x):
    if x.ndim != 3:
      raise ValueError(f"expected 3D input (got {x.ndim}D input)")
    assert x.shape[-1] == self.num_features


class BatchNorm2d(BatchNorm):
  r"""2-D batch normalization [1]_.

  The data should be of `(b, h, w, c)`, where `b` is the batch dimension,
  `h` is the height dimension, `w` is the width dimension, and `c` is the
  channel dimension.

  .. math::
     y=\frac{x-\mathrm{E}[x]}{\sqrt{\operatorname{Var}[x]+\epsilon}} * \gamma+\beta

  .. note::
      This :attr:`momentum` argument is different from one used in optimizer
      classes and the conventional notion of momentum. Mathematically, the
      update rule for running statistics here is
      :math:`\hat{x}_\text{new} = \text{momentum} \times \hat{x} + (1-\text{momentum}) \times x_t`,
      where :math:`\hat{x}` is the estimated statistic and :math:`x_t` is the
      new observed value.

  Parameters
  ----------
  num_features: int
    ``C`` from an expected input of size ``(B, H, W, C)``.
  axis: int, tuple, list
    axes where the data will be normalized. The feature (channel) axis should be excluded.
  epsilon: float
    a value added to the denominator for numerical stability. Default: 1e-5
  momentum: float
    The value used for the ``running_mean`` and ``running_var`` computation. Default: 0.99
  affine: bool
    A boolean value that when set to ``True``, this module has
    learnable affine parameters. Default: ``True``
  bias_initializer: Initializer, ArrayType, Callable
    an initializer generating the original translation matrix
  scale_initializer: Initializer, ArrayType, Callable
    an initializer generating the original scaling matrix
  axis_name: optional, str, sequence of str
    If not ``None``, it should be a string (or sequence of
    strings) representing the axis name(s) over which this module is being
    run within a jax map (e.g. ``jax.pmap`` or ``jax.vmap``). Supplying this
    argument means that batch statistics are calculated across all replicas
    on the named axes.
  axis_index_groups: optional, sequence
    Specifies how devices are grouped. Valid
    only within ``jax.pmap`` collectives.

  References
  ----------
  .. [1] Ioffe, Sergey and Christian Szegedy. “Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift.” ArXiv abs/1502.03167 (2015): n. pag.

  """

  def __init__(
      self,
      num_features: int,
      axis: Union[int, Sequence[int]] = (0, 1, 2),
      epsilon: float = 1e-5,
      momentum: float = 0.99,
      affine: bool = True,
      bias_initializer: Union[Initializer, ArrayType, Callable] = ZeroInit(),
      scale_initializer: Union[Initializer, ArrayType, Callable] = OneInit(),
      axis_name: Optional[Union[str, Sequence[str]]] = None,
      axis_index_groups: Optional[Sequence[Sequence[int]]] = None,
      mode: Optional[bm.Mode] = None,
      name: Optional[str] = None,
  ):
    super(BatchNorm2d, self).__init__(num_features=num_features,
                                      axis=axis,
                                      epsilon=epsilon,
                                      momentum=momentum,
                                      affine=affine,
                                      bias_initializer=bias_initializer,
                                      scale_initializer=scale_initializer,
                                      axis_name=axis_name,
                                      axis_index_groups=axis_index_groups,
                                      mode=mode,
                                      name=name)

  def _check_input_dim(self, x):
    if x.ndim != 4:
      raise ValueError(f"expected 4D input (got {x.ndim}D input)")
    assert x.shape[-1] == self.num_features


class BatchNorm3d(BatchNorm):
  r"""3-D batch normalization [1]_.

  The data should be of `(b, h, w, d, c)`, where `b` is the batch dimension,
  `h` is the height dimension, `w` is the width dimension, `d` is the depth
  dimension, and `c` is the channel dimension.

  .. math::
     y=\frac{x-\mathrm{E}[x]}{\sqrt{\operatorname{Var}[x]+\epsilon}} * \gamma+\beta

  .. note::
      This :attr:`momentum` argument is different from one used in optimizer
      classes and the conventional notion of momentum. Mathematically, the
      update rule for running statistics here is
      :math:`\hat{x}_\text{new} = \text{momentum} \times \hat{x} + (1-\text{momentum}) \times x_t`,
      where :math:`\hat{x}` is the estimated statistic and :math:`x_t` is the
      new observed value.

  Parameters
  ----------
  num_features: int
    ``C`` from an expected input of size ``(B, H, W, D, C)``.
  axis: int, tuple, list
    axes where the data will be normalized. The feature (channel) axis should be excluded.
  epsilon: float
    a value added to the denominator for numerical stability. Default: 1e-5
  momentum: float
    The value used for the ``running_mean`` and ``running_var`` computation. Default: 0.99
  affine: bool
    A boolean value that when set to ``True``, this module has
    learnable affine parameters. Default: ``True``
  bias_initializer: Initializer, ArrayType, Callable
    an initializer generating the original translation matrix
  scale_initializer: Initializer, ArrayType, Callable
    an initializer generating the original scaling matrix
  axis_name: optional, str, sequence of str
    If not ``None``, it should be a string (or sequence of
    strings) representing the axis name(s) over which this module is being
    run within a jax map (e.g. ``jax.pmap`` or ``jax.vmap``). Supplying this
    argument means that batch statistics are calculated across all replicas
    on the named axes.
  axis_index_groups: optional, sequence
    Specifies how devices are grouped. Valid
    only within ``jax.pmap`` collectives.

  References
  ----------
  .. [1] Ioffe, Sergey and Christian Szegedy. “Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift.” ArXiv abs/1502.03167 (2015): n. pag.

  """

  def __init__(
      self,
      num_features: int,
      axis: Union[int, Sequence[int]] = (0, 1, 2, 3),
      epsilon: float = 1e-5,
      momentum: float = 0.99,
      affine: bool = True,
      bias_initializer: Union[Initializer, ArrayType, Callable] = ZeroInit(),
      scale_initializer: Union[Initializer, ArrayType, Callable] = OneInit(),
      axis_name: Optional[Union[str, Sequence[str]]] = None,
      axis_index_groups: Optional[Sequence[Sequence[int]]] = None,
      mode: Optional[bm.Mode] = None,
      name: Optional[str] = None,
  ):
    super(BatchNorm3d, self).__init__(num_features=num_features,
                                      axis=axis,
                                      epsilon=epsilon,
                                      momentum=momentum,
                                      affine=affine,
                                      bias_initializer=bias_initializer,
                                      scale_initializer=scale_initializer,
                                      axis_name=axis_name,
                                      axis_index_groups=axis_index_groups,
                                      mode=mode,
                                      name=name)

  def _check_input_dim(self, x):
    if x.ndim != 5:
      raise ValueError(f"expected 5D input (got {x.ndim}D input)")
    assert x.shape[-1] == self.num_features


class LayerNorm(Layer):
  r"""Layer normalization (https://arxiv.org/abs/1607.06450).

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
  bias_initializer: Initializer, ArrayType, Callable
    an initializer generating the original translation matrix
  scale_initializer: Initializer, ArrayType, Callable
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
      bias_initializer: Union[Initializer, ArrayType, Callable] = ZeroInit(),
      scale_initializer: Union[Initializer, ArrayType, Callable] = OneInit(),
      elementwise_affine: bool = True,
      mode: Optional[bm.Mode] = None,
      name: Optional[str] = None
  ):
    super(LayerNorm, self).__init__(name=name, mode=mode)

    self.epsilon = epsilon
    self.bias_initializer = bias_initializer
    self.scale_initializer = scale_initializer
    if isinstance(normalized_shape, int):
      normalized_shape = (normalized_shape,)
    self.normalized_shape = tuple(normalized_shape)
    assert all([isinstance(s, int) for s in normalized_shape]), 'Must be a sequence of integer.'
    self.elementwise_affine = elementwise_affine
    if self.elementwise_affine:
      assert isinstance(self.mode, bm.TrainingMode)
      self.bias = bm.TrainVar(parameter(self.bias_initializer, self.normalized_shape))
      self.scale = bm.TrainVar(parameter(self.scale_initializer, self.normalized_shape))

  def update(self, x):
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


class GroupNorm(Layer):
  r"""Group normalization layer.

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
  bias_initializer: Initializer, ArrayType, Callable
    An initializer generating the original translation matrix
  scale_initializer: Initializer, ArrayType, Callable
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
      bias_initializer: Union[Initializer, ArrayType, Callable] = ZeroInit(),
      scale_initializer: Union[Initializer, ArrayType, Callable] = OneInit(),
      mode: Optional[bm.Mode] = None,
      name: Optional[str] = None,
  ):
    super(GroupNorm, self).__init__(name=name, mode=mode)
    if num_channels % num_groups != 0:
      raise ValueError('num_channels must be divisible by num_groups')
    self.num_groups = num_groups
    self.num_channels = num_channels
    self.epsilon = epsilon
    self.affine = affine
    self.bias_initializer = bias_initializer
    self.scale_initializer = scale_initializer
    if self.affine:
      assert isinstance(self.mode, bm.TrainingMode)
      self.bias = bm.TrainVar(parameter(self.bias_initializer, self.num_channels))
      self.scale = bm.TrainVar(parameter(self.scale_initializer, self.num_channels))

  def update(self, x):
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
  r"""Instance normalization layer.

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
  bias_initializer: Initializer, ArrayType, Callable
    an initializer generating the original translation matrix
  scale_initializer: Initializer, ArrayType, Callable
    an initializer generating the original scaling matrix
  """

  def __init__(
      self,
      num_channels: int,
      epsilon: float = 1e-5,
      affine: bool = True,
      bias_initializer: Union[Initializer, ArrayType, Callable] = ZeroInit(),
      scale_initializer: Union[Initializer, ArrayType, Callable] = OneInit(),
      mode: Optional[bm.Mode] = None,
      name: Optional[str] = None,
  ):
    super(InstanceNorm, self).__init__(num_channels=num_channels,
                                       num_groups=num_channels,
                                       epsilon=epsilon,
                                       affine=affine,
                                       bias_initializer=bias_initializer,
                                       scale_initializer=scale_initializer,
                                       mode=mode,
                                       name=name)


BatchNorm1D = BatchNorm1d
BatchNorm2D = BatchNorm2d
BatchNorm3D = BatchNorm3d
