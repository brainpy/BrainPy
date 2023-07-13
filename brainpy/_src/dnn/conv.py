# -*- coding: utf-8 -*-

from typing import Union, Tuple, Optional, Sequence, Callable

from jax import lax

from brainpy import math as bm, tools, check
from brainpy._src.initialize import Initializer, XavierNormal, ZeroInit, parameter
from brainpy.types import ArrayType
from brainpy._src.dnn.base import Layer

__all__ = [
  'Conv1d', 'Conv2d', 'Conv3d',
  'Conv1D', 'Conv2D', 'Conv3D',
  'ConvTranspose1d', 'ConvTranspose2d', 'ConvTranspose3d',
]


def to_dimension_numbers(num_spatial_dims: int,
                         channels_last: bool,
                         transpose: bool) -> lax.ConvDimensionNumbers:
  """Create a `lax.ConvDimensionNumbers` for the given inputs."""
  num_dims = num_spatial_dims + 2
  if channels_last:
    spatial_dims = tuple(range(1, num_dims - 1))
    image_dn = (0, num_dims - 1) + spatial_dims
  else:
    spatial_dims = tuple(range(2, num_dims))
    image_dn = (0, 1) + spatial_dims
  if transpose:
    kernel_dn = (num_dims - 2, num_dims - 1) + tuple(range(num_dims - 2))
  else:
    kernel_dn = (num_dims - 1, num_dims - 2) + tuple(range(num_dims - 2))
  return lax.ConvDimensionNumbers(lhs_spec=image_dn,
                                  rhs_spec=kernel_dn,
                                  out_spec=image_dn)


class _GeneralConv(Layer):
  """Apply a convolution to the inputs.

  Parameters
  ----------
  num_spatial_dims: int
    The number of spatial dimensions of the input.
  in_channels: int
    The number of input channels.
  out_channels: int
    The number of output channels.
  kernel_size: int, sequence of int
    The shape of the convolutional kernel.
    For 1D convolution, the kernel size can be passed as an integer.
    For all other cases, it must be a sequence of integers.
  stride: int, sequence of int
    An integer or a sequence of `n` integers, representing the inter-window strides (default: 1).
  padding: str, int, sequence of int, sequence of tuple
    Either the string `'SAME'`, the string `'VALID'`, or a sequence of n `(low,
    high)` integer pairs that give the padding to apply before and after each
    spatial dimension.
  lhs_dilation: int, sequence of int
    An integer or a sequence of `n` integers, giving the
    dilation factor to apply in each spatial dimension of `inputs`
    (default: 1). Convolution with input dilation `d` is equivalent to
    transposed convolution with stride `d`.
  rhs_dilation: int, sequence of int
    An integer or a sequence of `n` integers, giving the
    dilation factor to apply in each spatial dimension of the convolution
    kernel (default: 1). Convolution with kernel dilation
    is also known as 'atrous convolution'.
  groups: int
    If specified, divides the input features into groups. default 1.
  w_initializer: Callable, ArrayType, Initializer
    The initializer for the convolutional kernel.
  b_initializer: Optional, Callable, ArrayType, Initializer
    The initializer for the bias.
  mask: ArrayType, Optional
    The optional mask of the weights.
  mode: Mode
    The computation mode of the current object. Default it is `training`.
  name: str, Optional
    The name of the object.
  """

  supported_modes = (bm.TrainingMode, bm.BatchingMode)

  def __init__(
      self,
      num_spatial_dims: int,
      in_channels: int,
      out_channels: int,
      kernel_size: Union[int, Tuple[int, ...]],
      stride: Union[int, Tuple[int, ...]] = 1,
      padding: Union[str, int, Tuple[int, int], Sequence[Tuple[int, int]]] = 'SAME',
      lhs_dilation: Union[int, Tuple[int, ...]] = 1,
      rhs_dilation: Union[int, Tuple[int, ...]] = 1,
      groups: int = 1,
      w_initializer: Union[Callable, ArrayType, Initializer] = XavierNormal(),
      b_initializer: Optional[Union[Callable, ArrayType, Initializer]] = ZeroInit(),
      mask: Optional[ArrayType] = None,
      mode: bm.Mode = None,
      name: str = None,
  ):
    super(_GeneralConv, self).__init__(name=name, mode=mode)
    check.is_subclass(self.mode, (bm.TrainingMode, bm.BatchingMode), self.name)

    self.num_spatial_dims = num_spatial_dims
    self.in_channels = in_channels
    self.out_channels = out_channels
    self.stride = tools.replicate(stride, num_spatial_dims, 'stride')
    self.kernel_size = tools.replicate(kernel_size, num_spatial_dims, 'kernel_size')
    self.lhs_dilation = tools.replicate(lhs_dilation, num_spatial_dims, 'lhs_dilation')
    self.rhs_dilation = tools.replicate(rhs_dilation, num_spatial_dims, 'rhs_dilation')
    self.groups = groups
    self.w_initializer = w_initializer
    self.b_initializer = b_initializer
    self.mask = mask
    self.dimension_numbers = to_dimension_numbers(num_spatial_dims, channels_last=True, transpose=False)

    if isinstance(padding, str):
      assert padding in ['SAME', 'VALID']
    elif isinstance(padding, int):
      padding = tuple((padding, padding) for _ in range(num_spatial_dims))
    elif isinstance(padding, (tuple, list)):
      if isinstance(padding[0], int):
        padding = (padding,) * num_spatial_dims
      elif isinstance(padding[0], (tuple, list)):
        if len(padding) == 1:
          padding = tuple(padding) * num_spatial_dims
        else:
          if len(padding) != num_spatial_dims:
            raise ValueError(f"Padding {padding} must be a Tuple[int, int], "
                             f"or sequence of Tuple[int, int] with length 1, "
                             f"or sequence of Tuple[int, int] with length {num_spatial_dims}.")
          padding = tuple(padding)
    else:
      raise ValueError
    self.padding = padding

    assert self.out_channels % self.groups == 0, '"out_channels" should be divisible by groups'
    assert self.in_channels % self.groups == 0, '"in_channels" should be divisible by groups'

    kernel_shape = tuple(self.kernel_size) + (self.in_channels // self.groups, self.out_channels)
    bias_shape = (1,) * len(self.kernel_size) + (self.out_channels,)
    self.w = parameter(self.w_initializer, kernel_shape, allow_none=False)
    self.b = parameter(self.b_initializer, bias_shape, allow_none=True)
    if isinstance(self.mode, bm.TrainingMode):
      self.w = bm.TrainVar(self.w)
      if self.b is not None:
        self.b = bm.TrainVar(self.b)

  def _check_input_dim(self, x):
    if x.ndim != self.num_spatial_dims + 2:
      raise ValueError(f"expected {self.num_spatial_dims + 2}D input (got {x.ndim}D input)")
    if self.in_channels != x.shape[-1]:
      raise ValueError(f"input channels={x.shape[-1]} needs to have "
                       f"the same size as in_channels={self.in_channels}.")

  def update(self, x):
    self._check_input_dim(x)
    w = self.w.value
    if self.mask is not None:
      try:
        lax.broadcast_shapes(self.w.shape, self.mask.shape)
      except:
        raise ValueError(f"Mask needs to have the same shape as weights. {self.mask.shape} != {self.w.shape}")
      w = w * self.mask
    y = lax.conv_general_dilated(lhs=bm.as_jax(x),
                                 rhs=bm.as_jax(w),
                                 window_strides=self.stride,
                                 padding=self.padding,
                                 lhs_dilation=self.lhs_dilation,
                                 rhs_dilation=self.rhs_dilation,
                                 feature_group_count=self.groups,
                                 dimension_numbers=self.dimension_numbers)
    return y if self.b is None else (y + self.b.value)

  def __repr__(self):
    return (f'{self.__class__.__name__}(in_channels={self.in_channels}, '
            f'out_channels={self.out_channels}, kernel_size={self.kernel_size}, '
            f'stride={self.stride}, padding={self.padding}, groups={self.groups})')


class Conv1d(_GeneralConv):
  """One-dimensional convolution.

  Parameters
  ----------
  in_channels: int
    The number of input channels.
  out_channels: int
    The number of output channels.
  kernel_size: int, sequence of int
    The shape of the convolutional kernel.
    For 1D convolution, the kernel size can be passed as an integer.
    For all other cases, it must be a sequence of integers.
  strides: int, sequence of int
    An integer or a sequence of `n` integers, representing the inter-window strides (default: 1).
  padding: str, int, sequence of int, sequence of tuple
    Either the string `'SAME'`, the string `'VALID'`, or a sequence of n `(low,
    high)` integer pairs that give the padding to apply before and after each
    spatial dimension.
  lhs_dilation: int, sequence of int
    An integer or a sequence of `n` integers, giving the
    dilation factor to apply in each spatial dimension of `inputs`
    (default: 1). Convolution with input dilation `d` is equivalent to
    transposed convolution with stride `d`.
  rhs_dilation: int, sequence of int
    An integer or a sequence of `n` integers, giving the
    dilation factor to apply in each spatial dimension of the convolution
    kernel (default: 1). Convolution with kernel dilation
    is also known as 'atrous convolution'.
  groups: int
    If specified, divides the input features into groups. default 1.
  w_initializer: Callable, ArrayType, Initializer
    The initializer for the convolutional kernel.
  b_initializer: Callable, ArrayType, Initializer
    The initializer for the bias.
  mask: ArrayType, Optional
    The optional mask of the weights.
  mode: Mode
    The computation mode of the current object. Default it is `training`.
  name: str, Optional
    The name of the object.
  """

  def __init__(
      self,
      in_channels: int,
      out_channels: int,
      kernel_size: Union[int, Tuple[int, ...]],
      stride: Union[int, Tuple[int, ...]] = None,
      strides: Union[int, Tuple[int, ...]] = None,  # deprecated
      padding: Union[str, int, Tuple[int, int], Sequence[Tuple[int, int]]] = 'SAME',
      lhs_dilation: Union[int, Tuple[int, ...]] = 1,
      rhs_dilation: Union[int, Tuple[int, ...]] = 1,
      groups: int = 1,
      w_initializer: Union[Callable, ArrayType, Initializer] = XavierNormal(),
      b_initializer: Optional[Union[Callable, ArrayType, Initializer]] = ZeroInit(),
      mask: Optional[ArrayType] = None,
      mode: Optional[bm.Mode] = None,
      name: Optional[str] = None,
  ):
    if stride is None:
      if strides is None:
        stride = 1
      else:
        stride = strides
    else:
      if strides is not None:
        raise ValueError('Cannot provide "stride" and "strides" both.')

    super(Conv1d, self).__init__(num_spatial_dims=1,
                                 in_channels=in_channels,
                                 out_channels=out_channels,
                                 kernel_size=kernel_size,
                                 stride=stride,
                                 padding=padding,
                                 lhs_dilation=lhs_dilation,
                                 rhs_dilation=rhs_dilation,
                                 groups=groups,
                                 w_initializer=w_initializer,
                                 b_initializer=b_initializer,
                                 mask=mask,
                                 mode=mode,
                                 name=name)

  def _check_input_dim(self, x):
    if x.ndim != 3:
      raise ValueError(f"expected 3D input (got {x.ndim}D input)")
    if self.in_channels != x.shape[-1]:
      raise ValueError(f"input channels={x.shape[-1]} needs to have "
                       f"the same size as in_channels={self.in_channels}.")


class Conv2d(_GeneralConv):
  """Two-dimensional convolution.

  Parameters
  ----------
  in_channels: int
    The number of input channels.
  out_channels: int
    The number of output channels.
  kernel_size: int, sequence of int
    The shape of the convolutional kernel.
    For 1D convolution, the kernel size can be passed as an integer.
    For all other cases, it must be a sequence of integers.
  stride: int, sequence of int
    An integer or a sequence of `n` integers, representing the inter-window strides (default: 1).
  padding: str, int, sequence of int, sequence of tuple
    Either the string `'SAME'`, the string `'VALID'`, or a sequence of n `(low,
    high)` integer pairs that give the padding to apply before and after each
    spatial dimension.
  lhs_dilation: int, sequence of int
    An integer or a sequence of `n` integers, giving the
    dilation factor to apply in each spatial dimension of `inputs`
    (default: 1). Convolution with input dilation `d` is equivalent to
    transposed convolution with stride `d`.
  rhs_dilation: int, sequence of int
    An integer or a sequence of `n` integers, giving the
    dilation factor to apply in each spatial dimension of the convolution
    kernel (default: 1). Convolution with kernel dilation
    is also known as 'atrous convolution'.
  groups: int
    If specified, divides the input features into groups. default 1.
  w_initializer: Callable, ArrayType, Initializer
    The initializer for the convolutional kernel.
  b_initializer: Callable, ArrayType, Initializer
    The initializer for the bias.
  mask: ArrayType, Optional
    The optional mask of the weights.
  mode: Mode
    The computation mode of the current object. Default it is `training`.
  name: str, Optional
    The name of the object.

  """

  def __init__(
      self,
      in_channels: int,
      out_channels: int,
      kernel_size: Union[int, Tuple[int, ...]],
      stride: Union[int, Tuple[int, ...]] = None,
      strides: Union[int, Tuple[int, ...]] = None,  # deprecated
      padding: Union[str, int, Tuple[int, int], Sequence[Tuple[int, int]]] = 'SAME',
      lhs_dilation: Union[int, Tuple[int, ...]] = 1,
      rhs_dilation: Union[int, Tuple[int, ...]] = 1,
      groups: int = 1,
      w_initializer: Union[Callable, ArrayType, Initializer] = XavierNormal(),
      b_initializer: Optional[Union[Callable, ArrayType, Initializer]] = ZeroInit(),
      mask: Optional[ArrayType] = None,
      mode: Optional[bm.Mode] = None,
      name: Optional[str] = None,
  ):
    if stride is None:
      if strides is None:
        stride = 1
      else:
        stride = strides
    else:
      if strides is not None:
        raise ValueError('Cannot provide "stride" and "strides" both.')

    super(Conv2d, self).__init__(num_spatial_dims=2,
                                 in_channels=in_channels,
                                 out_channels=out_channels,
                                 kernel_size=kernel_size,
                                 stride=stride,
                                 padding=padding,
                                 lhs_dilation=lhs_dilation,
                                 rhs_dilation=rhs_dilation,
                                 groups=groups,
                                 w_initializer=w_initializer,
                                 b_initializer=b_initializer,
                                 mask=mask,
                                 mode=mode,
                                 name=name)

  def _check_input_dim(self, x):
    if x.ndim != 4:
      raise ValueError(f"expected 4D input (got {x.ndim}D input)")
    if self.in_channels != x.shape[-1]:
      raise ValueError(f"input channels={x.shape[-1]} needs to have "
                       f"the same size as in_channels={self.in_channels}.")


class Conv3d(_GeneralConv):
  """Three-dimensional convolution.

  Parameters
  ----------
  in_channels: int
    The number of input channels.
  out_channels: int
    The number of output channels.
  kernel_size: int, sequence of int
    The shape of the convolutional kernel.
    For 1D convolution, the kernel size can be passed as an integer.
    For all other cases, it must be a sequence of integers.
  stride: int, sequence of int
    An integer or a sequence of `n` integers, representing the inter-window strides (default: 1).
  padding: str, int, sequence of int, sequence of tuple
    Either the string `'SAME'`, the string `'VALID'`, or a sequence of n `(low,
    high)` integer pairs that give the padding to apply before and after each
    spatial dimension.
  lhs_dilation: int, sequence of int
    An integer or a sequence of `n` integers, giving the
    dilation factor to apply in each spatial dimension of `inputs`
    (default: 1). Convolution with input dilation `d` is equivalent to
    transposed convolution with stride `d`.
  rhs_dilation: int, sequence of int
    An integer or a sequence of `n` integers, giving the
    dilation factor to apply in each spatial dimension of the convolution
    kernel (default: 1). Convolution with kernel dilation
    is also known as 'atrous convolution'.
  groups: int
    If specified, divides the input features into groups. default 1.
  w_initializer: Callable, ArrayType, Initializer
    The initializer for the convolutional kernel.
  b_initializer: Callable, ArrayType, Initializer
    The initializer for the bias.
  mask: ArrayType, Optional
    The optional mask of the weights.
  mode: Mode
    The computation mode of the current object. Default it is `training`.
  name: str, Optional
    The name of the object.

  """

  def __init__(
      self,
      in_channels: int,
      out_channels: int,
      kernel_size: Union[int, Tuple[int, ...]],
      stride: Union[int, Tuple[int, ...]] = None,
      strides: Union[int, Tuple[int, ...]] = None,  # deprecated
      padding: Union[str, int, Tuple[int, int], Sequence[Tuple[int, int]]] = 'SAME',
      lhs_dilation: Union[int, Tuple[int, ...]] = 1,
      rhs_dilation: Union[int, Tuple[int, ...]] = 1,
      groups: int = 1,
      w_initializer: Union[Callable, ArrayType, Initializer] = XavierNormal(),
      b_initializer: Optional[Union[Callable, ArrayType, Initializer]] = ZeroInit(),
      mask: Optional[ArrayType] = None,
      mode: Optional[bm.Mode] = None,
      name: Optional[str] = None,
  ):
    if stride is None:
      if strides is None:
        stride = 1
      else:
        stride = strides
    else:
      if strides is not None:
        raise ValueError('Cannot provide "stride" and "strides" both.')

    super(Conv3d, self).__init__(num_spatial_dims=3,
                                 in_channels=in_channels,
                                 out_channels=out_channels,
                                 kernel_size=kernel_size,
                                 stride=stride,
                                 padding=padding,
                                 lhs_dilation=lhs_dilation,
                                 rhs_dilation=rhs_dilation,
                                 groups=groups,
                                 w_initializer=w_initializer,
                                 b_initializer=b_initializer,
                                 mask=mask,
                                 mode=mode,
                                 name=name)

  def _check_input_dim(self, x):
    if x.ndim != 5:
      raise ValueError(f"expected 5D input (got {x.ndim}D input)")
    if self.in_channels != x.shape[-1]:
      raise ValueError(f"input channels={x.shape[-1]} needs to have "
                       f"the same size as in_channels={self.in_channels}.")


Conv1D = Conv1d
Conv2D = Conv2d
Conv3D = Conv3d


class _GeneralConvTranspose(Layer):
  supported_modes = (bm.TrainingMode, bm.BatchingMode)

  def __init__(
      self,
      num_spatial_dims: int,
      in_channels: int,
      out_channels: int,
      kernel_size: Union[int, Tuple[int, ...]],
      stride: Union[int, Tuple[int, ...]] = 1,
      padding: Union[str, int, Tuple[int, int], Sequence[Tuple[int, int]]] = 'SAME',
      w_initializer: Union[Callable, ArrayType, Initializer] = XavierNormal(in_axis=-1, out_axis=-2),
      b_initializer: Optional[Union[Callable, ArrayType, Initializer]] = ZeroInit(),
      mask: Optional[ArrayType] = None,
      precision: Optional[lax.Precision] = None,
      mode: bm.Mode = None,
      name: str = None,
  ):
    super().__init__(name=name, mode=mode)

    assert self.mode.is_parent_of(bm.TrainingMode, bm.BatchingMode)

    self.num_spatial_dims = num_spatial_dims
    self.in_channels = in_channels
    self.out_channels = out_channels
    self.stride = tools.replicate(stride, num_spatial_dims, 'stride')
    self.kernel_size = tools.replicate(kernel_size, num_spatial_dims, 'kernel_size')
    self.w_initializer = w_initializer
    self.b_initializer = b_initializer
    self.precision = precision
    self.mask = mask
    self.dimension_numbers = to_dimension_numbers(num_spatial_dims, channels_last=True, transpose=False)

    if isinstance(padding, str):
      assert padding in ['SAME', 'VALID']
    elif isinstance(padding, int):
      padding = tuple((padding, padding) for _ in range(num_spatial_dims))
    elif isinstance(padding, (tuple, list)):
      if isinstance(padding[0], int):
        padding = (padding,) * num_spatial_dims
      elif isinstance(padding[0], (tuple, list)):
        if len(padding) == 1:
          padding = tuple(padding) * num_spatial_dims
        else:
          if len(padding) != num_spatial_dims:
            raise ValueError(f"Padding {padding} must be a Tuple[int, int], "
                             f"or sequence of Tuple[int, int] with length 1, "
                             f"or sequence of Tuple[int, int] with length {num_spatial_dims}.")
          padding = tuple(padding)
    else:
      raise ValueError
    self.padding = padding

    kernel_shape = tuple(self.kernel_size) + (self.in_channels, self.out_channels)
    bias_shape = (1,) * len(self.kernel_size) + (self.out_channels,)
    self.w = parameter(self.w_initializer, kernel_shape, allow_none=False)
    self.b = parameter(self.b_initializer, bias_shape, allow_none=True)
    if isinstance(self.mode, bm.TrainingMode):
      self.w = bm.TrainVar(self.w)
      if self.b is not None:
        self.b = bm.TrainVar(self.b)

  def _check_input_dim(self, x):
    raise NotImplementedError

  def update(self, x):
    self._check_input_dim(x)

    w = self.w.value
    if self.mask is not None:
      try:
        lax.broadcast_shapes(self.w.shape, self.mask.shape)
      except:
        raise ValueError(f"Mask needs to have the same shape as weights. {self.mask.shape} != {self.w.shape}")
      w = w * self.mask
    y = lax.conv_transpose(lhs=bm.as_jax(x),
                           rhs=bm.as_jax(w),
                           strides=self.stride,
                           padding=self.padding,
                           precision=self.precision,
                           rhs_dilation=None,
                           dimension_numbers=self.dimension_numbers)
    return y if self.b is None else (y + self.b.value)

  def __repr__(self):
    return (f'{self.__class__.__name__}(in_channels={self.in_channels}, '
            f'out_channels={self.out_channels}, kernel_size={self.kernel_size}, '
            f'stride={self.stride}, padding={self.padding})')


class ConvTranspose1d(_GeneralConvTranspose):
  """One dimensional transposed convolution (aka. deconvolution)."""

  def __init__(
      self,
      in_channels: int,
      out_channels: int,
      kernel_size: Union[int, Tuple[int, ...]],
      stride: Union[int, Tuple[int, ...]] = 1,
      padding: Union[str, int, Tuple[int, int], Sequence[Tuple[int, int]]] = 'SAME',
      w_initializer: Union[Callable, ArrayType, Initializer] = XavierNormal(in_axis=-1, out_axis=-2),
      b_initializer: Optional[Union[Callable, ArrayType, Initializer]] = ZeroInit(),
      mask: Optional[ArrayType] = None,
      precision: Optional[lax.Precision] = None,
      mode: bm.Mode = None,
      name: str = None,
  ):
    """Initializes the module.

    Args:
      output_channels: Number of output channels.
      kernel_shape: The shape of the kernel. Either an integer or a sequence of
        length 1.
      stride: Optional stride for the kernel. Either an integer or a sequence of
        length 1. Defaults to 1.
      output_shape: Output shape of the spatial dimensions of a transpose
        convolution. Can be either an integer or an iterable of integers. If a
        `None` value is given, a default shape is automatically calculated.
      padding: Optional padding algorithm. Either ``VALID`` or ``SAME``.
        Defaults to ``SAME``. See:
        https://www.tensorflow.org/xla/operation_semantics#conv_convolution.
      with_bias: Whether to add a bias. By default, true.
      w_init: Optional weight initialization. By default, truncated normal.
      b_init: Optional bias initialization. By default, zeros.
      data_format: The data format of the input. Either ``NWC`` or ``NCW``. By
        default, ``NWC``.
      mask: Optional mask of the weights.
      name: The name of the module.
    """
    super().__init__(
      num_spatial_dims=1,
      in_channels=in_channels,
      out_channels=out_channels,
      kernel_size=kernel_size,
      stride=stride,
      padding=padding,
      w_initializer=w_initializer,
      b_initializer=b_initializer,
      precision=precision,
      mode=mode,
      mask=mask,
      name=name
    )

  def _check_input_dim(self, x):
    if x.ndim != 3:
      raise ValueError(f"expected 3D input (got {x.ndim}D input)")
    if self.in_channels != x.shape[-1]:
      raise ValueError(f"input channels={x.shape[-1]} needs to have "
                       f"the same size as in_channels={self.in_channels}.")


class ConvTranspose2d(_GeneralConvTranspose):
  """Two dimensional transposed convolution (aka. deconvolution)."""

  def __init__(
      self,
      in_channels: int,
      out_channels: int,
      kernel_size: Union[int, Tuple[int, ...]],
      stride: Union[int, Tuple[int, ...]] = 1,
      padding: Union[str, int, Tuple[int, int], Sequence[Tuple[int, int]]] = 'SAME',
      w_initializer: Union[Callable, ArrayType, Initializer] = XavierNormal(in_axis=-1, out_axis=-2),
      b_initializer: Optional[Union[Callable, ArrayType, Initializer]] = ZeroInit(),
      mask: Optional[ArrayType] = None,
      precision: Optional[lax.Precision] = None,
      mode: bm.Mode = None,
      name: str = None,
  ):
    """Initializes the module.

    Args:
      out_channels: Number of output channels.
      kernel_size: The shape of the kernel. Either an integer or a sequence of
        length 2.
      stride: Optional stride for the kernel. Either an integer or a sequence of
        length 2. Defaults to 1.
      padding: Optional padding algorithm. Either ``VALID`` or ``SAME``.
        Defaults to ``SAME``. See:
        https://www.tensorflow.org/xla/operation_semantics#conv_convolution.
      w_initializer: Optional weight initialization. By default, truncated normal.
      b_initializer: Optional bias initialization. By default, zeros.
      mask: Optional mask of the weights.
      name: The name of the module.
    """
    super().__init__(
      num_spatial_dims=2,
      in_channels=in_channels,
      out_channels=out_channels,
      kernel_size=kernel_size,
      stride=stride,
      padding=padding,
      w_initializer=w_initializer,
      b_initializer=b_initializer,
      precision=precision,
      mode=mode,
      mask=mask,
      name=name
    )

  def _check_input_dim(self, x):
    if x.ndim != 4:
      raise ValueError(f"expected 4D input (got {x.ndim}D input)")
    if self.in_channels != x.shape[-1]:
      raise ValueError(f"input channels={x.shape[-1]} needs to have "
                       f"the same size as in_channels={self.in_channels}.")


class ConvTranspose3d(_GeneralConvTranspose):
  """Three dimensional transposed convolution (aka. deconvolution)."""

  def __init__(
      self,
      in_channels: int,
      out_channels: int,
      kernel_size: Union[int, Tuple[int, ...]],
      stride: Union[int, Tuple[int, ...]] = 1,
      padding: Union[str, int, Tuple[int, int], Sequence[Tuple[int, int]]] = 'SAME',
      w_initializer: Union[Callable, ArrayType, Initializer] = XavierNormal(in_axis=-1, out_axis=-2),
      b_initializer: Optional[Union[Callable, ArrayType, Initializer]] = ZeroInit(),
      mask: Optional[ArrayType] = None,
      precision: Optional[lax.Precision] = None,
      mode: bm.Mode = None,
      name: str = None,
  ):
    """Initializes the module.

    Args:
      output_channels: Number of output channels.
      kernel_shape: The shape of the kernel. Either an integer or a sequence of
        length 3.
      stride: Optional stride for the kernel. Either an integer or a sequence of
        length 3. Defaults to 1.
      output_shape: Output shape of the spatial dimensions of a transpose
        convolution. Can be either an integer or an iterable of integers. If a
        `None` value is given, a default shape is automatically calculated.
      padding: Optional padding algorithm. Either ``VALID`` or ``SAME``.
        Defaults to ``SAME``. See:
        https://www.tensorflow.org/xla/operation_semantics#conv_convolution.
      with_bias: Whether to add a bias. By default, true.
      w_init: Optional weight initialization. By default, truncated normal.
      b_init: Optional bias initialization. By default, zeros.
      data_format: The data format of the input. Either ``NDHWC`` or ``NCDHW``.
        By default, ``NDHWC``.
      mask: Optional mask of the weights.
      name: The name of the module.
    """
    super().__init__(
      num_spatial_dims=3,
      in_channels=in_channels,
      out_channels=out_channels,
      kernel_size=kernel_size,
      stride=stride,
      padding=padding,
      w_initializer=w_initializer,
      b_initializer=b_initializer,
      precision=precision,
      mode=mode,
      mask=mask,
      name=name
    )

  def _check_input_dim(self, x):
    if x.ndim != 5:
      raise ValueError(f"expected 5D input (got {x.ndim}D input)")
    if self.in_channels != x.shape[-1]:
      raise ValueError(f"input channels={x.shape[-1]} needs to have "
                       f"the same size as in_channels={self.in_channels}.")
