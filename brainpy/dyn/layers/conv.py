# -*- coding: utf-8 -*-


from jax import lax
from typing import Union, Tuple, Optional, Sequence

from brainpy import math as bm, tools
from brainpy.dyn.base import DynamicalSystem
from brainpy.initialize import Initializer, XavierNormal, ZeroInit, parameter
from brainpy.modes import Mode, TrainingMode, training
from brainpy.types import Array

__all__ = [
  'Conv1D',
  'Conv2D',
  'Conv3D'
]


def to_dimension_numbers(num_spatial_dims: int, channels_last: bool, transpose: bool) -> lax.ConvDimensionNumbers:
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


class GeneralConv(DynamicalSystem):
  """Applies a convolution to the inputs.

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
  strides: int, sequence of int
    An integer or a sequence of `n` integers, representing the inter-window strides (default: 1).
  padding: str, sequence of int, sequence of tuple
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
  w_init: Initializer
    The initializer for the convolutional kernel.
  b_init: Initializer
    The initializer for the bias.
  mask: Array, Optional
    The optional mask of the weights.
  mode: Mode
    The computation mode of the current object. Default it is `training`.
  name: str, Optional
    The name of the object.
  """

  def __init__(
      self,
      num_spatial_dims: int,
      in_channels: int,
      out_channels: int,
      kernel_size: Union[int, Tuple[int, ...]],
      strides: Union[int, Tuple[int, ...]] = 1,
      padding: Union[str, Tuple[int, int], Sequence[Tuple[int, int]]] = 'SAME',
      lhs_dilation: Union[int, Tuple[int, ...]] = 1,
      rhs_dilation: Union[int, Tuple[int, ...]] = 1,
      groups: int = 1,
      w_init: Initializer = XavierNormal(),
      b_init: Initializer = ZeroInit(),
      mask: Optional[Array] = None,
      mode: Mode = training,
      name: str = None,
  ):
    super(GeneralConv, self).__init__(name=name, mode=mode)

    self.num_spatial_dims = num_spatial_dims
    self.in_channels = in_channels
    self.out_channels = out_channels
    self.strides = tools.replicate(strides, num_spatial_dims, 'strides')
    self.kernel_size = tools.replicate(kernel_size, num_spatial_dims, 'kernel_size')
    self.lhs_dilation = tools.replicate(lhs_dilation, num_spatial_dims, 'lhs_dilation')
    self.rhs_dilation = tools.replicate(rhs_dilation, num_spatial_dims, 'rhs_dilation')
    self.groups = groups
    self.w_init = w_init
    self.b_init = b_init
    self.mask = mask
    self.dimension_numbers = to_dimension_numbers(num_spatial_dims, channels_last=True, transpose=False)

    if isinstance(padding, str):
      assert padding in ['SAME', 'VALID']
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
                             f"or sequence of Tuple[int, int] length {num_spatial_dims}.")
          padding = tuple(padding)
    else:
      raise ValueError
    self.padding = padding

    assert self.out_channels % self.groups == 0, '"out_channels" should be divisible by groups'
    assert self.in_channels % self.groups == 0, '"in_channels" should be divisible by groups'

    kernel_shape = tuple(self.kernel_size) + (self.in_channels // self.groups, self.out_channels)
    bias_shape = (1,) * len(self.kernel_size) + (self.out_channels,)
    self.w = parameter(self.w_init, kernel_shape)
    self.b = parameter(self.b_init, bias_shape)
    if isinstance(self.mode, TrainingMode):
      self.w = bm.TrainVar(self.w)
      self.b = bm.TrainVar(self.b)

  def _check_input_dim(self, x):
    raise NotImplementedError

  def update(self, sha, x):
    self._check_input_dim(x)
    w = self.w.value
    if self.mask is not None:
      if self.mask.shape != self.w.shape:
        raise ValueError(f"Mask needs to have the same shape as weights. {self.mask.shape} != {self.w.shape}")
      w *= self.mask
    y = lax.conv_general_dilated(lhs=bm.as_jax(x),
                                 rhs=bm.as_jax(w),
                                 window_strides=self.strides,
                                 padding=self.padding,
                                 lhs_dilation=self.lhs_dilation,
                                 rhs_dilation=self.rhs_dilation,
                                 feature_group_count=self.groups,
                                 dimension_numbers=self.dimension_numbers)
    if self.b is None:
      return y
    else:
      return y + self.b.value

  def reset_state(self, batch_size=None):
    pass


class Conv1D(GeneralConv):
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
  padding: str, sequence of int, sequence of tuple
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
  w_init: Initializer
    The initializer for the convolutional kernel.
  b_init: Initializer
    The initializer for the bias.
  mask: Array, Optional
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
      strides: Union[int, Tuple[int, ...]] = 1,
      padding: Union[str, Tuple[int, int], Sequence[Tuple[int, int]]] = 'SAME',
      lhs_dilation: Union[int, Tuple[int, ...]] = 1,
      rhs_dilation: Union[int, Tuple[int, ...]] = 1,
      groups: int = 1,
      w_init: Initializer = XavierNormal(),
      b_init: Initializer = ZeroInit(),
      mask: Optional[Array] = None,
      mode: Mode = training,
      name: str = None,
  ):
    super(Conv1D, self).__init__(num_spatial_dims=1,
                                 in_channels=in_channels,
                                 out_channels=out_channels,
                                 kernel_size=kernel_size,
                                 strides=strides,
                                 padding=padding,
                                 lhs_dilation=lhs_dilation,
                                 rhs_dilation=rhs_dilation,
                                 groups=groups,
                                 w_init=w_init,
                                 b_init=b_init,
                                 mask=mask,
                                 mode=mode,
                                 name=name)

  def _check_input_dim(self, x):
    if x.ndim != 3:
      raise ValueError(f"expected 3D input (got {x.ndim}D input)")
    if self.in_channels != x.shape[-1]:
      raise ValueError(f"input channels={x.shape[-1]} needs to have "
                       f"the same size as in_channels={self.in_channels}.")


class Conv2D(GeneralConv):
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
  strides: int, sequence of int
    An integer or a sequence of `n` integers, representing the inter-window strides (default: 1).
  padding: str, sequence of int, sequence of tuple
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
  w_init: Initializer
    The initializer for the convolutional kernel.
  b_init: Initializer
    The initializer for the bias.
  mask: Array, Optional
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
      strides: Union[int, Tuple[int, ...]] = 1,
      padding: Union[str, Tuple[int, int], Sequence[Tuple[int, int]]] = 'SAME',
      lhs_dilation: Union[int, Tuple[int, ...]] = 1,
      rhs_dilation: Union[int, Tuple[int, ...]] = 1,
      groups: int = 1,
      w_init: Initializer = XavierNormal(),
      b_init: Initializer = ZeroInit(),
      mask: Optional[Array] = None,
      mode: Mode = training,
      name: str = None,
  ):
    super(Conv2D, self).__init__(num_spatial_dims=2,
                                 in_channels=in_channels,
                                 out_channels=out_channels,
                                 kernel_size=kernel_size,
                                 strides=strides,
                                 padding=padding,
                                 lhs_dilation=lhs_dilation,
                                 rhs_dilation=rhs_dilation,
                                 groups=groups,
                                 w_init=w_init,
                                 b_init=b_init,
                                 mask=mask,
                                 mode=mode,
                                 name=name)

  def _check_input_dim(self, x):
    if x.ndim != 4:
      raise ValueError(f"expected 4D input (got {x.ndim}D input)")
    if self.in_channels != x.shape[-1]:
      raise ValueError(f"input channels={x.shape[-1]} needs to have "
                       f"the same size as in_channels={self.in_channels}.")


class Conv3D(GeneralConv):
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
  strides: int, sequence of int
    An integer or a sequence of `n` integers, representing the inter-window strides (default: 1).
  padding: str, sequence of int, sequence of tuple
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
  w_init: Initializer
    The initializer for the convolutional kernel.
  b_init: Initializer
    The initializer for the bias.
  mask: Array, Optional
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
      strides: Union[int, Tuple[int, ...]] = 1,
      padding: Union[str, Tuple[int, int], Sequence[Tuple[int, int]]] = 'SAME',
      lhs_dilation: Union[int, Tuple[int, ...]] = 1,
      rhs_dilation: Union[int, Tuple[int, ...]] = 1,
      groups: int = 1,
      w_init: Initializer = XavierNormal(),
      b_init: Initializer = ZeroInit(),
      mask: Optional[Array] = None,
      mode: Mode = training,
      name: str = None,
  ):
    super(Conv3D, self).__init__(num_spatial_dims=3,
                                 in_channels=in_channels,
                                 out_channels=out_channels,
                                 kernel_size=kernel_size,
                                 strides=strides,
                                 padding=padding,
                                 lhs_dilation=lhs_dilation,
                                 rhs_dilation=rhs_dilation,
                                 groups=groups,
                                 w_init=w_init,
                                 b_init=b_init,
                                 mask=mask,
                                 mode=mode,
                                 name=name)

  def _check_input_dim(self, x):
    if x.ndim != 5:
      raise ValueError(f"expected 5D input (got {x.ndim}D input)")
    if self.in_channels != x.shape[-1]:
      raise ValueError(f"input channels={x.shape[-1]} needs to have "
                       f"the same size as in_channels={self.in_channels}.")
