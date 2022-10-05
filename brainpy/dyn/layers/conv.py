# -*- coding: utf-8 -*-


import jax.lax

import brainpy.math as bm
from brainpy.dyn.base import DynamicalSystem
from brainpy.initialize import XavierNormal, ZeroInit, parameter
from brainpy.modes import Mode, TrainingMode, training

__all__ = [
  'GeneralConv',
  'Conv1D',
  'Conv2D',
  'Conv3D'
]


def _check_tuple(v):
  if isinstance(v, (tuple, list)):
    return tuple(v)
  elif isinstance(v, int):
    return (v, v)
  else:
    raise ValueError


def _conv_dimension_numbers(input_shape):
  """Computes the dimension numbers based on the input shape."""
  ndim = len(input_shape)
  lhs_spec = (0, ndim - 1) + tuple(range(1, ndim - 1))
  rhs_spec = (ndim - 1, ndim - 2) + tuple(range(0, ndim - 2))
  out_spec = lhs_spec
  return jax.lax.ConvDimensionNumbers(lhs_spec, rhs_spec, out_spec)


class GeneralConv(DynamicalSystem):
  """Applies a convolution to the inputs.

  Parameters
  ----------
  in_channels: integer
    number of input channels.
  out_channels: integer
    number of output channels.
  kernel_size: sequence[int]
    shape of the convolutional kernel. For 1D convolution,
    the kernel size can be passed as an integer. For all other cases, it must
    be a sequence of integers.
  strides: sequence[int]
    an integer or a sequence of `n` integers, representing the inter-window strides (default: 1).
  padding: str, sequence[int]
    either the string `'SAME'`, the string `'VALID'`, the string
    `'CIRCULAR'` (periodic boundary conditions), or a sequence of `n` `(low,
    high)` integer pairs that give the padding to apply before and after each
    spatial dimension. A single int is interpeted as applying the same padding
    in all dims and passign a single int in a sequence causes the same padding
    to be used on both sides.
  input_dilation: integer, sequence[int]
    an integer or a sequence of `n` integers, giving the
    dilation factor to apply in each spatial dimension of `inputs`
    (default: 1). Convolution with input dilation `d` is equivalent to
    transposed convolution with stride `d`.
  kernel_dilation: integer, sequence[int]
    an integer or a sequence of `n` integers, giving the
    dilation factor to apply in each spatial dimension of the convolution
    kernel (default: 1). Convolution with kernel dilation
    is also known as 'atrous convolution'.
  groups: integer, default 1.
    If specified divides the input
    features into groups.
  w_init: brainpy.init.Initializer
    initializer for the convolutional kernel.
  b_init: brainpy.init.Initializer
    initializer for the bias.
  """

  def __init__(
      self,
      in_channels,
      out_channels,
      kernel_size,
      strides=None,
      padding='SAME',
      input_dilation=None,
      kernel_dilation=None,
      groups=1,
      w_init=XavierNormal(),
      b_init=ZeroInit(),
      mode: Mode = training,
      name: str = None,
  ):
    super(GeneralConv, self).__init__(name=name, mode=mode)

    self.in_channels = in_channels
    self.out_channels = out_channels
    self.kernel_size = kernel_size
    self.strides = strides
    self.padding = padding
    self.input_dilation = input_dilation
    self.kernel_dilation = kernel_dilation
    self.groups = groups
    self.w_init = w_init
    self.b_init = b_init
    self.dimension_numbers = None

    if isinstance(padding, str):
      assert padding in ['SAME', 'VALID']
    elif isinstance(padding, tuple):
      for k in padding:
        assert isinstance(k, int)
    else:
      raise ValueError

    assert out_channels % self.groups == 0, '"nout" should be divisible by groups'

    assert self.in_channels % self.groups == 0, '"nin" should be divisible by groups'
    kernel_shape = _check_tuple(self.kernel_size) + (self.in_channels // self.groups, self.out_channels)
    self.w = parameter(self.w_init, kernel_shape)
    self.b = parameter(self.b_init, (1,) * len(self.kernel_size) + (self.out_channels,))
    if isinstance(self.mode, TrainingMode):
      self.w = bm.TrainVar(self.w)
      self.b = bm.TrainVar(self.b)

  def _check_input_dim(self, x):
    pass

  def update(self, sha, x):
    self._check_input_dim(x)
    if self.strides is None:
      self.strides = (1,) * (len(x.shape) - 2)
    y = jax.lax.conv_general_dilated(lhs=x.value if isinstance(x, bm.JaxArray) else x,
                                     rhs=self.w.value,
                                     window_strides=self.strides,
                                     padding=self.padding,
                                     lhs_dilation=self.input_dilation,
                                     rhs_dilation=self.kernel_dilation,
                                     feature_group_count=self.groups,
                                     dimension_numbers=self.dimension_numbers)
    if self.b is None:
      return y
    return y + self.b.value

  def reset_state(self, batch_size=None):
    pass


class Conv1D(GeneralConv):
  def __init__(
      self,
      in_channels,
      out_channels,
      kernel_size,
      **kwargs
  ):
    super(Conv1D, self).__init__(in_channels, out_channels, kernel_size, **kwargs)

    self.dimension_numbers = ('NWC', 'WIO', 'NWC')

  def _check_input_dim(self, x):
    ndim = len(x.shape)
    if ndim != 3:
      raise ValueError(
        "expected 3D input (got {}D input)".format(ndim)
      )
    if self.in_channels != x.shape[-1]:
      raise ValueError(
        f"input channels={x.shape[-1]} needs to have the same size as in_channels={self.in_channels}."
      )
    assert len(self.kernel_size) == 1, "expected 1D kernel size (got {}D input)".format(self.kernel_size)


class Conv2D(GeneralConv):
  def __init__(
      self,
      in_channels,
      out_channels,
      kernel_size,
      **kwargs
  ):
    super(Conv2D, self).__init__(in_channels, out_channels, kernel_size, **kwargs)

    self.dimension_numbers = ('NHWC', 'HWIO', 'NHWC')

  def _check_input_dim(self, x):
    ndim = len(x.shape)
    if ndim != 4:
      raise ValueError(
        "expected 4D input (got {}D input)".format(ndim)
      )
    if self.in_channels != x.shape[-1]:
      raise ValueError(
        f"input channels={x.shape[-1]} needs to have the same size as in_channels={self.in_channels}."
      )
    assert len(self.kernel_size) == 2, "expected 2D kernel size (got {}D input)".format(self.kernel_size)


class Conv3D(GeneralConv):
  def __init__(
      self,
      in_channels,
      out_channels,
      kernel_size,
      **kwargs
  ):
    super(Conv3D, self).__init__(in_channels, out_channels, kernel_size, **kwargs)

    self.dimension_numbers = ('NHWDC', 'HWDIO', 'NHWDC')

  def _check_input_dim(self, x):
    ndim = len(x.shape)
    if ndim != 5:
      raise ValueError(
        "expected 5D input (got {}D input)".format(ndim)
      )
    if self.in_channels != x.shape[-1]:
      raise ValueError(
        f"input channels={x.shape[-1]} needs to have the same size as in_channels={self.in_channels}."
      )
    assert len(self.kernel_size) == 3, "expected 3D kernel size (got {}D input)".format(self.kernel_size)
