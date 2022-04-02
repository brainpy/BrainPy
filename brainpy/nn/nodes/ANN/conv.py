# -*- coding: utf-8 -*-


import jax.lax

import brainpy.math as bm
from brainpy.initialize import XavierNormal, ZeroInit, init_param
from brainpy.nn.base import Node

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
  print(input_shape)
  ndim = len(input_shape[0])
  lhs_spec = (0, ndim - 1) + tuple(range(1, ndim - 1))
  rhs_spec = (ndim - 1, ndim - 2) + tuple(range(0, ndim - 2))
  out_spec = lhs_spec
  return jax.lax.ConvDimensionNumbers(lhs_spec, rhs_spec, out_spec)


class GeneralConv(Node):
  """Applies a convolution to the inputs.
  """

  def __init__(self, in_channels, out_channels, kernel_size, strides=None, padding='SAME',
               input_dilation=None, kernel_dilation=None, groups=1, w_init=XavierNormal(), b_init=ZeroInit(), **kwargs):
    super(GeneralConv, self).__init__(**kwargs)

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
    self.trainable = True

    if isinstance(padding, str):
      assert padding in ['SAME', 'VALID']
    elif isinstance(padding, tuple):
      for k in padding:
        assert isinstance(k, int)
    else:
      raise ValueError

    assert in_channels % groups == 0, '"nin" should be divisible by groups'
    assert out_channels % groups == 0, '"nout" should be divisible by groups'

  def _check_input_dim(self):
    pass

  def init_ff_conn(self):
    input_shapes = self.feedforward_shapes
    kernel_shape = _check_tuple(self.kernel_size) + (self.in_channels // self.groups, self.out_channels)
    self.w = init_param(self.w_init, kernel_shape)
    self.b = init_param(self.b_init, (self.out_channels,) + (1,) * len(self.kernel_size))
    if self.trainable:
      self.w = bm.TrainVar(self.w)
      self.b = bm.TrainVar(self.b)

    if self.strides is None:
      self.strides = (1,) * (len(input_shapes[0]) - 2)

    output_shapes = jax.lax.conv_transpose_shape_tuple(
      input_shapes, kernel_shape, self.strides, self.padding, dimension_numbers=self.dimension_numbers)
    self.set_output_shape(output_shapes)

  def init_fb_conn(self):
    pass

  def forward(self, ff, fb=None, **shared_kwargs):
    ff = ff[0]
    y = jax.lax.conv_general_dilated(lhs=ff.value if isinstance(ff, bm.JaxArray) else ff,
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


class Conv1D(GeneralConv):
  def __init__(self, in_channels, out_channels, kernel_size, **kwargs):
    super(Conv1D, self).__init__(in_channels, out_channels, kernel_size, **kwargs)

    self.dimension_numbers = ('NCW', 'WIO', 'NCW')

  def _check_input_dim(self):
    ndim = len(self.feedforward_shapes)
    if ndim != 3:
      raise ValueError(
        "expected 3D input (got {}D input)".format(ndim)
      )

    assert len(self.kernel_size) == 1, "expected 1D kernel size (got {}D input)".format(self.kernel_size)


class Conv2D(GeneralConv):
  def __init__(self, in_channels, out_channels, kernel_size, **kwargs):
    super(Conv2D, self).__init__(in_channels, out_channels, kernel_size, **kwargs)

    self.dimension_numbers = ('NCHW', 'HWIO', 'NCHW')

  def _check_input_dim(self):
    ndim = len(self.feedforward_shapes)
    if ndim != 4:
      raise ValueError(
        "expected 4D input (got {}D input)".format(ndim)
      )

    assert len(self.kernel_size) == 2, "expected 2D kernel size (got {}D input)".format(self.kernel_size)


class Conv3D(GeneralConv):
  def __init__(self, in_channels, out_channels, kernel_size, **kwargs):
    super(Conv3D, self).__init__(in_channels, out_channels, kernel_size, **kwargs)

    self.dimension_numbers = ('NCHWD', 'HWDIO', 'NCHWD')

  def _check_input_dim(self):
    ndim = len(self.feedforward_shapes)
    if ndim != 5:
      raise ValueError(
        "expected 5D input (got {}D input)".format(ndim)
      )

    assert len(self.kernel_size) == 3, "expected 3D kernel size (got {}D input)".format(self.kernel_size)


