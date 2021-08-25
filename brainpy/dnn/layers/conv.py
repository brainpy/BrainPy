# -*- coding: utf-8 -*-

from brainpy.dnn.base import Module
from brainpy.dnn.imports import jmath, jax
from brainpy.dnn.inits import XavierNormal, Initializer, ZeroInit

__all__ = [
  'Conv2D',
]


def _check_tuple(v):
  if isinstance(v, (tuple, list)):
    return tuple(v)
  elif isinstance(v, int):
    return (v, v)
  else:
    raise ValueError


class Conv2D(Module):
  """Apply a 2D convolution on a 4D-input batch of shape (N,C,H,W).

  Parameters
  ----------
  nin : int
    The number of channels of the input tensor.
  nout : int
    The number of channels of the output tensor.
  kernel_size : int, tuple of int
    The size of the convolution kernel, either tuple (height, width)
    or single number if they're the same.
  strides : int, tuple of int
    The convolution strides, either tuple (stride_y, stride_x) or
    single number if they're the same.
  dilations : int, tuple of int
    The spacing between kernel points (also known as astrous convolution),
    either tuple (dilation_y, dilation_x) or single number if they're the same.
  groups : int
    The number of input and output channels group. When groups > 1 convolution
    operation is applied individually for each group. nin and nout must both
    be divisible by groups.
  padding : int, str
    The padding of the input tensor, either "SAME", "VALID" or numerical values
    (low, high).
  w_init : Initializer
    The initializer for convolution kernel (a function that takes in a HWIO
    shape and returns a 4D matrix).
  b_init : Initializer
    The bias initialization.
  """

  def __init__(self, nin, nout, kernel_size, strides=1, dilations=1, groups=1,
               padding='SAME', w_init=XavierNormal(), b_init=ZeroInit(), name=None):
    super(Conv2D, self).__init__(name=name)

    assert nin % groups == 0, '"nin" should be divisible by groups'
    assert nout % groups == 0, '"nout" should be divisible by groups'

    self.strides = _check_tuple(strides)
    self.dilations = _check_tuple(dilations)
    if isinstance(padding, str):
      assert padding in ['SAME', 'VALID']
    elif isinstance(padding, tuple):
      assert len(padding) == 2
      for k in padding:
        isinstance(k, int)
    else:
      raise ValueError
    self.padding = padding
    self.groups = groups

    # weight initialization
    self.b = jmath.TrainVar(b_init((nout, 1, 1)))
    self.w = jmath.TrainVar(w_init((*_check_tuple(kernel_size), nin // groups, nout)))  # HWIO
    self.w_init = w_init
    self.b_init = b_init

  def __call__(self, x):
    nin = self.w.value.shape[2] * self.groups
    assert x.shape[1] == nin, (f'Attempting to convolve an input with {x.shape[1]} input channels '
                               f'when the convolution expects {nin} channels. For reference, '
                               f'self.w.value.shape={self.w.value.shape} and x.shape={x.shape}.')

    y = jax.lax.conv_general_dilated(lhs=x.value if isinstance(x, jmath.JaxArray) else x,
                                     rhs=self.w.value,
                                     window_strides=self.strides,
                                     padding=self.padding,
                                     rhs_dilation=self.dilations,
                                     feature_group_count=self.groups,
                                     dimension_numbers=('NCHW', 'HWIO', 'NCHW'))
    y += self.b.value
    return y
