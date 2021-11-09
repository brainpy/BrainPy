# -*- coding: utf-8 -*-

from brainpy import math
from brainpy.simulation._imports import mjax, jax
from brainpy.simulation.initialize import XavierNormal, Initializer, ZeroInit
from .base import Module

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
  num_input : int
    The number of channels of the input tensor.
  num_output : int
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

  steps : tuple of str, tuple of function, dict of (str, function), optional
      The callable function, or a list of callable functions.
  monitors : None, list, tuple, datastructures.Monitor
      Variables to monitor.
  name : str, optional
      The name of the dynamic system.
  """

  def __init__(self, num_input, num_output, kernel_size, strides=1, dilations=1,
               groups=1, padding='SAME', w_init=XavierNormal(), b_init=ZeroInit(),
               has_bias=True, **kwargs):
    super(Conv2D, self).__init__(**kwargs)

    # parameters
    assert num_input % groups == 0, '"nin" should be divisible by groups'
    assert num_output % groups == 0, '"nout" should be divisible by groups'
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
    self.has_bias = has_bias

    # weight initialization
    self.b = mjax.TrainVar(b_init((num_output, 1, 1)))
    self.w = mjax.TrainVar(w_init((*_check_tuple(kernel_size), num_input // groups, num_output)))  # HWIO
    self.w_init = w_init
    if has_bias:
      self.b_init = b_init

  def update(self, x, **kwargs):
    nin = self.w.value.shape[2] * self.groups
    assert x.shape[1] == nin, (f'Attempting to convolve an input with {x.shape[1]} input channels '
                               f'when the convolution expects {nin} channels. For reference, '
                               f'self.w.value.shape={self.w.value.shape} and x.shape={x.shape}.')

    y = jax.lax.conv_general_dilated(lhs=x.value if isinstance(x, mjax.JaxArray) else x,
                                     rhs=self.w.value,
                                     window_strides=self.strides,
                                     padding=self.padding,
                                     rhs_dilation=self.dilations,
                                     feature_group_count=self.groups,
                                     dimension_numbers=('NCHW', 'HWIO', 'NCHW'))
    if self.has_bias: y += self.b.value
    return y
