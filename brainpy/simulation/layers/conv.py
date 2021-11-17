# -*- coding: utf-8 -*-


import jax.lax
import brainpy.math.jax as bm
from brainpy.simulation.initialize import Initializer, XavierNormal, ZeroInit
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
  w : Initializer, JaxArray, jax.numpy.ndarray
    The initializer for convolution kernel (a function that takes in a HWIO
    shape and returns a 4D matrix).
  b : Initializer, JaxArray, jax.numpy.ndarray, optional
    The bias initialization.

  steps : tuple of str, tuple of function, dict of (str, function), optional
      The callable function, or a list of callable functions.
  monitors : None, list, tuple, datastructures.Monitor
      Variables to monitor.
  name : str, optional
      The name of the dynamic system.
  """

  def __init__(self, num_input, num_output, kernel_size, strides=1, dilations=1,
               groups=1, padding='SAME', w=XavierNormal(), b=ZeroInit(), **kwargs):
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
    self.has_bias = True

    # weight initialization
    self.w = self.get_param(w, (*_check_tuple(kernel_size), num_input // groups, num_output))
    self.b = self.get_param(b, (num_output, 1, 1))

  def update(self, x):
    nin = self.w.value.shape[2] * self.groups
    assert x.shape[1] == nin, (f'Attempting to convolve an input with {x.shape[1]} input channels '
                               f'when the convolution expects {nin} channels. For reference, '
                               f'self.w.value.shape={self.w.value.shape} and x.shape={x.shape}.')

    y = jax.lax.conv_general_dilated(lhs=x.value if isinstance(x, bm.JaxArray) else x,
                                     rhs=self.w.value,
                                     window_strides=self.strides,
                                     padding=self.padding,
                                     rhs_dilation=self.dilations,
                                     feature_group_count=self.groups,
                                     dimension_numbers=('NCHW', 'HWIO', 'NCHW'))
    if self.b is None: return y
    return y + self.b.value
