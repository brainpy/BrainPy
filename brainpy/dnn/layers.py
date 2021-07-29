# -*- coding: utf-8 -*-

import collections
import inspect

from brainpy import errors
from brainpy.dnn import activations
from brainpy.dnn.imports import jmath, jax
from brainpy.dnn.initializers import XavierNormal, Initializer, ZerosInit
from brainpy.simulation.base import DynamicSystem, Container

__all__ = [
  # abstract class
  'Module', 'Sequential',

  # commonly used layers
  'Activation', 'Linear', 'Dropout', 'Conv2D',
]


def _check_config(f, config):
  pars_in_f = inspect.signature(f).parameters
  if 'config' in pars_in_f.keys():
    return {'config': config}
  else:
    return {}


def _check_args(args):
  return (args,) if not isinstance(args, tuple) else args


def _check_tuple(v):
  if isinstance(v, (tuple, list)):
    return tuple(v)
  elif isinstance(v, int):
    return (v, v)
  else:
    raise ValueError


class Module(DynamicSystem):
  """Basic DNN module.

  Parameters
  ----------
  name : str, optional
    The name of the module.
  """

  def __init__(self, name=None):
    super(Module, self).__init__(name=name, steps=None, monitors=None)

  def __call__(self, *args, **kwargs):
    raise NotImplementedError


class Sequential(Container):
  """Basic DNN sequential object.

  Parameters
  ----------
  arg_modules
    The modules without name specifications.
  name : str, optional
    The name of the sequential module.
  kwarg_modules
    The modules with name specifications.
  """

  def __init__(self, *arg_modules, name=None, **kwarg_modules):
    all_systems = collections.OrderedDict()
    # check "args"
    for module in arg_modules:
      if not isinstance(module, Module):
        raise errors.ModelUseError(f'Only support {Module.__name__}, '
                                   f'but we got {type(module)}.')
      all_systems[module.name] = module

    # check "kwargs"
    for key, module in kwarg_modules.items():
      if not isinstance(module, Module):
        raise errors.ModelUseError(f'Only support {Module.__name__}, '
                                   f'but we got {type(module)}.')
      all_systems[key] = module

    # initialize base class
    super(Sequential, self).__init__(name=name, steps=None, monitors=None,
                                     **all_systems)

  def __call__(self, *args, config=dict()):
    """Functional call.

    Parameters
    ----------
    args : list, tuple
      The *args arguments.
    config : dict
      The config arguments. The configuration used across modules.
      If the "__call__" function in submodule receives "config" arguments,
      This "config" parameter will be passed into this function.
    """
    keys = list(self.steps.keys())
    calls = list(self.steps.values())

    # module 0
    try:
      args = calls[0](*args, **_check_config(calls[0], config))
    except Exception as e:
      raise type(e)(f'Sequential [{keys[0]}] {calls[0]} {e}')

    # other modules
    for i in range(1, len(self.steps)):
      try:
        args = calls[i](*_check_args(args=args), **_check_config(calls[i], config))
      except Exception as e:
        raise type(e)(f'Sequential [{keys[i]}] {calls[i]} {e}')
    return args


class Activation(Module):
  def __init__(self, activation, name=None, **setting):
    super(Activation, self).__init__(name=name)
    self.activation = activations._get(activation)
    self.setting = setting

  def __call__(self, x):
    return self.activation(x, **self.setting)


class Linear(Module):
  """A fully connected layer implemented as the dot product of inputs and
  weights.

  Parameters
  ----------
  n_out : int
      Desired size or shape of layer output
  n_in : int
      The layer input size feeding into this layer
  w_init : Initializer
      Initializer for the weights.
  b_init : Initializer
      Initializer for the bias.
  name : str, optional
  """

  def __init__(self, n_in, n_out, w_init=XavierNormal(), b_init=ZerosInit(), name=None):
    self.n_out = n_out
    self.n_in = n_in

    self.w = w_init((n_in, n_out))
    self.b = b_init(n_out)
    super(Linear, self).__init__(name=name)

  def __call__(self, x):
    """Returns the results of applying the linear transformation to input x."""
    y = jmath.dot(x, self.w) + self.b
    return y


class Dropout(Module):
  """A layer that stochastically ignores a subset of inputs each training step.

  In training, to compensate for the fraction of input values dropped (`rate`),
  all surviving values are multiplied by `1 / (1 - rate)`.

  The parameter `shared_axes` allows to specify a list of axes on which
  the mask will be shared: we will use size 1 on those axes for dropout mask
  and broadcast it. Sharing reduces randomness, but can save memory.

  This layer is active only during training (`mode='train'`). In other
  circumstances it is a no-op.

  Originally introduced in the paper "Dropout: A Simple Way to Prevent Neural
  Networks from Overfitting" available under the following link:
  https://www.cs.toronto.edu/~hinton/absps/JMLRdropout.pdf

  Parameters
  ----------
  prob : float
    Probability to keep element of the tensor.
  """

  def __init__(self, prob, name=None):
    self.prob = prob
    super(Dropout, self).__init__(name=name)

  def __call__(self, x, config=dict()):
    if config.get('train', True):
      keep_mask = jmath.random.bernoulli(self.prob, x.shape)
      return jmath.where(keep_mask, x / self.prob, 0.)
    else:
      return x


class Conv2D(Module):
  """Apply a 2D convolution on a 4D-input batch of shape (N,C,H,W).

  Parameters
  ----------
  nin : int
    number of channels of the input tensor.
  nout : int
    number of channels of the output tensor.
  kernel_size : int, tuple of int
    size of the convolution kernel, either tuple (height, width) or single
    number if they're the same.
  strides : int, tuple of int
    convolution strides, either tuple (stride_y, stride_x) or single number
    if they're the same.
  dilations : int, tuple of int
    spacing between kernel points (also known as astrous convolution),
    either tuple (dilation_y, dilation_x) or single number if they're the same.
  groups : int
    number of input and output channels group. When groups > 1 convolution
    operation is applied individually for each group. nin and nout must both
    be divisible by groups.
  padding : int, str
    padding of the input tensor, either Padding.SAME, Padding.VALID or numerical values.
  w_init : Initializer
    initializer for convolution kernel (a function that takes in a HWIO
    shape and returns a 4D matrix).
  b_init : Initializer
    The bias initialization.
  """

  def __init__(self, nin, nout, kernel_size, strides=1, dilations=1, groups=1,
               padding='SAME', w_init=XavierNormal(), b_init=ZerosInit(), name=None):
    super(Conv2D, self).__init__(name=name)

    assert nin % groups == 0, 'nin should be divisible by groups'
    assert nout % groups == 0, 'nout should be divisible by groups'
    self.b = b_init((nout, 1, 1))
    self.w = w_init((*_check_tuple(kernel_size), nin // groups, nout))  # HWIO
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

    # others
    self.w_init = w_init
    self.b_init = b_init

  def __call__(self, x):
    nin = self.w.value.shape[2] * self.groups
    assert x.shape[1] == nin, (f'Attempting to convolve an input with {x.shape[1]} input channels '
                               f'when the convolution expects {nin} channels. For reference, '
                               f'self.w.value.shape={self.w.value.shape} and x.shape={x.shape}.')

    y = jax.lax.conv_general_dilated(lhs=x.value if isinstance(x, jmath.ndarray) else x,
                                     rhs=self.w.value,
                                     window_strides=self.strides,
                                     padding=self.padding,
                                     rhs_dilation=self.dilations,
                                     feature_group_count=self.groups,
                                     dimension_numbers=('NCHW', 'HWIO', 'NCHW'))
    y += self.b.value
    return y


class BatchNorm(Module):
  pass


class RNN(Module):
  pass
