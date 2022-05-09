# -*- coding: utf-8 -*-

import warnings

import jax.numpy as jnp
import numpy as onp

import brainpy.math as bm
from brainpy.base.base import Base

__all__ = [
  'Module',
]


def _check_args(args):
  if args is None:
    return tuple()
  elif isinstance(args, tuple):
    return args
  else:
    return (args,)


class Module(Base):
  """Basic module class.

  .. deprecated:: 2.1.0
  """

  @staticmethod
  def get_param(param, size):
    return bm.TrainVar(Module.init_param(param, size))

  @staticmethod
  def init_param(param, size):
    if param is None:
      return None
    if callable(param):
      param = param(size)
    elif isinstance(param, onp.ndarray):
      param = bm.asarray(param)
    elif isinstance(param, (bm.JaxArray, jnp.ndarray)):
      pass
    else:
      raise ValueError(f'Unknown param type {type(param)}: {param}')
    assert param.shape == size, f'"param.shape" is not the required size {size}'
    return param

  def __init__(self, name=None):  # initialize parameters
    warnings.warn('Please use "brainpy.rnns.Module" instead. '
                  '"brainpy.layers.Module" is deprecated since '
                  'version 2.1.0.', DeprecationWarning)
    super(Module, self).__init__(name=name)

  def __call__(self, *args, **kwargs):  # initialize variables
    return self.call(*args, **kwargs)

  def call(self, *args, **kwargs):
    raise NotImplementedError

