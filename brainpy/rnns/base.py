# -*- coding: utf-8 -*-

import inspect

import jax.numpy as jnp
import numpy as onp

import brainpy.math as bm
from brainpy import errors
from brainpy.base.base import Base
from brainpy.base.collector import Collector

__all__ = [
  'Module',
  'Sequential',
]


def _check_args(args):
  if args is None:
    return tuple()
  elif isinstance(args, tuple):
    return args
  else:
    return (args,)


class Module(Base):
  """Basic module class."""

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
    super(Module, self).__init__(name=name)

  def __call__(self, *args, **kwargs):  # initialize variables
    return self.call(*args, **kwargs)

  def call(self, *args, **kwargs):
    raise NotImplementedError


class FeedForward(Module):
  """Feedforward motif for the brain modeling."""
  pass


class FeedBack(Module):
  """Feedback motif for the brain modeling."""
  pass


class Recurrent(Module):
  """Recurrent motif for the brain modeling."""
  pass
