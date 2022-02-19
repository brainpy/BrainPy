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
    elif callable(param):
      param = param(size)
    elif isinstance(param, onp.ndarray):
      param = bm.asarray(param)
    elif isinstance(param, (bm.JaxArray, jnp.ndarray)):
      param = bm.asarray(param)
    else:
      raise ValueError(f'Unknown param type {type(param)}: {param}')
    assert param.shape == size, f'"param.shape" is not the required size {size}'
    return param

  def __init__(self, name=None):  # initialize parameters
    super(Module, self).__init__(name=name)

  def init(self):
    pass

  def reset(self, state=None):
    pass

  def __call__(self, *args, **kwargs):  # initialize variables
    return self.call(*args, **kwargs)

  def call(self, *args, **kwargs):
    raise NotImplementedError

  def __rshift__(self, other):
    raise NotImplementedError

  def __rrshift__(self, other):
    raise NotImplementedError

  def __add__(self, other):
    raise NotImplementedError

  @property
  def in_size(self):
    raise

  @property
  def out_size(self):
    raise


class FeedForwardModule(Module):
  """Feedforward motif for the RNN modeling."""
  pass


class FeedBackModule(Module):
  """Feedback motif for the RNN modeling."""
  pass


class RecurrentModule(Module):
  """Recurrent motif for the RNN modeling."""

  # the output
  pass






