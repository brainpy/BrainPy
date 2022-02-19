# -*- coding: utf-8 -*-

import jax.numpy as jnp
import numpy as onp

import brainpy.math as bm
from brainpy.base.base import Base

__all__ = [
  'Module',
  'FeedForwardModule',
  'FeedBackModule',
  'RecurrentModule',
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

    self._in_size = None
    self._out_size = None
    self._fb_nodes = []

  def init(self, x=None):
    raise NotImplementedError

  def reset(self, state=None):
    raise NotImplementedError

  def __call__(self, *args, **kwargs):
    return self.call(*args, **kwargs)

  def call(self, *args, **kwargs):
    raise NotImplementedError

  def __rshift__(self, other):
    pass

  def __lshift__(self, other):
    pass

  def __rrshift__(self, other):
    pass

  def __rlshift__(self, other):
    pass

  def __add__(self, other):
    pass

  @property
  def in_size(self):
    return self._in_size

  @in_size.setter
  def in_size(self, size):
    self._in_size = size

  @property
  def out_size(self):
    return self._out_size

  @out_size.setter
  def out_size(self, size):
    self._out_size = size

  @property
  def has_feedback(self):
    return len(self._fb_nodes) > 0


class FrozenModule(Module):
  pass


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


class Model(Module):
  pass

