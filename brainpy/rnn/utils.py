# -*- coding: utf-8 -*-

from collections import defaultdict
from typing import Iterable, Any, Callable
from functools import wraps

import jax.numpy as jnp
import numpy as onp

import brainpy.math as bm
from .base import Node


__all__ = [
  'init_param',
  'safe_defaultdict_copy',
  'check_all_nodes',
  'online',
  'offline',
]


def init_param(param, size):
  if param is None:
    return None
  elif callable(param):
    param = param(size)
  elif isinstance(param, (onp.ndarray, jnp.ndarray)):
    param = bm.asarray(param)
  elif isinstance(param, (bm.JaxArray, )):
    param = param
  else:
    raise ValueError(f'Unknown param type {type(param)}: {param}')
  assert param.shape == size, f'"param.shape" is not the required size {size}'
  return param


def safe_defaultdict_copy(d):
  new_d = defaultdict(list)
  for key, item in d.items():
    if isinstance(item, Iterable):
      new_d[key] = list(item)
    else:
      new_d[key] += item
  return new_d


def check_all_nodes(*nodes: Any):

  msg = "Impossible to link nodes: object {} is neither a " \
        "'brainpy.rnn.Node' nor a 'brainpy.rnn.Network'."
  for nn in nodes:
    if isinstance(nn, Iterable):
      for n in nn:
        if not isinstance(n, Node):
          raise TypeError(msg.format(n))
    else:
      if not isinstance(nn, Node):
        raise TypeError(msg.format(nn))


def online(fun: Callable):
  @wraps(fun)
  def train(self, *args, **kwargs):
    return fun(self, *args, **kwargs)
  train.mode = 'online'
  return train


def offline(fun: Callable):
  @wraps(fun)
  def train(self, *args, **kwargs):
    return fun(self, *args, **kwargs)
  train.mode = 'offline'
  return train
