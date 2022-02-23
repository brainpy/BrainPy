# -*- coding: utf-8 -*-

from functools import wraps
from brainpy.types import Tensor
from typing import Callable, Union, Sequence, Dict, Any

import jax.numpy as jnp
import numpy as onp

import brainpy.math as bm

__all__ = [
  'summation',
  'init_param',
  'check_shape',
  'online',
  'offline',
]


def summation(values: Union[Sequence[Tensor], Dict[Any, Tensor], Tensor]):
  if isinstance(values, (bm.ndarray, jnp.ndarray)):
    return values
  if isinstance(values, dict):
    values = list(values.values())
  elif isinstance(values, (tuple, list)):
    values = list(values)
  else:
    raise ValueError('Unknown types of tensors.')
  res = values[0]
  for v in values[1:]:
    res = res + v
  return res


def init_param(param, size):
  if param is None:
    return None
  elif callable(param):
    param = param(size)
  elif isinstance(param, (onp.ndarray, jnp.ndarray)):
    param = bm.asarray(param)
  elif isinstance(param, (bm.JaxArray,)):
    param = param
  else:
    raise ValueError(f'Unknown param type {type(param)}: {param}')
  assert param.shape == size, f'"param.shape" is not the required size {size}'
  return param


def check_shape(all_shapes, free_axes: Union[Sequence[int], int] = -1):
  # check "all_shapes"
  if isinstance(all_shapes, dict):
    all_shapes = tuple(all_shapes.values())
  elif isinstance(all_shapes, (tuple, list)):
    all_shapes = tuple(all_shapes)
  else:
    raise ValueError
  # maximum number of dimension
  max_dim = max([len(shape) for shape in all_shapes])
  all_shapes = [[1] * (max_dim - len(s)) + list(s) for s in all_shapes]
  # check "free_axes"
  type_ = 'seq'
  if isinstance(free_axes, int):
    free_axes = (free_axes,)
    type_ = 'int'
  elif isinstance(free_axes, (tuple, list)):
    free_axes = tuple(free_axes)
  assert isinstance(free_axes, tuple)
  free_axes = [(axis + max_dim if axis < 0 else axis) for axis in free_axes]
  fixed_axes = [i for i in range(max_dim) if i not in free_axes]
  # get all free shapes
  if type_ == 'int':
    free_shape = [shape[free_axes[0]] for shape in all_shapes]
  else:
    free_shape = [[shape[axis] for axis in free_axes] for shape in all_shapes]
  # get all assumed fixed shapes
  fixed_shapes = [[shape[axis] for shape in all_shapes] for axis in fixed_axes]
  max_fixed_shapes = [max(shape) for shape in fixed_shapes]
  # check whether they can broadcast compatible
  for i, shape in enumerate(fixed_shapes):
    if len(set(shape) - {1, max_fixed_shapes[i]}):
      raise ValueError(f'Shapes out of axes {free_axes} are not '
                       f'broadcast compatible: \n'
                       f'{all_shapes}')
  return free_shape, max_fixed_shapes



def online(fun: Callable):
  @wraps(fun)
  def train(self, *args, **kwargs):
    return fun(self, *args, **kwargs)

  train.train_mode = 'online'
  return train


def offline(fun: Callable):
  @wraps(fun)
  def train(self, *args, **kwargs):
    return fun(self, *args, **kwargs)

  train.train_mode = 'offline'
  return train
