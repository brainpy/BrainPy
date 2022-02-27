# -*- coding: utf-8 -*-

from functools import wraps
from typing import Callable, Union, Sequence, Dict, Any

import jax.numpy as jnp
import numpy as onp

import brainpy.math as bm
from brainpy.types import Tensor


__all__ = [
  'tensor_sum',
  'init_param',
  'check_shape',
  'check_shape_consistency',
  'check_shape_broadcastable',
  'check_dict_data',
]


def tensor_sum(values: Union[Sequence[Tensor], Dict[Any, Tensor], Tensor]):
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


def check_shape_consistency(shapes, free_axes=None, return_format_shapes=False):
  assert isinstance(shapes, (tuple, list)), f'Must be a sequence of shape. While we got {shapes}.'
  for shape in shapes:
    assert isinstance(shapes, (tuple, list)), (f'Must be a sequence of shape. While '
                                               f'we got one element is {shape}.')
  dims = onp.unique([len(shape) for shape in shapes])
  if len(dims) > 1:
    raise ValueError(f'The provided shape dimensions are not consistent. ')
  if free_axes is None:
    type_ = 'none'
    free_axes = ()
  elif isinstance(free_axes, (tuple, list)):
    type_ = 'seq'
    free_axes = tuple(free_axes)
  elif isinstance(free_axes, int):
    type_ = 'int'
    free_axes = (free_axes,)
  else:
    raise ValueError
  free_axes = [(dims[0] + axis if axis < 0 else axis) for axis in free_axes]
  all_shapes = []
  for shape in shapes:
    assert isinstance(shapes, (tuple, list)), (f'Must be a sequence of shape. While '
                                               f'we got one element is {shape}.')
    shape = tuple([sh for i, sh in enumerate(shape) if i not in free_axes])
    all_shapes.append(shape)
  unique_shape = tuple(set(all_shapes))
  if len(unique_shape) > 1:
    if len(free_axes):
      raise ValueError(f'The provided shape (without axes of {free_axes}) are not consistent.')
    else:
      raise ValueError(f'The provided shape are not consistent.')
  if return_format_shapes:
    if type_ == 'int':
      free_shapes = [shape[free_axes[0]] for shape in shapes]
    elif type_ == 'seq':
      free_shapes = [tuple([shape[axis] for axis in free_axes]) for shape in shapes]
    else:
      free_shapes = None
    return unique_shape[0], tuple(free_shapes)


def check_shape_broadcastable(shapes, free_axes=(), return_format_shapes=False):
  max_dim = max([len(shape) for shape in shapes])
  shapes = [[1] * (max_dim - len(s)) + list(s) for s in shapes]
  return check_shape_consistency(shapes, free_axes, return_format_shapes)


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


def check_dict_data(a_dict: Dict, key_type, val_type):
  assert isinstance(a_dict, dict), f'Must be a dict, while we got {type(a_dict)}'
  for key, value in a_dict.items():
    assert isinstance(key, str), (f'Must be a dict of ({key_type}, {val_type}), while we got '
                                  f'({type(key)}, {type(value)})')
    assert isinstance(value, val_type), ('Must be a dict of ({key_type}, {val_type}), while we got '
                                         f'({type(key)}, {type(value)})')