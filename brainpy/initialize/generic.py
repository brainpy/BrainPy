# -*- coding: utf-8 -*-

from typing import Union, Callable, Optional

import jax.numpy as jnp
import numpy as np

import brainpy.math as bm
from brainpy.tools.others import to_size
from brainpy.types import Shape, Tensor
from brainpy.modes import Mode, NonBatching, Batching, Training
from .base import Initializer


__all__ = [
  'parameter',
  'variable',
  'noise',
  'delay',

  # deprecated
  'init_param',
]


def parameter(
    param: Union[Callable,
                 Initializer,
                 bm.ndarray,
                 np.ndarray,
                 jnp.ndarray,
                 float, int, bool],
    size: Shape,
    allow_none: bool = True,
):
  """Initialize parameters.

  Parameters
  ----------
  param: callable, Initializer, bm.ndarray, jnp.ndarray, onp.ndarray, float, int, bool
    The initialization of the parameter.
    - If it is None, the created parameter will be None.
    - If it is a callable function :math:`f`, the ``f(size)`` will be returned.
    - If it is an instance of :py:class:`brainpy.init.Initializer``, the ``f(size)`` will be returned.
    - If it is a tensor, then this function check whether ``tensor.shape`` is equal to the given ``size``.
  size: int, sequence of int
    The shape of the parameter.
  allow_none: bool
    Whether allow the parameter is None.

  Returns
  -------
  param: JaxArray, float, None
    The initialized parameter.
  """
  if param is None:
    if allow_none:
      return None
    else:
      raise ValueError(f'Expect a parameter with type of float, JaxArray, Initializer, or '
                       f'Callable function, but we got None. ')
  size = to_size(size)
  if isinstance(param, (float, int, bool)):
    return param
  elif callable(param):
    param = bm.asarray(param(size))
  elif isinstance(param, (np.ndarray, jnp.ndarray)):
    param = bm.asarray(param)
  elif isinstance(param, bm.Variable):
    param = param
  elif isinstance(param, bm.JaxArray):
    param = param
  else:
    raise ValueError(f'Unknown param type {type(param)}: {param}')
  if param.shape != () and param.shape != (1,) and param.shape != size:
    raise ValueError(f'The shape of the parameters should be (), (1,) '
                     f'or {size}, but we got {param.shape}')
  return param


def init_param(
    param: Union[Callable, Initializer, bm.ndarray, jnp.ndarray, float, int, bool],
    size: Shape,
    allow_none: bool = True,
):
  return parameter(param, size, allow_none)


def variable(
    data: Union[Callable, Tensor],
    batch_size_or_mode: Optional[Union[int, bool, Mode]] = None,
    var_shape: Shape = None,
    batch_axis: int = 0,
):
  var_shape = to_size(var_shape)
  if callable(data):
    if var_shape is None:
      raise ValueError('"varshape" cannot be None when data is a callable function.')
    if isinstance(batch_size_or_mode, NonBatching):
      return bm.Variable(data(var_shape))
    elif isinstance(batch_size_or_mode, Batching):
      new_shape = var_shape[:batch_axis] + (1,) + var_shape[batch_axis:]
      return bm.Variable(data(new_shape), batch_axis=batch_axis)
    elif batch_size_or_mode in (None, False):
      return bm.Variable(data(var_shape))
    else:
      new_shape = var_shape[:batch_axis] + (int(batch_size_or_mode),) + var_shape[batch_axis:]
      return bm.Variable(data(new_shape), batch_axis=batch_axis)
  else:
    if var_shape is not None:
      if bm.shape(data) != var_shape:
        raise ValueError(f'The shape of "data" {bm.shape(data)} does not match with "var_shape" {var_shape}')
    if isinstance(batch_size_or_mode, NonBatching):
      return bm.Variable(data(var_shape))
    elif isinstance(batch_size_or_mode, Batching):
      return bm.Variable(bm.expand_dims(data, axis=batch_axis), batch_axis=batch_axis)
    elif batch_size_or_mode in (None, False):
      return bm.Variable(data)
    else:
      return bm.Variable(bm.repeat(bm.expand_dims(data, axis=batch_axis),
                                   int(batch_size_or_mode),
                                   axis=batch_axis),
                         batch_axis=batch_axis)


def noise(
    noises: Optional[Union[int, bm.ndarray, jnp.ndarray, Initializer, Callable]],
    size: Shape,
    num_vars: int = 1,
    noise_idx: int = 0,
) -> Optional[Callable]:
  if callable(noises):
    return noises
  elif noises is None:
    return None
  else:
    noises = parameter(noises, size, allow_none=False)
    if num_vars > 1:
      noises_ = [None] * num_vars
      noises_[noise_idx] = noises
      noises = tuple(noises_)
    return lambda *args, **kwargs: noises


def delay(
    delay_step: Union[int, bm.ndarray, jnp.ndarray, Callable, Initializer],
    delay_target: Union[bm.ndarray, jnp.ndarray],
    delay_data: Union[bm.ndarray, jnp.ndarray] = None
):
  """Initialize delay variable.

  Parameters
  ----------
  delay_step: int, ndarray, JaxArray
    The number of delay steps. It can an integer of an array of integers.
  delay_target: ndarray, JaxArray
    The target variable to delay.
  delay_data: optional, ndarray, JaxArray
    The initial delay data.

  Returns
  -------
  info: tuple
    The triple of delay type, delay steps, and delay variable.
  """
  # check delay type
  if delay_step is None:
    delay_type = 'none'
  elif isinstance(delay_step, int):
    delay_type = 'homo'
  elif isinstance(delay_step, (bm.ndarray, jnp.ndarray, np.ndarray)):
    delay_type = 'heter'
    delay_step = bm.asarray(delay_step)
  elif callable(delay_step):
    delay_step = parameter(delay_step, delay_target.shape, allow_none=False)
    delay_type = 'heter'
  else:
    raise ValueError(f'Unknown "delay_steps" type {type(delay_step)}, only support '
                     f'integer, array of integers, callable function, brainpy.init.Initializer.')
  if delay_type == 'heter':
    if delay_step.dtype not in [bm.int32, bm.int64]:
      raise ValueError('Only support delay steps of int32, int64. If your '
                       'provide delay time length, please divide the "dt" '
                       'then provide us the number of delay steps.')
    if delay_target.shape[0] != delay_step.shape[0]:
      raise ValueError(f'Shape is mismatched: {delay_target.shape[0]} != {delay_step.shape[0]}')

  # init delay data
  if delay_type == 'homo':
    delays = bm.LengthDelay(delay_target, delay_step, initial_delay_data=delay_data)
  elif delay_type == 'heter':
    if delay_step.size != delay_target.size:
      raise ValueError('Heterogeneous delay must have a length '
                       f'of the delay target {delay_target.shape}, '
                       f'while we got {delay_step.shape}')
    delays = bm.LengthDelay(delay_target, int(delay_step.max()))
  else:
    delays = None

  return delay_type, delay_step, delays

