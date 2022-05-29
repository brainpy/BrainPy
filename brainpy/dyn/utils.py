# -*- coding: utf-8 -*-


from typing import Union, Callable, Optional, Dict

import jax.numpy as jnp
import numpy as np
from jax.tree_util import tree_flatten

from brainpy import math as bm
from brainpy.initialize import init_param, Initializer
from brainpy.types import Shape
from brainpy.tools.checking import check_dict_data

__all__ = [
  'init_noise',
  'init_noise',
]


def serialize_kwargs(shared_kwargs: Optional[Dict]):
  """Serialize kwargs."""
  shared_kwargs = dict() if shared_kwargs is None else shared_kwargs
  check_dict_data(shared_kwargs,
                  key_type=str,
                  val_type=(bool, float, int, complex),
                  name='shared_kwargs')
  shared_kwargs = {key: shared_kwargs[key] for key in sorted(shared_kwargs.keys())}
  return str(shared_kwargs)

def check_data_batch_size(data, num_batch=None, batch_idx=0):
  leaves, tree = tree_flatten(data, is_leaf=lambda x: isinstance(x, bm.JaxArray))
  batches = [leaf.shape[batch_idx] for leaf in leaves]
  if len(set(batches)) != 1:
    raise ValueError('Batch sizes are not consistent among the given data. '
                     f'Got {set(batches)}. We expect only one batch size.')
  batch_size = batches[0]
  if (num_batch is not None) and batch_size != num_batch:
    raise ValueError(f'Batch size is not consistent with the expected {batch_size} != {num_batch}')
  return batch_size


def init_noise(
    noise: Optional[Union[int, bm.ndarray, jnp.ndarray, Initializer, Callable]],
    size: Shape,
    num_vars: int = 1,
    noise_idx: int = 0,
) -> Optional[Callable]:
  if callable(noise):
    return noise
  elif noise is None:
    return None
  else:
    noise = init_param(noise, size, allow_none=False)
    if num_vars > 1:
      noises = [None] * num_vars
      noises[noise_idx] = noise
      noise = tuple(noises)
    return lambda *args, **kwargs: noise


def init_delay(
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
    delay_step = init_param(delay_step, delay_target.shape, allow_none=False)
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
