# -*- coding: utf-8 -*-
import warnings
from typing import Union, Callable, Optional

import jax.numpy as jnp
import numpy as np

import brainpy.math as bm
from brainpy.tools.others import to_size
from brainpy.types import Shape, Array
from brainpy.modes import Mode, NormalMode, BatchingMode
from .base import Initializer

__all__ = [
  'parameter',
  'variable',
  'variable_',
  'noise',
  'delay',

  # deprecated
  'init_param',
]


def parameter(
    param: Union[Callable, Initializer, bm.ndarray, np.ndarray, jnp.ndarray, float, int, bool],
    size: Shape,
    allow_none: bool = True,
    allow_scalar: bool = True,
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
  allow_scalar: bool
    Whether allow the parameter is a scalar value.

  Returns
  -------
  param: JaxArray, float, int, bool, None
    The initialized parameter.

  See Also
  --------
  variable_, noise, delay
  """
  if param is None:
    if allow_none:
      return None
    else:
      raise ValueError(f'Expect a parameter with type of float, JaxArray, Initializer, or '
                       f'Callable function, but we got None. ')
  size = to_size(size)
  if allow_scalar and isinstance(param, (float, int, bool)):
    return param
  if callable(param):
    param = bm.asarray(param(size))
  elif isinstance(param, (np.ndarray, jnp.ndarray)):
    param = bm.asarray(param)
  elif isinstance(param, bm.Variable):
    param = param
  elif isinstance(param, bm.JaxArray):
    param = param
  else:
    raise ValueError(f'Unknown param type {type(param)}: {param}')
  if allow_scalar:
    if param.shape == () or param.shape == (1,):
      return param
  if param.shape != size:
    raise ValueError(f'The shape of the parameters should be {size}, but we got {param.shape}')
  return param


def init_param(
    param: Union[Callable, Initializer, bm.ndarray, jnp.ndarray, float, int, bool],
    size: Shape,
    allow_none: bool = True,
):
  """Initialize parameters. Same as ``parameter()``.

  .. deprecated:: 2.2.3.4
     Will be removed since version 2.4.0.
  """
  return parameter(param, size, allow_none)


def variable_(
    data: Union[Callable, Array],
    size: Shape = None,
    batch_size_or_mode: Optional[Union[int, bool, Mode]] = None,
    batch_axis: int = 0,
):
  """Initialize variables. Same as `variable()`.

  Parameters
  ----------
  data: callable, function, Array
    The data to be initialized as a ``Variable``.
  batch_size_or_mode: int, bool, Mode, optional
    The batch size, model ``Mode``, boolean state.
    This is used to specify the batch size of this variable.
    If it is a boolean or an instance of ``Mode``, the batch size will be 1.
    If it is None, the variable has no batch axis.
  size: Shape
    The shape of the variable.
  batch_axis: int
    The batch axis.

  Returns
  -------
  variable: bm.Variable
    The target ``Variable`` instance.

  See Also
  --------
  variable, parameter, noise, delay

  """
  return variable(data, batch_size_or_mode, size, batch_axis)


def variable(
    data: Union[Callable, Array],
    batch_size_or_mode: Optional[Union[int, bool, Mode]] = None,
    size: Shape = None,
    batch_axis: int = 0,
):
  """Initialize variables.

  Parameters
  ----------
  data: callable, function, Array
    The data to be initialized as a ``Variable``.
  batch_size_or_mode: int, bool, Mode, optional
    The batch size, model ``Mode``, boolean state.
    This is used to specify the batch size of this variable.
    If it is a boolean or an instance of ``Mode``, the batch size will be 1.
    If it is None, the variable has no batch axis.
  size: Shape
    The shape of the variable.
  batch_axis: int
    The batch axis.

  Returns
  -------
  variable: bm.Variable
    The target ``Variable`` instance.

  See Also
  --------
  variable_, parameter, noise, delay

  """
  size = to_size(size)
  if callable(data):
    if size is None:
      raise ValueError('"varshape" cannot be None when data is a callable function.')
    if isinstance(batch_size_or_mode, NormalMode):
      return bm.Variable(data(size))
    elif isinstance(batch_size_or_mode, BatchingMode):
      new_shape = size[:batch_axis] + (1,) + size[batch_axis:]
      return bm.Variable(data(new_shape), batch_axis=batch_axis)
    elif batch_size_or_mode in (None, False):
      return bm.Variable(data(size))
    elif isinstance(batch_size_or_mode, int):
      new_shape = size[:batch_axis] + (int(batch_size_or_mode),) + size[batch_axis:]
      return bm.Variable(data(new_shape), batch_axis=batch_axis)
    else:
      raise ValueError('Unknown batch_size_or_mode.')

  else:
    if size is not None:
      if bm.shape(data) != size:
        raise ValueError(f'The shape of "data" {bm.shape(data)} does not match with "var_shape" {size}')
    if isinstance(batch_size_or_mode, NormalMode):
      return bm.Variable(data)
    elif isinstance(batch_size_or_mode, BatchingMode):
      return bm.Variable(bm.expand_dims(data, axis=batch_axis), batch_axis=batch_axis)
    elif batch_size_or_mode in (None, False):
      return bm.Variable(data)
    elif isinstance(batch_size_or_mode, int):
      return bm.Variable(bm.repeat(bm.expand_dims(data, axis=batch_axis),
                                   int(batch_size_or_mode),
                                   axis=batch_axis),
                         batch_axis=batch_axis)
    else:
      raise ValueError('Unknown batch_size_or_mode.')


def noise(
    noises: Optional[Union[int, float, bm.ndarray, jnp.ndarray, Initializer, Callable]],
    size: Shape,
    num_vars: int = 1,
    noise_idx: int = 0,
) -> Optional[Callable]:
  """Initialize a noise function.

  Parameters
  ----------
  noises: Any
  size: Shape
    The size of the noise.
  num_vars: int
    The number of variables.
  noise_idx: int
    The index of the current noise among all noise variables.

  Returns
  -------
  noise_func: function, None
    The noise function.

  See Also
  --------
  variable_, parameter, delay

  """
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

  See Also
  --------
  variable_, parameter, noise
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
