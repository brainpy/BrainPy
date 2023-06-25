# -*- coding: utf-8 -*-

from typing import Union, Callable, Optional, Sequence

import jax
import jax.numpy as jnp
import numpy as np

import brainpy.math as bm
from brainpy.tools import to_size
from brainpy.types import Shape, ArrayType, Sharding
from .base import Initializer

__all__ = [
  'parameter',
  'variable',
  'variable_',
  'noise',
  'delay',
]


def _check_none(x, allow_none: bool = False):
  pass


def _is_scalar(x):
  return isinstance(x, (float, int, bool, complex))


def parameter(
    param: Union[Callable, Initializer, bm.ndarray, np.ndarray, jnp.ndarray, float, int, bool],
    sizes: Shape,
    allow_none: bool = True,
    allow_scalar: bool = True,
    sharding: Optional[Sharding] = None
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
  sizes: int, sequence of int
    The shape of the parameter.
  allow_none: bool
    Whether allow the parameter is None.
  allow_scalar: bool
    Whether allow the parameter is a scalar value.
  sharding: Sharding
    The axes for automatic array sharding.

  Returns
  -------
  param: ArrayType, float, int, bool, None
    The initialized parameter.

  See Also
  --------
  variable_, noise, delay
  """
  if param is None:
    if allow_none:
      return None
    else:
      raise ValueError(f'Expect a parameter with type of float, ArrayType, Initializer, or '
                       f'Callable function, but we got None. ')
  sizes = to_size(sizes)
  if allow_scalar and _is_scalar(param):
    return param

  if callable(param):
    param = param(sizes)  # TODO
    # return bm.jit(param, static_argnums=0, out_shardings=bm.sharding.get_sharding(axis_names))(size)

  elif isinstance(param, (np.ndarray, jnp.ndarray)):
    param = bm.asarray(param)
  elif isinstance(param, bm.Variable):
    param = param
  elif isinstance(param, bm.Array):
    param = param
  else:
    raise ValueError(f'Unknown param type {type(param)}: {param}')

  if allow_scalar:
    if param.shape == () or param.shape == (1,):
      return param
  if param.shape != sizes:
    raise ValueError(f'The shape of the parameters should be {sizes}, but we got {param.shape}')
  return bm.sharding.partition(param, sharding)


def variable_(
    init: Union[Callable, bm.Array, jax.Array],
    sizes: Shape = None,
    batch_or_mode: Optional[Union[int, bool, bm.Mode]] = None,
    batch_axis: int = 0,
    axis_names: Optional[Sequence[str]] = None,
    batch_axis_name: Optional[str] = None,
):
  """Initialize a :math:`~.Variable` from a callable function or a data.

  Parameters
  ----------
  init: callable, function, ArrayType
    The data to be initialized as a ``Variable``.
  batch_or_mode: int, bool, Mode, optional
    The batch size, model ``Mode``, boolean state.
    This is used to specify the batch size of this variable.
    If it is a boolean or an instance of ``Mode``, the batch size will be 1.
    If it is None, the variable has no batch axis.
  sizes: Shape
    The shape of the variable.
  batch_axis: int
    The batch axis.
  axis_names: sequence of str
    The name for each axis. These names should match the given ``axes``.
  batch_axis_name: str
    The name for the batch axis. The name will be used if ``batch_size_or_mode`` is given.

  Returns
  -------
  variable: bm.Variable
    The target ``Variable`` instance.

  See Also
  --------
  variable, parameter, noise, delay

  """
  return variable(init,
                  batch_or_mode,
                  sizes=sizes,
                  batch_axis=batch_axis,
                  axis_names=axis_names,
                  batch_axis_name=batch_axis_name)


def variable(
    init: Union[Callable, ArrayType],
    batch_or_mode: Optional[Union[int, bool, bm.Mode]] = None,
    sizes: Shape = None,
    batch_axis: int = 0,
    axis_names: Optional[Sequence[str]] = None,
    batch_axis_name: Optional[str] = None,
):
  """Initialize variables.

  Parameters
  ----------
  init: callable, function, ArrayType
    The data to be initialized as a ``Variable``.
  batch_or_mode: int, bool, Mode, optional
    The batch size, model ``Mode``, boolean state.
    This is used to specify the batch size of this variable.
    If it is a boolean or an instance of ``Mode``, the batch size will be 1.
    If it is None, the variable has no batch axis.
  sizes: Shape
    The shape of the variable.
  batch_axis: int
    The batch axis.
  axis_names: sequence of str
    The name for each axis. These names should match the given ``axes``.
  batch_axis_name: str
    The name for the batch axis. The name will be used if ``batch_size_or_mode`` is given.

  Returns
  -------
  variable: bm.Variable
    The target ``Variable`` instance.

  See Also
  --------
  variable_, parameter, noise, delay

  """

  sizes = to_size(sizes)
  if axis_names is not None:
    axis_names = list(axis_names)
    assert len(sizes) == len(axis_names)
    if batch_or_mode is not None and not isinstance(batch_or_mode, bm.NonBatchingMode):
      axis_names.insert(batch_axis, batch_axis_name)

  if callable(init):
    if sizes is None:
      raise ValueError('"varshape" cannot be None when data is a callable function.')
    if isinstance(batch_or_mode, bm.NonBatchingMode):
      data = bm.Variable(init(sizes), axis_names=axis_names)
    elif isinstance(batch_or_mode, bm.BatchingMode):
      new_shape = sizes[:batch_axis] + (batch_or_mode.batch_size,) + sizes[batch_axis:]
      data = bm.Variable(init(new_shape), batch_axis=batch_axis, axis_names=axis_names)
    elif batch_or_mode in (None, False):
      data = bm.Variable(init(sizes), axis_names=axis_names)
    elif isinstance(batch_or_mode, int):
      new_shape = sizes[:batch_axis] + (int(batch_or_mode),) + sizes[batch_axis:]
      data = bm.Variable(init(new_shape), batch_axis=batch_axis, axis_names=axis_names)
    else:
      raise ValueError(f'Unknown batch_size_or_mode: {batch_or_mode}')

  else:
    if sizes is not None:
      if bm.shape(init) != sizes:
        raise ValueError(f'The shape of "data" {bm.shape(init)} does not match with "var_shape" {sizes}')
    if isinstance(batch_or_mode, bm.NonBatchingMode):
      data = bm.Variable(init, axis_names=axis_names)
    elif isinstance(batch_or_mode, bm.BatchingMode):
      data = bm.Variable(bm.repeat(bm.expand_dims(init, axis=batch_axis),
                                   batch_or_mode.batch_size,
                                   axis=batch_axis),
                         batch_axis=batch_axis,
                         axis_names=axis_names)
    elif batch_or_mode in (None, False):
      data = bm.Variable(init, axis_names=axis_names)
    elif isinstance(batch_or_mode, int):
      data = bm.Variable(bm.repeat(bm.expand_dims(init, axis=batch_axis),
                                   int(batch_or_mode),
                                   axis=batch_axis),
                         batch_axis=batch_axis,
                         axis_names=axis_names)
    else:
      raise ValueError('Unknown batch_size_or_mode.')
  return bm.sharding.partition_by_axname(data, axis_names)


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
  delay_step: int, ndarray, ArrayType
    The number of delay steps. It can an integer of an array of integers.
  delay_target: ndarray, ArrayType
    The target variable to delay.
  delay_data: optional, ndarray, ArrayType
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
