# -*- coding: utf-8 -*-


from collections.abc import Iterable
from typing import Union, Callable
import jax.numpy as jnp
import numpy as np

from brainpy import math as bm
from brainpy.dyn.base import DynamicalSystem
from brainpy.errors import RunningError
from brainpy.running.monitor import Monitor
from brainpy.initialize import init_param, Initializer

__all__ = [
  'size2len',
  'init_delay',
  'check_and_format_inputs',
  'check_and_format_monitors',
]

SUPPORTED_INPUT_OPS = ['-', '+', '*', '/', '=']
SUPPORTED_INPUT_TYPE = ['fix', 'iter', 'func']


def init_delay(delay_step: Union[int, bm.ndarray, jnp.ndarray, Callable, Initializer],
               delay_target: Union[bm.ndarray, jnp.ndarray],
               delay_data: Union[bm.ndarray, jnp.ndarray] = None):
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


def size2len(size):
  if isinstance(size, int):
    return size
  elif isinstance(size, (tuple, list)):
    a = 1
    for b in size:
      a *= b
    return a
  else:
    raise ValueError


def check_and_format_inputs(host, inputs):
  """Check inputs and get the formatted inputs for the given population.

  Parameters
  ----------
  host : DynamicalSystem
      The host which contains all data.
  inputs : tuple, list
      The inputs of the population.

  Returns
  -------
  formatted_inputs : tuple, list
      The formatted inputs of the population.
  """

  # 1. check inputs
  # ---------
  if inputs is None:
    inputs = []
  if not isinstance(inputs, (tuple, list)):
    raise RunningError('"inputs" must be a tuple/list.')
  if len(inputs) > 0 and not isinstance(inputs[0], (list, tuple)):
    if isinstance(inputs[0], str):
      inputs = [inputs]
    else:
      raise RunningError('Unknown input structure, only support inputs '
                         'with format of "(target, value, [type, operation])".')
  for one_input in inputs:
    if not 2 <= len(one_input) <= 4:
      raise RunningError('For each target, you must specify '
                         '"(target, value, [type, operation])".')
    if len(one_input) == 3 and one_input[2] not in SUPPORTED_INPUT_TYPE:
      raise RunningError(f'Input type only supports '
                         f'"{SUPPORTED_INPUT_TYPE}", '
                         f'not "{one_input[2]}".')
    if len(one_input) == 4 and one_input[3] not in SUPPORTED_INPUT_OPS:
      raise RunningError(f'Input operation only supports '
                         f'"{SUPPORTED_INPUT_OPS}", '
                         f'not "{one_input[3]}".')

  # 2. get targets and attributes
  # ---------
  inputs_which_found_target = []
  inputs_not_found_target = []

  # checking 1: absolute access
  #    Check whether the input target node is accessible,
  #    and check whether the target node has the attribute
  nodes = host.nodes(method='absolute')
  nodes[host.name] = host
  for one_input in inputs:
    key = one_input[0]
    if not isinstance(key, str):
      raise RunningError(f'For each input, input[0] must be a string  to '
                         f'specify variable of the target, but we got {key}.')
    splits = key.split('.')
    target = '.'.join(splits[:-1])
    key = splits[-1]
    if target == '':
      real_target = host
    else:
      if target not in nodes:
        inputs_not_found_target.append(one_input)
        continue
      real_target = nodes[target]
    if not hasattr(real_target, key):
      raise RunningError(f'Input target key "{key}" is not defined in {real_target}.')
    inputs_which_found_target.append((real_target, key) + tuple(one_input[1:]))

  # checking 2: relative access
  #    Check whether the input target node is accessible
  #    and check whether the target node has the attribute
  if len(inputs_not_found_target):
    nodes = host.nodes(method='relative')
    for one_input in inputs_not_found_target:
      splits = one_input[0].split('.')
      target, key = '.'.join(splits[:-1]), splits[-1]
      if target not in nodes:
        raise RunningError(f'Input target "{target}" is not defined in {host}.')
      real_target = nodes[target]
      if not hasattr(real_target, key):
        raise RunningError(f'Input target key "{key}" is not defined in {real_target}.')
      inputs_which_found_target.append((real_target, key) + tuple(one_input[1:]))

  # 3. format inputs
  # ---------
  formatted_inputs = []
  for one_input in inputs_which_found_target:
    # input value
    data_value = one_input[2]

    # input type
    if len(one_input) >= 4:
      if one_input[3] == 'iter':
        if not isinstance(data_value, Iterable):
          raise ValueError(f'Input "{data_value}" for "{one_input[0]}.{one_input[1]}" '
                           f'is set to be "iter" type, however we got the value with '
                           f'the type of {type(data_value)}')
      elif one_input[3] == 'func':
        if not callable(data_value):
          raise ValueError(f'Input "{data_value}" for "{one_input[0]}.{one_input[1]}" '
                           f'is set to be "func" type, however we got the value with '
                           f'the type of {type(data_value)}')
      elif one_input[3] != 'fix':
        raise RunningError(f'Only support {SUPPORTED_INPUT_TYPE} input type, but '
                           f'we got "{one_input[3]}" in {one_input}')

      data_type = one_input[3]
    else:
      data_type = 'fix'

    # operation
    if len(one_input) == 5:
      data_op = one_input[4]
    else:
      data_op = '+'
    if data_op not in SUPPORTED_INPUT_OPS:
      raise RunningError(f'Only support {SUPPORTED_INPUT_OPS}, while we got '
                         f'{data_op} in {one_input}')

    # final
    format_inp = one_input[:2] + (data_value, data_type, data_op)
    formatted_inputs.append(format_inp)

  return formatted_inputs


def check_and_format_monitors(host, mon):
  """Return a formatted monitor items:

  >>> [(node, key, target, variable, idx, interval),
  >>>  ...... ]

  """
  assert isinstance(host, DynamicalSystem)
  assert isinstance(mon, Monitor)

  formatted_mon_items = []

  # master node:
  #    Check whether the input target node is accessible,
  #    and check whether the target node has the attribute
  name2node = {node.name: node for node in list(host.nodes().unique().values())}
  for key, idx, interval in zip(mon.item_names, mon.item_indices, mon.item_intervals):
    # target and variable
    splits = key.split('.')
    if len(splits) == 1:
      if not hasattr(host, splits[0]):
        raise RunningError(f'{host} does not has variable {key}.')
      target = host
      variable = splits[-1]
    else:
      if not hasattr(host, splits[0]):
        if splits[0] not in name2node:
          raise RunningError(f'Cannot find target {key} in monitor of {host}, please check.')
        else:
          target = name2node[splits[0]]
          assert len(splits) == 2
          variable = splits[-1]
      else:
        target = host
        for s in splits[:-1]:
          try:
            target = getattr(target, s)
          except KeyError:
            raise RunningError(f'Cannot find {key} in {host}, please check.')
        variable = splits[-1]

    # idx
    if isinstance(idx, int): idx = bm.array([idx])

    # interval
    if interval is not None:
      if not isinstance(interval, float):
        raise RunningError(f'"interval" must be a float (denotes time), but we got {interval}')

    # append
    formatted_mon_items.append((key, target, variable, idx, interval,))

  return formatted_mon_items
