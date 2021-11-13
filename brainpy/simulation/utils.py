# -*- coding: utf-8 -*-

import sys
import time
from collections import Iterable
from pprint import pprint

from brainpy import errors, math, tools

__all__ = [
  'size2len',
  'check_duration',
  'run_model',
  'check_and_format_inputs',
  'build_input_func',
  'check_and_format_monitors',
  'build_monitor_func',
]

NORMAL_RUN = None
STRUCT_RUN = 'struct_run'
SUPPORTED_INPUT_OPS = ['-', '+', '*', '/', '=']
SUPPORTED_INPUT_TYPE = ['fix', 'iter']


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


def check_duration(duration):
  """Check the running duration.

  Parameters
  ----------
  duration : int, list, tuple
      The running duration, it can be an int (which represents the end
      of the simulation), of a tuple/list of int (which represents the
      [start, end] / [end, start] of the simulation).

  Returns
  -------
  duration : tuple
      The tuple of running duration includes (start, end).
  """
  if isinstance(duration, (int, float)):
    start, end = 0., duration
  elif isinstance(duration, (tuple, list)):
    assert len(duration) == 2, 'Only support duration setting with the ' \
                               'format of "(start, end)" or "end".'
    start, end = duration
  else:
    raise ValueError(f'Unknown duration type: {type(duration)}. Currently, BrainPy only '
                     f'support duration specification with the format of "(start, end)" '
                     f'or "end".')

  if start > end:
    start, end = end, start
  return start, end


def get_run_length_by_duration(duration):
  start, end = check_duration(duration)
  mon_length = int((end - start) / math.get_dt())
  return mon_length


def run_model(run_func, times, report, dt=None, extra_func=None):
  """Run the model.

  The "run_func" can be the step run function of a dynamical system.

  Parameters
  ----------
  run_func : callable
      The step run function.
  times : iterable
      The model running times.
  report : float
      The percent of the total running length for each report.
  """

  # numerical integration step
  if dt is None:
    dt = math.get_dt()
  assert isinstance(dt, (int, float))

  # running function
  if extra_func is None:
    running_func = run_func
  else:
    def running_func(t_and_dt):
      extra_func(*t_and_dt)
      run_func(t_and_dt)

  # simulations
  run_length = len(times)
  if report:
    t0 = time.time()
    running_func((times[0], dt))
    compile_time = time.time() - t0
    print('Compilation used {:.4f} s.'.format(compile_time))

    print("Start running ...")
    report_gap = int(run_length * report)
    t0 = time.time()
    for run_idx in range(1, run_length):
      running_func((times[run_idx], dt))
      if (run_idx + 1) % report_gap == 0:
        percent = (run_idx + 1) / run_length * 100
        print('Run {:.1f}% used {:.3f} s.'.format(percent, time.time() - t0))
    running_time = time.time() - t0
    print('Simulation is done in {:.3f} s.'.format(running_time))
    print()

  else:
    t0 = time.time()
    for run_idx in range(run_length):
      running_func((times[run_idx], dt))
    running_time = time.time() - t0

  return running_time


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
    raise errors.BrainPyError('"inputs" must be a tuple/list.')
  if len(inputs) > 0 and not isinstance(inputs[0], (list, tuple)):
    if isinstance(inputs[0], str):
      inputs = [inputs]
    else:
      raise errors.BrainPyError('Unknown input structure, only support inputs '
                                'with format of "(target, value, [type, operation])".')
  for one_input in inputs:
    if not 2 <= len(one_input) <= 4:
      raise errors.BrainPyError('For each target, you must specify '
                                '"(target, value, [type, operation])".')
    if len(one_input) == 3 and one_input[2] not in SUPPORTED_INPUT_TYPE:
      raise errors.BrainPyError(f'Input type only supports '
                                f'"{SUPPORTED_INPUT_TYPE}", '
                                f'not "{one_input[2]}".')
    if len(one_input) == 4 and one_input[3] not in SUPPORTED_INPUT_OPS:
      raise errors.BrainPyError(f'Input operation only supports '
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
      raise errors.BrainPyError(f'For each input, input[0] must be a string  to '
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
      raise errors.BrainPyError(f'Input target key "{key}" is not defined in {real_target}.')
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
        raise errors.BrainPyError(f'Input target "{target}" is not defined in {host}.')
      real_target = nodes[target]
      if not hasattr(real_target, key):
        raise errors.BrainPyError(f'Input target key "{key}" is not defined in {real_target}.')
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
      elif one_input[3] != 'fix':
        raise errors.BrainPyError(f'Only support {SUPPORTED_INPUT_TYPE} input type, but '
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
      raise errors.BrainPyError(f'Only support {SUPPORTED_INPUT_OPS}, while we got '
                                f'{data_op} in {one_input}')

    # final
    format_inp = one_input[:2] + (data_value, data_type, data_op)
    formatted_inputs.append(format_inp)

  return formatted_inputs


def build_input_func(inputs, show_code=False):
  input_func_name = 'input_step'
  code_scope = {'sys': sys}
  code_lines = []
  for target, key, value, type_, op in inputs:
    variable = getattr(target, key)

    # code scope
    code_scope[target.name] = target

    # code line left
    if isinstance(variable, math.Variable):
      left = f'{target.name}.{key}[:]'
    else:
      raise errors.BrainPyError(f'"{key}" in {target} is not a dynamically changed Variable, '
                                f'its value will not change, we think there is no need to '
                                f'give its input.')

    # code line right
    if type_ == 'iter':
      code_scope[f'{target.name}_input_data_of_{key}'] = iter(value)
      right = f'next({target.name}_input_data_of_{key})'
    else:
      code_scope[f'{target.name}_input_data_of_{key}'] = value
      right = f'{target.name}_input_data_of_{key}'

    # code line
    if op == '=':
      line = f'{left} = {right}'
    else:
      line = f'{left} {op}= {right}'

    code_lines.append(line)

  if len(code_lines):
    code_scope_old = {k: v for k, v in code_scope.items()}
    # function
    code, func = tools.code_lines_to_func(
      lines=code_lines,
      func_name=input_func_name,
      func_args=['_t', '_dt'],
      scope=code_scope,
      remind='Please check: \n'
             '1. whether the "iter" input is set to "fix". \n'
             '2. whether the dimensions are not match.\n')
    if show_code:
      print(code)
      print()
      pprint(code_scope_old)
      print()
  else:
    func = lambda _t, _dt: None

  return func


def check_and_format_monitors(host):
  """Return a formatted monitor items:

  >>> [(node, key, target, variable, idx, interval),
  >>>  ...... ]

  """
  formatted_mon_items = []

  # reshape monitors
  # ----
  all_nodes = list(host.nodes().unique().values())
  for node in all_nodes:
    if hasattr(node, 'mon'):
      node.mon.build()  # build the monitor
      for key in node.mon.item_contents.keys():
        node.mon.item_contents[key] = []  # reshape the monitor items

  # master node:
  #    Check whether the input target node is accessible,
  #    and check whether the target node has the attribute
  name2node = {node.name: node for node in all_nodes}
  for node in all_nodes:
    if hasattr(node, 'mon'):
      mon = node.mon
      for key, idx, interval in zip(mon.item_names, mon.item_indices, mon.item_intervals):
        # target and variable
        splits = key.split('.')
        if len(splits) == 1:
          if not hasattr(node, splits[0]):
            raise errors.BrainPyError(f'{node} does not has variable {key}.')
          target = node
          variable = splits[-1]
        else:
          if not hasattr(node, splits[0]):
            if splits[0] not in name2node:
              raise errors.BrainPyError(f'Cannot find target {key} in monitor of {node}, please check.')
            else:
              target = name2node[splits[0]]
              assert len(splits) == 2
              variable = splits[-1]
          else:
            target = node
            for s in splits[:-1]:
              try:
                target = getattr(target, s)
              except KeyError:
                raise errors.BrainPyError(f'Cannot find {key} in {node}, please check.')
            variable = splits[-1]

        # idx
        if isinstance(idx, int): idx = math.array([idx])

        # interval
        if interval is not None:
          if not isinstance(interval, float):
            raise errors.BrainPyError(f'"interval" must be a float (denotes time), but we got {interval}')

        # append
        formatted_mon_items.append((node, key, target, variable, idx, interval,))

  return formatted_mon_items


def build_monitor_func(monitors, show_code=False, func_name='monitor_step', method=None):
  """Get the monitor function according to the user's setting.

  This method will consider the following things:

  1. the monitor variable
  2. the monitor index
  3. the monitor interval

  Parameters
  ----------
  monitors : list, tuple
    The items to monitor.
  func_name : str
    The name of the monitor function.
  show_code : bool
      Whether show the code.
  """
  assert method in [None, STRUCT_RUN], f'Only support two kinds of method: None ' \
                                            f'or "{STRUCT_RUN}". But we got {method}'
  if method == STRUCT_RUN:
    assert math.is_jax_backend(), f'Running with "{STRUCT_RUN}" only supports "jax" backend.'
  code_lines = []
  code_scope = dict()
  returns = []
  assigns = []
  for node, key, target, variable, idx, interval in monitors:
    code_scope[node.name] = node
    code_scope[target.name] = target

    # get data
    data = target
    for k in variable.split('.'): data = getattr(data, k)

    # get the data key in the host
    if not isinstance(data, math.Variable):
      raise errors.BrainPyError(f'"{key}" in {target} is not a dynamically changed Variable, '
                                f'its value will not change, we think there is no need to '
                                f'monitor its trajectory.')
    if math.ndim(data) == 1:
      key_in_host = f'{target.name}.{variable}.value'
    else:
      key_in_host = f'{target.name}.{variable}.value.flatten()'
    if math.is_numpy_backend():
      key_in_host += '.copy()'

    # format the monitor index
    if idx is None:
      right = key_in_host
    else:
      if hasattr(idx, 'value'): idx = idx.value
      right = f'{key_in_host}[{node.name}_mon_{key.replace(".", "_")}_idx]'
      code_scope[f'{node.name}_mon_{key.replace(".", "_")}_idx'] = idx

    # format the monitor lines according to the time interval
    if method == STRUCT_RUN:
      returns.append(right)
      assigns.append([node.name, key])
      if interval is not None:
        raise ValueError(f'Running with "{STRUCT_RUN}" method does not '
                         f'support "interval" in the monitor.')
    else:
      if interval is None:
        code_lines.append(f'{node.name}.mon.item_contents["{key}"].append({right})')
      else:
        code_scope[f'{node.name}_mon_{key.replace(".", "_")}_next_time'] = interval
        code_lines.extend([f'global {node.name}_mon_{key.replace(".", "_")}_next_time',
                           f'if _t >= {node.name}_mon_{key.replace(".", "_")}_next_time:',
                           f'  {node.name}.mon.item_contents["{key}"].append({right})',
                           f'  {node.name}.mon.item_contents["{key}.t"].append(_t)',
                           f'  {node.name}_mon_{key.replace(".", "_")}_next_time += {interval}'])

  if len(code_lines) or len(returns):
    if method == STRUCT_RUN: code_lines.append(f'return {", ".join(returns)}')
    # function
    code_scope_old = {k: v for k, v in code_scope.items()}
    code, func = tools.code_lines_to_func(lines=code_lines,
                                          func_name=func_name,
                                          func_args=['_t', '_dt'],
                                          scope=code_scope)
    if show_code:
      print(code)
      print()
      pprint(code_scope_old)
      print()
  else:
    func = lambda _t, _dt: None
  return func, assigns
