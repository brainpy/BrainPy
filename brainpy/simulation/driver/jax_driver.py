# -*- coding: utf-8 -*-

import math as math2
import sys

from brainpy import math, errors, tools
from brainpy.simulation.driver.numpy_driver import NumpyDSDriver

__all__ = [
  'JaxDSDriver',
]


def every_to_step_num(interval):
  num_interval = round(interval / math.get_dt())
  if math2.fmod(interval * 1000, math.get_dt() * 1000) != 0.:
    print(f'"{interval}" is not an integer multiple of the step '
          f'resolution ("{math.get_dt()}"). BrainPy adjust it '
          f'to "{num_interval * math.get_dt()}".')
  return num_interval


class JaxDSDriver(NumpyDSDriver):
  def _format_inputs_func(self, formatted_inputs, show_code):
    input_func_name = 'input_step'
    host_name = self.target.name

    # codes
    if len(formatted_inputs) > 0:
      nodes = self.target.nodes(method='absolute') + self.target.nodes(method='relative')
      code_scope = {host_name: self.target, 'sys': sys}
      code_lines = []
      for key, val, data_type, op in formatted_inputs:
        target, variable = self.split_input_target_and_variable(key)
        if target == '':
          real_target = self.target
        else:
          real_target = nodes[target]

        # code scope
        code_scope[real_target.name] = real_target

        # code line left
        left = f'{real_target.name}.{variable}.value'

        # code line right
        right = f'{host_name}.{self.input_data_name_of(key)}'
        if data_type == 'iter':
          right = right + '[_i]'

        # code line
        if op == '=':
          line = f'{left} = {right}'
        else:
          line = f'{left} {op}= {right}'

        code_lines.append(line)

      # function
      code, func = tools.code_lines_to_func(
        lines=code_lines,
        func_name=input_func_name,
        func_args=['_t', '_i'],
        scope=code_scope,
        remind='\n'
               'Please check: '
               '1. whether the "iter" input is set to "fix". '
               '2. whether the dimensions are not match.\n')
      if show_code:
        print(code)
        print(code_scope)
        print()
    else:
      func = lambda _t, _i: None

    # results
    self.target.input_step = func

  def _format_code_to_mon_key(self, node, key, idx, interval):
    code_scope = {}
    code_lines = []

    # get data
    key_splits = key.split('.')
    key_id = "_".join(key_splits)
    data = node
    for s in key_splits:
      data = getattr(data, s)

    # get the data key in the host
    if isinstance(data, (int, float)):
      if idx is not None:
        raise errors.BrainPyError(f'"{self.target.name}.{key}" is a scalar, '
                                  f'cannot define the slice index "{idx}"')
      key_in_host = f'{node.name}.{key}'
    elif math.ndim(data) == 1:
      if isinstance(data, math.ndarray):
        key_in_host = f'{node.name}.{key}.value.copy()'
      else:
        key_in_host = f'{node.name}.{key}.copy()'
    else:
      if isinstance(data, math.ndarray):
        key_in_host = f'{node.name}.{key}.value.flatten().copy()'
      else:
        key_in_host = f'{node.name}.{key}.flatten().copy()'

    # format the monitor index
    if idx is None:
      right = key_in_host
    else:
      right = f'{key_in_host}[{key_id}_idx_to_monitor]'
      code_scope[f'{key_id}_idx_to_monitor'] = idx if isinstance(idx, math.ndarray) else idx

    # format the monitor lines according to the time interval
    if interval is None:
      code_lines.append(f'{node.name}.mon.item_contents["{key}"].append({right})')
    else:
      num_interval = every_to_step_num(interval)
      code_scope[f'{key_id}_interval_to_monitor'] = num_interval
      code_lines.extend([f'if _i % {key_id}_interval_to_monitor == 0:',
                         f'  {node.name}.mon.item_contents["{key}"].append({right})',
                         f'  {node.name}.mon.item_contents["{key}.t"].append(_t)'])

    return code_scope, code_lines
