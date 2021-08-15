# -*- coding: utf-8 -*-

import math as math2
import sys
from pprint import pprint

from brainpy import errors, math, tools
from brainpy.simulation.driver import base

__all__ = [
  'NumpyDSDriver',
]


def every_to_step_num(interval):
  num_interval = round(interval / math.get_dt())
  if math2.fmod(interval * 1000, math.get_dt() * 1000) != 0.:
    print(f'"{interval}" is not an integer multiple of the step '
          f'resolution ("{math.get_dt()}"). BrainPy adjust it '
          f'to "{num_interval * math.get_dt()}".')
  return num_interval


class NumpyDSDriver(base.DSDriver):
  """BrainPy Running Driver for Tensor-oriented backends,
  such like NumPy, PyTorch, TensorFlow, etc.
  """

  def __init__(self, target):
    super(NumpyDSDriver, self).__init__(target=target)
    self.last_inputs = {}
    self.input_data = {}
    self.has_built = False

  def split_input_target_and_variable(self, key):
    splits = key.split('.')
    target = '.'.join(splits[:-1])
    variable = splits[-1]
    return target, variable

  def _check_inputs_change(self, inputs, show_code):
    """Check whether the input is changed,
    including the data shape and the data operation.

    Parameters
    ----------
    inputs : list, tuple
        The formatted inputs
    show_code : bool
        Whether show the code

    Returns
    -------
    input_keep_same : bool
        Whether the input is changed.
    """
    new_inputs = {}
    input_keep_same = True
    old_input_keys = list(self.last_inputs.keys())
    for key, val, data_type, op in inputs:
      # set data
      self.input_data[key] = val
      # self.upload(self.input_data_name_of(key), val)

      # compare
      if key in old_input_keys:
        old_input_keys.remove(key)
        if math.shape(self.last_inputs[key][0]) != math.shape(val):
          input_keep_same = False
          if show_code:
            print(f'The current "{key}" input shape {math.shape(val)} is different '
                  f'from the last input shape {math.shape(self.last_inputs[key][0])}.')
        if self.last_inputs[key][1] != data_type:
          input_keep_same = False
          if show_code:
            print(f'The current "{key}" input type "{data_type}" is different '
                  f'from the last input type "{self.last_inputs[key][1]}". ')
        if self.last_inputs[key][2] != op:
          input_keep_same = False
          if show_code:
            print(f'The current "{key}" input operation "{op}" is different '
                  f'from the last operation "{self.last_inputs[key][2]}". ')
      else:
        input_keep_same = False
        if show_code:
          print(f'The input to a new key "{key}" in {self.target}.')
      new_inputs[key] = (val, data_type, op)
    self.last_inputs = new_inputs

    if len(old_input_keys):
      input_keep_same = False
      if show_code:
        print(f'The inputs of {old_input_keys} in {self.target} are not provided.')

    return input_keep_same

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
        left = f'{real_target.name}.{variable}'

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

  def get_input_func(self, inputs, show_code=False):
    input_keep_same = self._check_inputs_change(inputs=inputs,
                                                show_code=show_code)
    if not input_keep_same:
      self._format_inputs_func(formatted_inputs=inputs,
                               show_code=show_code)

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
    if not isinstance(data, math.ndarray):
      raise errors.ModelUseError(f'BrainPy cannot monitor '
                                 f'"{self.target.name}.{key}", '
                                 f'because it is a scalar.')
    else:
      if math.ndim(data) == 1:
        key_in_host = f'{node.name}.{key}.copy()'
      else:
        key_in_host = f'{node.name}.{key}.flatten().copy()'

    # format the monitor index
    if idx is None:
      right = key_in_host
    else:
      right = f'{key_in_host}[{key_id}_idx_to_monitor]'
      code_scope[f'{key_id}_idx_to_monitor'] = idx

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

  def get_monitor_func(self, show_code=False):
    """Get the monitor function according to the user's setting.

    This method will consider the following things:

    1. the monitor variable
    2. the monitor index
    3. the monitor interval

    Parameters
    ----------
    show_code : bool
        Whether show the code.
    """
    nodes = self.target.nodes()
    monitor_func_name = 'monitor_step'

    code_lines = []
    code_scope = {'sys': sys}
    code_scope_for_call = {}
    for node in [self.target] + list(nodes.unique().values()):
      mon = node.mon
      if mon.num_item > 0:
        # code lines, code scope
        code_scope[node.name] = node
        code_scope_for_call[node.name] = node
        for key, idx, interval in zip(mon.item_names,
                                      mon.item_indices,
                                      mon.item_intervals):
          # "key" : 1. variable, like "V" (brainpy.math.ndarray, )
          #       : 2. variable in the node, like "exc.V" ("exc" is a DynamicSystem Node, "V" is the variable)
          _scope, _lines = self._format_code_to_mon_key(node, key, idx, interval)
          code_scope.update(_scope)
          code_lines.extend(_lines)

    if len(code_lines):
      # function
      code, func = tools.code_lines_to_func(lines=code_lines,
                                            func_name=monitor_func_name,
                                            func_args=['_t', '_i'],
                                            scope=code_scope)
      if show_code:
        print(code)
        pprint(code_scope)
        print()
      self.target.monitor_step = func

  def build_mon(self):
    for node in [self.target] + list(self.target.nodes().unique().values()):
      node.mon.build()  # build the monitor
      for key in node.mon.item_contents.keys():
        node.mon.item_contents[key] = []  # reshape the monitor items

  @staticmethod
  def step_lines_by_interval(step, lines, interval_name, code_scope):
    if hasattr(step, 'interval_time_to_run'):
      interval = step.interval_time_to_run
      if callable(interval):
        code_scope[interval_name] = interval
        line_calls = [f'if {interval_name}():']
      else:
        num_interval = every_to_step_num(interval)
        code_scope[interval_name] = num_interval
        line_calls = [f'if _i % {interval_name} == 0:']
      line_calls += [f'  {line}' for line in lines]
    else:
      line_calls = lines
    return line_calls, code_scope

  def build(self, rebuild=False, inputs=()):
    # get input function or check inputs
    self.get_input_func(inputs)

    # reshape the monitor
    self.build_mon()

    if not self.has_built or rebuild:
      # get monitor function
      self.get_monitor_func()
      self.has_built = True

    # return self.run

  def run(self, _t, _i):
    self.target.monitor_step(_t, _i)
    self.target.input_step(_t, _i)
    for step in self.target.steps.values():
      step(_t, _i)

  @staticmethod
  def input_data_name_of(key):
    return f'driver.input_data["{key}"]'
