# -*- coding: utf-8 -*-

from pprint import pprint

from brainpy import backend
from brainpy import errors
from brainpy.backend import ops
from brainpy.simulation import drivers
from . import utils

__all__ = [
  'TensorDiffIntDriver',
  'TensorDSDriver',
  'TensorNetDriver',
]


class TensorDiffIntDriver(drivers.BaseDiffIntDriver):
  def build(self, *args, **kwargs):
    # compile
    code = '\n'.join(self.code_lines)
    if self.show_code:
      print(code)
      print()
      pprint(self.code_scope)
      print()
    exec(compile(code, '', 'exec'), self.code_scope)

    # attribute assignment
    new_f = self.code_scope[self.func_name]
    for key, value in self.uploads.items():
      setattr(new_f, key, value)
    return new_f


class TensorDSDriver(drivers.BaseDSDriver):
  """BrainPy Running Driver for Tensor-oriented backends,
  such like NumPy, PyTorch, TensorFlow, etc.
  """

  def __init__(self, target):
    super(TensorDSDriver, self).__init__(target=target)
    self.last_inputs = {}
    self.formatted_funcs = {}
    self.run_func = None

  def _check_inputs_change(self, formatted_inputs, show_code):
    """Check whether the input is changed,
    including the data shape and the data operation.

    Parameters
    ----------
    formatted_inputs : list, tuple
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
    for key, val, op, data_type in formatted_inputs:
      # set data
      self.upload(self.input_data_name_of(key), val)
      # compare
      if key in old_input_keys:
        old_input_keys.remove(key)
        if ops.shape(self.last_inputs[key][0]) != ops.shape(val):
          input_keep_same = False
          if show_code:
            print(f'The current "{key}" input shape {ops.shape(val)} is different '
                  f'from the last input shape {ops.shape(self.last_inputs[key][0])}.')
        if self.last_inputs[key][1] != op:
          input_keep_same = False
          if show_code:
            print(f'The current "{key}" input operation "{op}" is different '
                  f'from the last operation "{self.last_inputs[key][1]}". ')
      else:
        input_keep_same = False
        if show_code:
          print(f'The input to a new key "{key}" in {self.target}.')
      new_inputs[key] = (val, op, data_type)
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
      code_scope = {host_name: self.target}
      code_lines = [f'def {input_func_name}(_i):']
      for key, val, ops, data_type in formatted_inputs:
        if ops == '=':
          line = f'  {host_name}.{key} = {host_name}.{self.input_data_name_of(key)}'
        else:
          line = f'  {host_name}.{key} {ops}= {host_name}.{self.input_data_name_of(key)}'
        if data_type == 'iter':
          line = line + '[_i]'
        code_lines.append(line)

      # function
      code = '\n'.join(code_lines)
      if show_code:
        print(code)
        print(code_scope)
        print()
      exec(compile(code, '', 'exec'), code_scope)
      func = code_scope[input_func_name]
    else:
      func = lambda _i: _i

    # results
    self.upload(input_func_name, func)
    self.formatted_funcs['input'] = {
      'func': func,
      'scope': {host_name: self.target},
      'call': [f'{host_name}.{input_func_name}(_i)'],
    }

  def get_input_func(self, formatted_inputs, show_code=False):
    input_keep_same = self._check_inputs_change(formatted_inputs=formatted_inputs, show_code=show_code)
    if not input_keep_same:
      self._format_inputs_func(formatted_inputs=formatted_inputs, show_code=show_code)
      need_rebuild = True
    else:
      need_rebuild = False
    return need_rebuild

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
    mon = self.target.mon
    if mon.num_item > 0:
      # build the monitor
      self.target.mon.build()

      # code lines, code scope
      host_name = self.target.name
      code_scope = {host_name: self.target, 'ops': ops}
      monitor_func_name = 'monitor_step'
      code_lines = [f'def {monitor_func_name}(_i, _t):']
      for key, idx, interval in zip(mon.item_names,
                                    mon.item_indices,
                                    mon.item_intervals):
        data = getattr(self.target, key)

        # get the data key in the host
        if isinstance(data, (int, float)):
          if idx is not None:
            raise errors.ModelUseError(f'"{self.target.name}.{key}" is a scalar, '
                                       f'cannot define the slice index "{idx}"')
          key_in_host = f'{host_name}.{key}'
        elif len(ops.shape(data)) == 1:
          key_in_host = f'{host_name}.{key}'
        else:
          key_in_host = f'ops.reshape({host_name}.{key}, (-1,))'

        # format the monitor index
        if idx is None:
          right = key_in_host
        else:
          right = f'{key_in_host}[{key}_idx_to_monitor]'
          code_scope[f'{key}_idx_to_monitor'] = idx

        # format the monitor lines according to the time interval
        if interval is None:
          lines = [f'  {host_name}.mon.{key}[_i] = {right}',
                   f'  {host_name}.mon.{key}_t[_i] = _t', ]
        else:
          num_interval = utils.every_to_step_num(interval)
          code_scope[f'{key}_interval_to_monitor'] = num_interval
          lines = [f'  if _i % {key}_interval_to_monitor == 0:',
                   f'    idx = int(_i / {key}_interval_to_monitor)',
                   f'    {host_name}.mon.{key}[idx] = {right}',
                   f'    {host_name}.mon.{key}_t[idx] = _t']

        # code line
        code_lines.extend(lines)

      # function
      code = '\n'.join(code_lines)
      if show_code:
        print(code)
        pprint(code_scope)
        print()
      exec(compile(code, '', 'exec'), code_scope)
      self.upload(monitor_func_name, code_scope[monitor_func_name])

      # results
      self.formatted_funcs['monitor'] = {
        'func': code_scope[monitor_func_name],
        'scope': {host_name: self.target},
        'call': [f'{host_name}.{monitor_func_name}(_i, _t)'],
      }

  def reshape_mon_items(self, mon_length):
    for key, interval in zip(self.target.mon.item_names, self.target.mon.item_intervals):
      if interval is None:
        num_interval = 1
      else:
        num_interval = utils.every_to_step_num(interval)
      mon_length = round(mon_length / num_interval)

      data = self.target.mon.item_contents[key]
      ts = self.target.mon.item_contents[f'{key}_t']
      shape = ops.shape(data)
      if mon_length < shape[0]:
        setattr(self.target.mon, key, data[:mon_length])
        setattr(self.target.mon, f'{key}_t', ts[:mon_length])
      elif mon_length > shape[0]:
        append1 = ops.zeros((mon_length - shape[0],) + shape[1:],
                            dtype=data.dtype if hasattr(data, 'dtype') else None)
        setattr(self.target.mon, key, ops.concatenate([data, append1], axis=0))
        append2 = ops.zeros((mon_length - shape[0],),
                            dtype=ts.dtype if hasattr(ts, 'dtype') else None)
        setattr(self.target.mon, f'{key}_t', ops.concatenate([ts, append2]))

  @staticmethod
  def step_lines_by_interval(step, lines, interval_name, code_scope):
    if hasattr(step, 'interval_time_to_run'):
      interval = step.interval_time_to_run
      if callable(interval):
        code_scope[interval_name] = interval
        line_calls = [f'if {interval_name}():']
      else:
        num_interval = utils.every_to_step_num(interval)
        code_scope[interval_name] = num_interval
        line_calls = [f'if _i % {interval_name} == 0:']
      line_calls += [f'  {line}' for line in lines]
    else:
      line_calls = lines
    return line_calls, code_scope

  def get_steps_func(self, show_code=False):
    for func_name, step in self.target.steps.items():
      class_args, arguments = utils.get_args(step)
      host_name = self.target.name

      # functional arguments
      calls = []
      for arg in arguments:
        if hasattr(self.target, arg):
          calls.append(f'{host_name}.{arg}')
        elif arg in backend.SYSTEM_KEYWORDS:
          calls.append(arg)
        else:
          raise errors.ModelDefError(f'Step function "{func_name}" of {self.target} '
                                     f'define an unknown argument "{arg}" which is not '
                                     f'an attribute of {self.target} nor the system '
                                     f'keywords {backend.SYSTEM_KEYWORDS}.')

      # format codes according to interval time
      line = f'{host_name}.{func_name}({", ".join(calls)})'
      line_calls, code_scope = self.step_lines_by_interval(
        step=step, lines=[line, ],
        interval_name=f'{host_name}_{func_name}_interval',
        code_scope={host_name: self.target})

      # formatted functions
      self.formatted_funcs[func_name] = {'func': step,
                                         'scope': code_scope,
                                         'call': line_calls}

  def build(self, formatted_inputs, mon_length, return_format_code=True, show_code=False):
    # inputs check
    # ------------
    assert isinstance(formatted_inputs, (tuple, list))
    need_rebuild = self.get_input_func(formatted_inputs, show_code=show_code)
    self.formatted_funcs['need_rebuild'] = need_rebuild

    # the run function does not build before
    # -------
    if self.run_func is None:
      # monitors
      self.get_monitor_func(show_code=show_code)

      # steps
      self.get_steps_func(show_code=show_code)

    # reshape the monitor
    self.reshape_mon_items(mon_length=mon_length)

    # build the model
    if need_rebuild or self.run_func is None:
      code_scope = dict()
      code_lines = ['def run_func(_t, _i, _dt):']
      for process in self.target.schedule():
        if (process not in self.formatted_funcs) and \
            (process in ['input', 'monitor']):
          continue
        process_result = self.formatted_funcs[process]
        code_scope.update(process_result['scope'])
        code_lines.extend(process_result['call'])

      # function
      code = '\n  '.join(code_lines)
      if show_code:
        print(code)
        pprint(code_scope)
        print()
      exec(compile(code, '', 'exec'), code_scope)
      self.run_func = code_scope['run_func']
      self.run_func.code = code

    if not return_format_code:
      return self.run_func
    else:
      return self.formatted_funcs

  @staticmethod
  def input_data_name_of(key):
    return f'_input_data_of_{key.replace(".", "_")}'


class TensorNetDriver():
  """General BrainPy Network Running Driver for NumPy, PyTorch, TensorFlow, etc.
  """

  def __init__(self, target):
    super(TensorNetDriver, self).__init__(target=target)
    assert hasattr(self.target, 'all_nodes') and isinstance(self.target.contained_members, dict)
    self.run_func = None

  def build(self, duration, formatted_inputs, show_code=False):
    """Build the network.

    Parameters
    ----------
    duration : int, float, tuple, list
        The running length.
    formatted_inputs : dict
        The user-defined inputs.
    show_code : bool
        Show the formatted code.

    Returns
    -------
    step_func : callable
        The step function.
    """
    # formatted step functions
    format_funcs_at_net = {'need_rebuild': False}
    for obj in self.target.contained_members.values():
      format_funcs = obj.build(inputs=formatted_inputs.get(obj.name, []),
                               inputs_is_formatted=True,
                               duration=duration,
                               return_format_code=True,
                               show_code=show_code)
      format_funcs_at_net['need_rebuild'] *= format_funcs.pop('need_rebuild')
      format_funcs_at_net.update({f'{obj.name}.{key}': val
                                  for key, val in format_funcs.items()})

    if (self.run_func is None) or format_funcs_at_net['need_rebuild']:
      # format code scope and code lines
      code_scope = {}
      code_lines = ['def run_func(_t, _i, _dt):']
      for p in self.target.schedule():
        if (p not in format_funcs_at_net) and \
            (p.split('.')[-1] in ['input', 'monitor']):
          continue
        p_codes = format_funcs_at_net[p]
        code_scope.update(p_codes['scope'])
        code_lines.extend(p_codes['call'])

      # compile the step function
      code = '\n  '.join(code_lines)
      if show_code:
        print(code)
        pprint(code_scope)
        print()
      exec(compile(code, '', 'exec'), code_scope)
      self.run_func = code_scope['run_func']
      self.run_func.code = code

    return self.run_func
