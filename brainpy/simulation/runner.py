# -*- coding: utf-8 -*-

import logging
import sys
import time
from pprint import pprint

import jax.numpy as jnp
import numpy as np

from brainpy import math, tools
from brainpy.errors import RunningError, MonitorError
from brainpy.simulation import utils
from brainpy.simulation.brainobjects.base import DynamicalSystem
from brainpy.simulation.monitor import Monitor

logger = logging.getLogger('brainpy.simulation.runner')

__all__ = [
  'Runner',
  'ReportRunner',
  'StructRunner',
  'NumpyRunner',
]

_mon_func_name = 'monitor_step'
_input_func_name = 'input_step'


class Runner(object):
  """Basic Runner Class."""

  def __call__(self, *args, **kwargs):
    raise NotImplementedError


class BaseRunner(Runner):
  """The base runner class.

  Parameters
  ----------
  target : DynamicalSystem
    The target model to run.
  monitors : None, list of str, tuple of str, Monitor
    Variables to monitor.
  inputs : list, tuple
    The inputs for the target DynamicalSystem. It should be the format
    of `[(target, value, [type, operation])]`, where `target` is the
    input target, `value` is the input value, `type` is the input type
    (such as "fix" or "iter"), `operation` is the operation for inputs
    (such as "+", "-", "*", "/", "=").

    - ``target``: should be a string. Can be specified by the *absolute access* or *relative access*.
    - ``value``: should be a scalar, vector, matrix, iterable function or objects.
    - ``type``: should be a string. "fix" means the input `value` is a constant. "iter" means the
      input `value` can be changed over time.
    - ``operation``: should be a string, support `+`, `-`, `*`, `/`, `=`.
    - Also, if you want to specify multiple inputs, just give multiple ``(target, value, [type, operation])``,
      for example ``[(target1, value1), (target2, value2)]``.
  """

  def __init__(self, target, monitors=None, inputs=(),
               dt=None, jit=False, dyn_vars=None,
               numpy_mon_after_run=False):
    dt = math.get_dt() if dt is None else dt
    if not isinstance(dt, (int, float)):
      raise RunningError(f'"dt" must be scalar, but got {dt}')
    assert isinstance
    self.dt = dt
    self.jit = jit
    self.numpy_mon_after_run = numpy_mon_after_run

    # target
    if not isinstance(target, DynamicalSystem):
      raise RunningError(f'"target" must be an instance of {DynamicalSystem.__name__}, '
                         f'but we got {type(target)}: {target}')
    self.target = target

    # dynamical changed variables
    if dyn_vars is None:
      dyn_vars = self.target.vars().unique()
    if isinstance(dyn_vars, (list, tuple)):
      dyn_vars = {f'_v{i}': v for i, v in enumerate(dyn_vars)}
    if not isinstance(dyn_vars, dict):
      raise RunningError(f'"dyn_vars" must be a dict, but we got {type(dyn_vars)}')
    self.dyn_vars = dyn_vars

    # monitors
    if monitors is None:
      self.mon = Monitor(target=self, variables=[])
    elif isinstance(monitors, (list, tuple, dict)):
      self.mon = Monitor(target=self, variables=monitors)
    elif isinstance(monitors, Monitor):
      self.mon = monitors
      self.mon.target = self
    else:
      raise MonitorError(f'"monitors" only supports list/tuple/dict/ '
                         f'instance of Monitor, not {type(monitors)}.')
    self.mon.build()  # build the monitor
    # Build the monitor function
    #   All the monitors are wrapped in a single function.
    self._monitor_step = self.build_monitors()

    # Build input function
    inputs = utils.check_and_format_inputs(host=target, inputs=inputs)
    self._input_step = self.build_inputs(inputs)

    # start simulation time
    self._start_t = None

  def build_inputs(self, inputs, show_code=False):
    raise NotImplementedError

  def build_monitors(self, show_code=False):
    raise NotImplementedError

  def __call__(self, duration, start_t=None):
    raise NotImplementedError

  def run(self, duration, start_t=None):
    return self.__call__(duration, start_t=start_t)


class ReportRunner(BaseRunner):
  """The runner provides convenient interface for debugging.
  It is also able to report the running progress.

  Parameters
  ----------
  target : DynamicalSystem
    The target model to run.
  monitors : None, list of str, tuple of str, Monitor
    Variables to monitor.
  inputs : list, tuple
    The input settings.
  report : float
    The percent of progress to report.
  """

  def __init__(self, target, inputs=(), monitors=None, report=0.1, dyn_vars=None,
               jit=False, dt=None, numpy_mon_after_run=True):
    super(ReportRunner, self).__init__(target=target, inputs=inputs, monitors=monitors,
                                       jit=jit, dt=dt, dyn_vars=dyn_vars,
                                       numpy_mon_after_run=numpy_mon_after_run)

    self.report = report

    # Build the update function
    self._update_step = lambda _t, _dt: [_step(_t=_t, _dt=_dt)
                                         for _step in self.target.steps.values()]
    if jit:
      self._update_step = math.jit(self._update_step, dyn_vars=self.dyn_vars)

  def build_monitors(self, show_code=False):
    """Get the monitor function according to the user's setting.

    This method will consider the following things:

    1. the monitor variable
    2. the monitor index
    3. the monitor interval

    """
    monitors = utils.check_and_format_monitors(host=self.target, mon=self.mon)

    host = self.target
    code_lines = []
    code_scope = dict(sys=sys, self_mon=self.mon)
    for key, target, variable, idx, interval in monitors:
      code_scope[host.name] = host
      code_scope[target.name] = target

      # get data
      data = target
      for k in variable.split('.'): data = getattr(data, k)

      # get the data key in the host
      if not isinstance(data, math.Variable):
        raise RunningError(f'"{key}" in {target} is not a dynamically changed Variable, '
                           f'its value will not change, we think there is no need to '
                           f'monitor its trajectory.')
      if math.ndim(data) == 1:
        key_in_host = f'{target.name}.{variable}.value'
      else:
        key_in_host = f'{target.name}.{variable}.value.flatten()'

      # format the monitor index
      if idx is None:
        right = key_in_host
      else:
        idx = math.asarray(idx)
        right = f'{key_in_host}[_{key.replace(".", "_")}_idx]'
        code_scope[f'_{key.replace(".", "_")}_idx'] = idx.value

      # format the monitor lines according to the time interval
      if interval is None:
        code_lines.append(f'self_mon.item_contents["{key}"].append({right})')
      else:
        code_scope[f'_{key.replace(".", "_")}_next_time'] = interval
        code_lines.extend([f'global _{key.replace(".", "_")}_next_time',
                           f'if _t >= _{key.replace(".", "_")}_next_time:',
                           f'  self_mon.item_contents["{key}"].append({right})',
                           f'  self_mon.item_contents["{key}.t"].append(_t)',
                           f'  _{key.replace(".", "_")}_next_time += {interval}'])

    if len(code_lines):
      # function
      code_scope_old = {k: v for k, v in code_scope.items()}
      code, func = tools.code_lines_to_func(lines=code_lines,
                                            func_name=_mon_func_name,
                                            func_args=['_t', '_dt'],
                                            scope=code_scope)
      if show_code:
        print(code)
        print()
        pprint(code_scope_old)
        print()
    else:
      func = lambda _t, _dt: None
    return func

  def build_inputs(self, inputs, show_code=False):
    code_scope = {'sys': sys}
    code_lines = []
    for target, key, value, type_, op in inputs:
      variable = getattr(target, key)

      # code scope
      code_scope[target.name] = target

      # code line left
      if isinstance(variable, math.Variable):
        left = f'{target.name}.{key}'
      else:
        raise RunningError(f'"{key}" in {target} is not a dynamically changed Variable, '
                           f'its value will not change, we think there is no need to '
                           f'give its input.')

      # code line right
      if type_ == 'iter':
        code_scope[f'{target.name}_input_data_of_{key}'] = iter(value)
        right = f'next({target.name}_input_data_of_{key})'
      elif type_ == 'func':
        code_scope[f'{target.name}_input_data_of_{key}'] = value
        right = f'{target.name}_input_data_of_{key}(_t, _dt)'
      else:
        code_scope[f'{target.name}_input_data_of_{key}'] = value
        right = f'{target.name}_input_data_of_{key}'

      # code line
      if op == '=':
        line = f'{left}[:] = {right}'
      else:
        line = f'{left} {op}= {right}'

      code_lines.append(line)

    if len(code_lines):
      code_scope_old = {k: v for k, v in code_scope.items()}
      # function
      code, func = tools.code_lines_to_func(
        lines=code_lines,
        func_name=_input_func_name,
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

  def _step(self, t_and_dt):
    self._input_step(_t=t_and_dt[0], _dt=t_and_dt[1])
    self._update_step(_t=t_and_dt[0], _dt=t_and_dt[1])
    self._monitor_step(_t=t_and_dt[0], _dt=t_and_dt[1])

  def __call__(self, duration, start_t=None):
    """The running function.

    Parameters
    ----------
    duration : float, int, tuple, list
      The running duration.
    start_t : float, optional
      The start simulation time.

    Returns
    -------
    running_time : float
      The total running time.
    """
    # time step
    if start_t is None:
      if self._start_t is None:
        start_t = 0.
      else:
        start_t = self._start_t
    end_t = start_t + duration

    # times
    times = math.arange(start_t, end_t, self.dt)

    # build inputs
    for key in self.mon.item_contents.keys():
      self.mon.item_contents[key] = []  # reshape the monitor items

    # simulations
    run_length = len(times)
    if self.report > 0.:
      t0 = time.time()
      self._step((times[0], self.dt))
      compile_time = time.time() - t0
      print('Compilation used {:.4f} s.'.format(compile_time))

      print("Start running ...")
      report_gap = int(run_length * self.report)
      t0 = time.time()
      for run_idx in range(1, run_length):
        self._step((times[run_idx], self.dt))
        if (run_idx + 1) % report_gap == 0:
          percent = (run_idx + 1) / run_length * 100
          print('Run {:.1f}% used {:.3f} s.'.format(percent, time.time() - t0))
      running_time = time.time() - t0
      print('Simulation is done in {:.3f} s.'.format(running_time))
      print()

    else:
      t0 = time.time()
      for run_idx in range(run_length):
        self._step((times[run_idx], self.dt))
      running_time = time.time() - t0

    # monitor post steps
    self.mon.ts = times
    for key, val in self.mon.item_contents.items():
      self.mon.item_contents[key] = math.asarray(val)
    self._start_t = end_t
    if self.numpy_mon_after_run:
      self.mon.numpy()
    return running_time


class StructRunner(BaseRunner):
  """The runner with the structural for loops.

  Parameters
  ----------
  target : DynamicalSystem
    The target model to run.
  monitors : None, list of str, tuple of str, Monitor
    Variables to monitor.
  inputs : list, tuple
    The input settings.
  """

  def __init__(self, target, monitors=None, inputs=(), dyn_vars=None,
               jit=False, report=0., dt=None, numpy_mon_after_run=True):
    self._has_iter_array = False  # default do not have iterable input array
    super(StructRunner, self).__init__(target=target,
                                       inputs=inputs,
                                       monitors=monitors,
                                       jit=jit, dt=dt,
                                       dyn_vars=dyn_vars,
                                       numpy_mon_after_run=numpy_mon_after_run)
    # JAX does not support iterator in fori_loop, scan, etc.
    #   https://github.com/google/jax/issues/3567
    # We use Variable i to index the current input data.
    if self._has_iter_array:
      self._i = math.Variable(math.asarray([0]))
      self.dyn_vars.update({'_i': self._i})
    else:
      self._i = None

    # report
    if report > 0.: logger.warning(f'"report={report}" can not work in {self.__class__.__name__}.')

    # build the update step
    self._step = math.make_loop(self._step, dyn_vars=self.dyn_vars, has_return=True)
    if jit: self._step = math.jit(self._step, dyn_vars=dyn_vars)

  def _step(self, t_and_dt):  # the step function
    t, dt = t_and_dt[0], t_and_dt[1]
    self._input_step(_t=t, _dt=dt)
    for step in self.target.steps.values():
      step(_t=t, _dt=dt)
    return self._monitor_step(_t=t, _dt=dt)

  def _post(self, times, returns):  # monitor
    self.mon.ts = times
    for i, key in enumerate(self.mon.item_names):
      self.mon.item_contents[key] = math.asarray(returns[i])

  def build_monitors(self, show_code=False):
    monitors = utils.check_and_format_monitors(host=self.target, mon=self.mon)

    returns = []
    code_lines = []
    host = self.target
    code_scope = dict(sys=sys)
    for key, target, variable, idx, interval in monitors:
      code_scope[host.name] = host
      code_scope[target.name] = target

      # get data
      data = target
      for k in variable.split('.'): data = getattr(data, k)

      # get the data key in the host
      if not isinstance(data, math.Variable):
        raise RunningError(f'"{key}" in {target} is not a dynamically changed Variable, '
                           f'its value will not change, we think there is no need to '
                           f'monitor its trajectory.')
      if math.ndim(data) == 1:
        key_in_host = f'{target.name}.{variable}.value'
      else:
        key_in_host = f'{target.name}.{variable}.value.flatten()'

      # format the monitor index
      if idx is None:
        right = key_in_host
      else:
        idx = math.asarray(idx)
        right = f'{key_in_host}[_{key.replace(".", "_")}_idx]'
        code_scope[f'_{key.replace(".", "_")}_idx'] = idx.value

      # format the monitor lines according to the time interval
      returns.append(right)
      if interval is not None:
        raise ValueError(f'Running with "{self.__class__.__name__}" does '
                         f'not support "interval" in the monitor.')

    if len(code_lines) or len(returns):
      code_lines.append(f'return {", ".join(returns) + ", "}')
      # function
      code_scope_old = {k: v for k, v in code_scope.items()}
      code, func = tools.code_lines_to_func(lines=code_lines,
                                            func_name=_mon_func_name,
                                            func_args=['_t', '_dt'],
                                            scope=code_scope)
      if show_code:
        print(code)
        print()
        pprint(code_scope_old)
        print()
    else:
      func = lambda _t, _dt: None
    return func

  def build_inputs(self, inputs, show_code=False):
    code_scope = {'sys': sys, '_runner': self}
    code_lines = []
    for target, key, value, type_, op in inputs:
      variable = getattr(target, key)

      # code scope
      code_scope[target.name] = target

      # code line left
      if isinstance(variable, math.Variable):
        left = f'{target.name}.{key}'
      else:
        raise RunningError(f'"{key}" in {target} is not a dynamically changed Variable, '
                           f'its value will not change, we think there is no need to '
                           f'give its input.')

      # code line right
      if type_ == 'iter':
        if isinstance(value, (math.ndarray, np.ndarray, jnp.ndarray)):
          code_scope[f'{target.name}_input_data_of_{key}'] = math.asarray(value)
          right = f'{target.name}_input_data_of_{key}[_runner._i[0]]'
          self._has_iter_array = True
        else:
          code_scope[f'{target.name}_input_data_of_{key}'] = iter(value)
          right = f'next({target.name}_input_data_of_{key})'
      elif type_ == 'func':
        code_scope[f'{target.name}_input_data_of_{key}'] = value
        right = f'{target.name}_input_data_of_{key}(_t, _dt)'
      else:
        code_scope[f'{target.name}_input_data_of_{key}'] = value
        right = f'{target.name}_input_data_of_{key}'

      # code line
      if op == '=':
        line = f'{left}[:] = {right}'
      else:
        line = f'{left} {op}= {right}'

      code_lines.append(line)

    if len(code_lines):
      if self._has_iter_array:
        code_lines.append('_runner._i += 1')
      code_scope_old = {k: v for k, v in code_scope.items()}
      # function
      code, func = tools.code_lines_to_func(
        lines=code_lines,
        func_name=_input_func_name,
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

  def __call__(self, duration, start_t=None):
    """The running function.

    Parameters
    ----------
    duration : float, int, tuple, list
      The running duration.
    start_t : float, optional

    Returns
    -------
    running_time : float
      The total running time.
    """
    # time step
    if start_t is None:
      if self._start_t is None:
        start_t = 0.
      else:
        start_t = self._start_t
    end_t = start_t + duration
    # times
    times = math.arange(start_t, end_t, self.dt)
    time_steps = math.ones_like(times) * self.dt
    # running
    t0 = time.time()
    _, hists = self._step([times.value, time_steps.value])
    running_time = time.time() - t0
    self._post(times, hists)
    self._start_t = end_t
    if self.numpy_mon_after_run:
      self.mon.numpy()
    return running_time


class NumpyRunner(BaseRunner):
  """The runner provides convenient interface for debugging.
  It is also able to report the running progress.

  Parameters
  ----------
  target : DynamicalSystem
    The target model to run.
  monitors : None, list of str, tuple of str, Monitor
    Variables to monitor.
  inputs : list, tuple
    The input settings.
  report : float
    The percent of progress to report.
  """

  def __init__(self, target, inputs=(), monitors=None, report=0.1,
               dt=None, numpy_mon_after_run=True):
    super(NumpyRunner, self).__init__(target=target, inputs=inputs, monitors=monitors,
                                      dt=dt, numpy_mon_after_run=numpy_mon_after_run)
    self.report = report

    # Build the update function
    self._update_step = lambda _t, _dt: [_step(_t=_t, _dt=_dt)
                                         for _step in self.target.steps.values()]

  def build_monitors(self, show_code=False):
    """Get the monitor function according to the user's setting.

    This method will consider the following things:

    1. the monitor variable
    2. the monitor index
    3. the monitor interval

    """
    from brainpy.math.numpy import Variable
    monitors = utils.check_and_format_monitors(host=self.target, mon=self.mon)

    host = self.target
    code_lines = []
    code_scope = dict(sys=sys, self_mon=self.mon)
    for key, target, variable, idx, interval in monitors:
      code_scope[host.name] = host
      code_scope[target.name] = target

      # get data
      data = target
      for k in variable.split('.'): data = getattr(data, k)

      # get the data key in the host
      if not isinstance(data, Variable):
        raise RunningError(f'"{key}" in {target} is not a dynamically changed Variable, '
                           f'its value will not change, we think there is no need to '
                           f'monitor its trajectory.')
      if np.ndim(data) == 1:
        key_in_host = f'{target.name}.{variable}.value'
      else:
        key_in_host = f'{target.name}.{variable}.value.flatten()'

      # format the monitor index
      if idx is None:
        right = key_in_host + '.copy()'
      else:
        idx = np.asarray(idx)
        right = f'{key_in_host}[_{key.replace(".", "_")}_idx].copy()'
        code_scope[f'_{key.replace(".", "_")}_idx'] = idx

      # format the monitor lines according to the time interval
      if interval is None:
        code_lines.append(f'self_mon.item_contents["{key}"].append({right})')
      else:
        code_scope[f'_{key.replace(".", "_")}_next_time'] = interval
        code_lines.extend([f'global _{key.replace(".", "_")}_next_time',
                           f'if _t >= _{key.replace(".", "_")}_next_time:',
                           f'  self_mon.item_contents["{key}"].append({right})',
                           f'  self_mon.item_contents["{key}.t"].append(_t)',
                           f'  _{key.replace(".", "_")}_next_time += {interval}'])

    if len(code_lines):
      # function
      code_scope_old = {k: v for k, v in code_scope.items()}
      code, func = tools.code_lines_to_func(lines=code_lines,
                                            func_name=_mon_func_name,
                                            func_args=['_t', '_dt'],
                                            scope=code_scope)
      if show_code:
        print(code)
        print()
        pprint(code_scope_old)
        print()
    else:
      func = lambda _t, _dt: None
    return func

  def build_inputs(self, inputs, show_code=False):
    from brainpy.math.numpy import Variable
    code_scope = {'sys': sys}
    code_lines = []
    for target, key, value, type_, op in inputs:
      variable = getattr(target, key)

      # code scope
      code_scope[target.name] = target

      # code line left
      if isinstance(variable, Variable):
        left = f'{target.name}.{key}'
      else:
        raise RunningError(f'"{key}" in {target} is not a dynamically changed Variable, '
                           f'its value will not change, we think there is no need to '
                           f'give its input.')

      # code line right
      if type_ == 'iter':
        code_scope[f'{target.name}_input_data_of_{key}'] = iter(value)
        right = f'next({target.name}_input_data_of_{key})'
      elif type_ == 'func':
        code_scope[f'{target.name}_input_data_of_{key}'] = value
        right = f'{target.name}_input_data_of_{key}(_t, _dt)'
      else:
        code_scope[f'{target.name}_input_data_of_{key}'] = value
        right = f'{target.name}_input_data_of_{key}'

      # code line
      if op == '=':
        line = f'{left}[:] = {right}'
      else:
        line = f'{left} {op}= {right}'

      code_lines.append(line)

    if len(code_lines):
      code_scope_old = {k: v for k, v in code_scope.items()}
      # function
      code, func = tools.code_lines_to_func(
        lines=code_lines,
        func_name=_input_func_name,
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

  def _step(self, t_and_dt):
    self._input_step(_t=t_and_dt[0], _dt=t_and_dt[1])
    self._update_step(_t=t_and_dt[0], _dt=t_and_dt[1])
    self._monitor_step(_t=t_and_dt[0], _dt=t_and_dt[1])

  def __call__(self, duration, start_t=None):
    """The running function.

    Parameters
    ----------
    duration : float, int, tuple, list
      The running duration.
    start_t : float, optional
      The start simulation time.

    Returns
    -------
    running_time : float
      The total running time.
    """
    # time step
    if start_t is None:
      if self._start_t is None:
        start_t = 0.
      else:
        start_t = self._start_t
    end_t = start_t + duration

    # times
    times = np.arange(start_t, end_t, self.dt)

    # build inputs
    for key in self.mon.item_contents.keys():
      self.mon.item_contents[key] = []  # reshape the monitor items

    # simulations
    run_length = len(times)
    if self.report > 0.:
      t0 = time.time()
      self._step((times[0], self.dt))
      compile_time = time.time() - t0
      print('Compilation used {:.4f} s.'.format(compile_time))

      print("Start running ...")
      report_gap = int(run_length * self.report)
      t0 = time.time()
      for run_idx in range(1, run_length):
        self._step((times[run_idx], self.dt))
        if (run_idx + 1) % report_gap == 0:
          percent = (run_idx + 1) / run_length * 100
          print('Run {:.1f}% used {:.3f} s.'.format(percent, time.time() - t0))
      running_time = time.time() - t0
      print('Simulation is done in {:.3f} s.'.format(running_time))
      print()

    else:
      t0 = time.time()
      for run_idx in range(run_length):
        self._step((times[run_idx], self.dt))
      running_time = time.time() - t0

    # monitor post steps
    self.mon.ts = times
    for key, val in self.mon.item_contents.items():
      self.mon.item_contents[key] = np.asarray(val)
    self._start_t = end_t
    if self.numpy_mon_after_run:
      self.mon.numpy()
    return running_time
