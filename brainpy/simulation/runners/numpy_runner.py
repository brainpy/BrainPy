# -*- coding: utf-8 -*-

import logging
import sys
import time
from pprint import pprint

import numpy as np

from brainpy import tools
from brainpy.building.brainobjects import DynamicalSystem
from brainpy.errors import RunningError
from brainpy.simulation import utils
from .ds_runner import *

logger = logging.getLogger('brainpy.simulation.runner')

__all__ = [
  'NumpyRunner',
]

_mon_func_name = 'monitor_step'
_input_func_name = 'input_step'



class NumpyRunner(BaseRunner):
  """The runner provided interface for model simulation with pure NumPy, along
  with the model acceleration with Numba.

  .. deprecated:: 2.0.3
     This API has been deprecated. Please run dynamical systems using JAX backend.

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

