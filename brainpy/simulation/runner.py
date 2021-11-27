# -*- coding: utf-8 -*-

import time
from pprint import pprint

from brainpy import math, tools
from brainpy.errors import RunningError
from brainpy.simulation import utils
from brainpy.simulation.brainobjects.base import DynamicalSystem

__all__ = [
  'Runner',
  'ReportRunner',
  'StructRunner',
]


class Runner(object):
  def __call__(self, *args, **kwargs):
    pass


class ReportRunner(Runner):
  def __init__(self, model, report=0., dyn_vars=None, jit=False, show_code=False):
    assert isinstance(model, DynamicalSystem)
    self.model = model
    self.report = report
    self.jit = jit

    if dyn_vars is None:
      dyn_vars = self.model.vars().unique()
    if isinstance(dyn_vars, (list, tuple)):
      dyn_vars = {f'_v{i}': v for i, v in enumerate(dyn_vars)}
    assert isinstance(dyn_vars, dict)
    self.dyn_vars = dyn_vars

    # input step function
    self._input_step = None

    # Build the monitors:
    #   All the monitors are wrapped in a single function.
    self._monitor_step = self._build_monitor_func(show_code=show_code)

    # Build the update function
    self._update_step = lambda _t, _dt: [_step(_t=_t, _dt=_dt)
                                         for _step in self.model.steps.values()]
    if jit:
      self._update_step = math.jit(self._update_step, dyn_vars=self.dyn_vars)

  def _build_monitor_func(self, show_code=False, func_name='monitor_step'):
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
    monitors = utils.check_and_format_monitors(host=self.model)

    code_lines = []
    code_scope = dict()
    for node, key, target, variable, idx, interval in monitors:
      code_scope[node.name] = node
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
        if hasattr(idx, 'value'): idx = idx.value
        right = f'{key_in_host}[{node.name}_mon_{key.replace(".", "_")}_idx]'
        code_scope[f'{node.name}_mon_{key.replace(".", "_")}_idx'] = idx

      # format the monitor lines according to the time interval
      if interval is None:
        code_lines.append(f'{node.name}.mon.item_contents["{key}"].append({right})')
      else:
        code_scope[f'{node.name}_mon_{key.replace(".", "_")}_next_time'] = interval
        code_lines.extend([f'global {node.name}_mon_{key.replace(".", "_")}_next_time',
                           f'if _t >= {node.name}_mon_{key.replace(".", "_")}_next_time:',
                           f'  {node.name}.mon.item_contents["{key}"].append({right})',
                           f'  {node.name}.mon.item_contents["{key}.t"].append(_t)',
                           f'  {node.name}_mon_{key.replace(".", "_")}_next_time += {interval}'])

    if len(code_lines):
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
    return func

  def build_inputs(self, inputs=(), show_code=False):
    """Build input function."""
    # Build the inputs:
    #   All the inputs are wrapped into a single function.
    inputs = utils.check_and_format_inputs(host=self.model, inputs=inputs)
    self._input_step = utils.build_input_func(inputs, show_code=show_code)

  def _step(self, t_and_dt):
    self._input_step(_t=t_and_dt[0], _dt=t_and_dt[1])
    self._update_step(_t=t_and_dt[0], _dt=t_and_dt[1])
    self._monitor_step(_t=t_and_dt[0], _dt=t_and_dt[1])

  def __call__(self, duration, dt=None, inputs=()):
    """The running function.

    Parameters
    ----------
    inputs : list, tuple
      The inputs for this instance of DynamicalSystem. It should the format
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

    duration : float, int, tuple, list
      The running duration.

    report : float
      The percent of progress to report. [0, 1]. If zero, the model
      will not output report progress.

    dt : float, optional
      The numerical integration step size.

    Returns
    -------
    running_time : float
      The total running time.
    """

    # time step
    if dt is None: dt = math.get_dt()
    assert isinstance(dt, (int, float))

    # times
    start, end = utils.check_duration(duration)
    times = math.arange(start, end, dt)

    # build inputs
    self.build_inputs(inputs=inputs)
    for node in self.model.nodes().values():
      if hasattr(node, 'mon'):
        for key in node.mon.item_contents.keys():
          node.mon.item_contents[key] = []  # reshape the monitor items

    # simulations
    run_length = len(times)
    if self.report > 0.:
      t0 = time.time()
      self._step((times[0], dt))
      compile_time = time.time() - t0
      print('Compilation used {:.4f} s.'.format(compile_time))

      print("Start running ...")
      report_gap = int(run_length * self.report)
      t0 = time.time()
      for run_idx in range(1, run_length):
        self._step((times[run_idx], dt))
        if (run_idx + 1) % report_gap == 0:
          percent = (run_idx + 1) / run_length * 100
          print('Run {:.1f}% used {:.3f} s.'.format(percent, time.time() - t0))
      running_time = time.time() - t0
      print('Simulation is done in {:.3f} s.'.format(running_time))
      print()

    else:
      t0 = time.time()
      for run_idx in range(run_length):
        self._step((times[run_idx], dt))
      running_time = time.time() - t0

    # monitor post steps
    for node in self.model.nodes().values():
      if hasattr(node, 'mon'):
        if node.mon.num_item > 0:
          node.mon.ts = times
          for key, val in node.mon.item_contents.items():
            node.mon.item_contents[key] = math.asarray(val)

    return running_time


class StructRunner(Runner):
  def __init__(self, model, inputs=(), dyn_vars=None, jit=False, show_code=False):
    assert isinstance(model, DynamicalSystem)
    self.model = model
    self.jit = jit

    if dyn_vars is None:
      dyn_vars = self.model.vars().unique()
    if isinstance(dyn_vars, (list, tuple)):
      dyn_vars = {f'_v{i}': v for i, v in enumerate(dyn_vars)}
    assert isinstance(dyn_vars, dict)
    self.dyn_vars = dyn_vars

    # input step function
    inputs = utils.check_and_format_inputs(host=self.model, inputs=inputs)
    self._input_step = utils.build_input_func(inputs, show_code=show_code)

    # Build the monitors:
    #   All the monitors are wrapped in a single function.
    self._monitor_step, self._assigns = self._build_monitor_func(show_code=show_code)

    # build the update step
    self._step = math.make_loop(self._step, dyn_vars=dyn_vars, has_return=True)
    if jit: self._step = math.jit(self._step, dyn_vars=dyn_vars)

  def _step(self, t_and_dt):  # the step function
    self._input_step(_t=t_and_dt[0], _dt=t_and_dt[1])
    for step in self.model.steps.values():
      step(_t=t_and_dt[0], _dt=t_and_dt[1])
    return self._monitor_step(_t=t_and_dt[0], _dt=t_and_dt[1])

  def _post(self, times, returns):  # monitor
    nodes = self.model.nodes()
    for i, (n, k) in enumerate(self._assigns):
      nodes[n].mon.item_contents[k] = math.asarray(returns[i])
      nodes[n].mon.ts = times

  def _build_monitor_func(self, show_code=False, func_name='monitor_step'):
    monitors = utils.check_and_format_monitors(host=self.model)
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
        if hasattr(idx, 'value'): idx = idx.value
        right = f'{key_in_host}[{node.name}_mon_{key.replace(".", "_")}_idx]'
        code_scope[f'{node.name}_mon_{key.replace(".", "_")}_idx'] = idx

      # format the monitor lines according to the time interval
      returns.append(right)
      assigns.append([node.name, key])
      if interval is not None:
        raise ValueError(f'Running with "{self.__class__.__name__}" method does not '
                         f'support "interval" in the monitor.')

    if len(code_lines) or len(returns):
      code_lines.append(f'return {", ".join(returns) + ", "}')
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

  def __call__(self, duration, dt=None):
    # time step
    if dt is None: dt = math.get_dt()
    assert isinstance(dt, (int, float))
    # times
    start, end = utils.check_duration(duration)
    times = math.arange(start, end, dt)
    time_steps = math.ones_like(times) * dt
    # running
    t0 = time.time()
    _, hists = self._step([times.value, time_steps.value])
    running_time = time.time() - t0
    self._post(times, hists)
    return running_time
