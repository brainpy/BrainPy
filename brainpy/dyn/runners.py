# -*- coding: utf-8 -*-

import time

import jax.numpy as jnp
import numpy as np
import tqdm.auto
from jax.experimental.host_callback import id_tap

from brainpy.base.base import TensorCollector
from brainpy import math as bm
from brainpy.dyn import utils
from brainpy.dyn.base import DynamicalSystem
from brainpy.errors import RunningError
from brainpy.running.runner import Runner

__all__ = [
  'DSRunner', 'ReportRunner', 'StructRunner',
]


class DSRunner(Runner):
  """The runner for dynamical systems.

  Parameters
  ----------
  target : DynamicalSystem
    The target model to run.
  inputs : list, tuple
    The inputs for the target DynamicalSystem. It should be the format
    of `[(target, value, [type, operation])]`, where `target` is the
    input target, `value` is the input value, `type` is the input type
    (such as "fix", "iter", "func"), `operation` is the operation for inputs
    (such as "+", "-", "*", "/", "=").

    - ``target``: should be a string. Can be specified by the *absolute access* or *relative access*.
    - ``value``: should be a scalar, vector, matrix, iterable function or objects.
    - ``type``: should be a string. "fix" means the input `value` is a constant. "iter" means the
      input `value` can be changed over time. "func" mean the input is obtained through the functional call.
    - ``operation``: should be a string, support `+`, `-`, `*`, `/`, `=`.
    - Also, if you want to specify multiple inputs, just give multiple ``(target, value, [type, operation])``,
      for example ``[(target1, value1), (target2, value2)]``.
  """

  def __init__(self, target: DynamicalSystem, inputs=(), dt=None, **kwargs):
    super(DSRunner, self).__init__(target=target, **kwargs)

    # parameters
    dt = bm.get_dt() if dt is None else dt
    if not isinstance(dt, (int, float)):
      raise RunningError(f'"dt" must be scalar, but got {dt}')
    self.dt = dt
    if not isinstance(target, DynamicalSystem):
      raise RunningError(f'"target" must be an instance of {DynamicalSystem.__name__}, '
                         f'but we got {type(target)}: {target}')

    # Build the monitor function
    self._monitor_step = self.build_monitors()

    # whether has iterable input data
    self._has_iter_array = False  # default do not have iterable input array
    self._i = bm.Variable(bm.asarray([0]))

    # Build input function
    inputs = utils.check_and_format_inputs(host=target, inputs=inputs)
    self._input_step = self.build_inputs(inputs)

    # start simulation time
    self._start_t = None

    # JAX does not support iterator in fori_loop, scan, etc.
    #   https://github.com/google/jax/issues/3567
    # We use Variable i to index the current input data.
    if self._has_iter_array:  # must behind of "self.build_input()"
      self.dyn_vars.update({'_i': self._i})
    else:
      self._i = None

    # run function
    self._run_func = self.build_run_function()

  def build_inputs(self, inputs):
    fix_inputs = {'=': [], '+': [], '-': [], '*': [], '/': []}
    next_inputs = {'=': [], '+': [], '-': [], '*': [], '/': []}
    func_inputs = {'=': [], '+': [], '-': [], '*': [], '/': []}
    array_inputs = {'=': [], '+': [], '-': [], '*': [], '/': []}

    for target, key, value, type_, op in inputs:
      # variable
      variable = getattr(target, key)
      if not isinstance(variable, bm.Variable):
        raise RunningError(f'"{key}" in {target} is not a dynamically changed Variable, '
                           f'its value will not change, we think there is no need to '
                           f'give its input.')

      # input data
      if type_ == 'iter':
        if isinstance(value, (bm.ndarray, np.ndarray, jnp.ndarray)):
          array_inputs[op].append([variable, bm.asarray(value)])
          self._has_iter_array = True
        else:
          next_inputs[op].append([variable, iter(value)])
      elif type_ == 'func':
        func_inputs[op].append([variable, value])
      else:
        fix_inputs[op].append([variable, value])

    def _f_ops(ops, var, data):
      if ops == '=':
        var[:] = data
      elif ops == '+':
        var += data
      elif ops == '-':
        var -= data
      elif ops == '*':
        var *= data
      elif ops == '/':
        var /= data
      else:
        raise ValueError

    def func(_t, _dt):
      for ops, values in fix_inputs.items():
        for var, data in values:
          _f_ops(ops, var, data)
      for ops, values in array_inputs.items():
        for var, data in values:
          _f_ops(ops, var, data[self._i[0]])
      for ops, values in func_inputs.items():
        for var, data in values:
          _f_ops(ops, var, data(_t, _dt))
      for ops, values in next_inputs.items():
        for var, data in values:
          _f_ops(ops, var, next(data))
      if self._has_iter_array:
        self._i += 1

    return func

  def build_monitors(self):
    monitors = utils.check_and_format_monitors(host=self.target, mon=self.mon)

    return_with_idx = dict()
    return_without_idx = dict()
    for key, target, variable, idx, interval in monitors:
      if interval is not None:
        raise ValueError(f'Running with "{self.__class__.__name__}" does '
                         f'not support "interval" in the monitor.')
      data = target
      for k in variable.split('.'):
        data = getattr(data, k)
      if not isinstance(data, bm.Variable):
        raise RunningError(f'"{key}" in {target} is not a dynamically changed Variable, '
                           f'its value will not change, we think there is no need to '
                           f'monitor its trajectory.')
      if idx is None:
        return_without_idx[key] = data
      else:
        return_with_idx[key] = (data, bm.asarray(idx))

    def func(_t, _dt):
      res = {k: (v.flatten() if bm.ndim(v) > 1 else v.value)
             for k, v in return_without_idx.items()}
      res.update({k: (v.flatten()[idx] if bm.ndim(v) > 1 else v[idx])
                  for k, (v, idx) in return_with_idx.items()})
      return res

    return func

  def _run_one_step(self, _t):
    self._input_step(_t, self.dt)
    self.target.update(_t, self.dt)
    if self.progress_bar:
      id_tap(lambda *args: self._pbar.update(), ())
    return self._monitor_step(_t, self.dt)

  def build_run_function(self):
    if self.jit:
      dyn_vars = TensorCollector()
      dyn_vars.update(self.dyn_vars)
      dyn_vars.update(self.target.vars().unique())
      f_run = bm.make_loop(self._run_one_step,
                           dyn_vars=dyn_vars,
                           has_return=True)
    else:
      def f_run(all_t):
        for i in range(all_t.shape[0]):
          mon = self._run_one_step(all_t[i])
          for k, v in mon.items():
            self.mon.item_contents[k].append(v)
        return None, {}
    return f_run

  def run(self, duration, start_t=None):
    return self.__call__(duration, start_t=start_t)

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
        start_t = float(self._start_t)
    end_t = float(start_t + duration)
    # times
    times = np.arange(start_t, end_t, self.dt)
    # build monitor
    for key in self.mon.item_contents.keys():
      self.mon.item_contents[key] = []  # reshape the monitor items
    # running
    if self.progress_bar:
      self._pbar = tqdm.auto.tqdm(total=times.size)
      self._pbar.set_description(f"Running a duration of {round(float(duration), 3)} ({times.size} steps)",
                                 refresh=True)
    t0 = time.time()
    _, hists = self._run_func(times)
    running_time = time.time() - t0
    if self.progress_bar:
      self._pbar.close()
    # post-running
    if self.jit:
      self.mon.ts = times + self.dt
      for key in self.mon.item_names:
        self.mon.item_contents[key] = bm.asarray(hists[key])
    else:
      self.mon.ts = times + self.dt
      for key in self.mon.item_names:
        self.mon.item_contents[key] = bm.asarray(self.mon.item_contents[key])
    self._start_t = end_t
    if self.numpy_mon_after_run:
      self.mon.numpy()
    return running_time


class StructRunner(DSRunner):
  """The runner with the structural for-loop.

  .. deprecated:: 2.0.3
     Prefer the use of :py:class:`brainpy.dyn.DSRunner` for dynamical system running.
     This runner is deprecated since 2.0.3.
  """

  def __init__(self, target, *args, **kwargs):
    super(StructRunner, self).__init__(target, *args, **kwargs)


class ReportRunner(DSRunner):
  """The runner provides convenient interface for debugging.
  It is also able to report the running progress.

  .. deprecated:: 2.0.3
     Prefer the use of :py:class:`brainpy.dyn.DSRunner` for dynamical system running.
     This runner is deprecated since 2.0.3.

  Parameters
  ----------
  target : DynamicalSystem
    The target model to run.
  monitors : None, list of str, tuple of str, Monitor
    Variables to monitor.
  inputs : list, tuple
    The input settings.
  """

  def __init__(self, target, inputs=(), jit=False, dt=None, **kwargs):
    super(ReportRunner, self).__init__(target=target, inputs=inputs, dt=dt, jit=False, **kwargs)

    # Build the update function
    if jit:
      dyn_vars = TensorCollector()
      dyn_vars.update(self.dyn_vars)
      dyn_vars.update(self.target.vars().unique())
      self._update_step = bm.jit(self.target.update, dyn_vars=dyn_vars)
    else:
      self._update_step = self.target.update

  def _run_one_step(self, _t):
    self._input_step(_t, self.dt)
    self._update_step(_t, self.dt)
    if self.progress_bar:
      self._pbar.update()
    return self._monitor_step(_t, self.dt)

  def build_run_function(self):
    def f_run(all_t):
      for i in range(all_t.shape[0]):
        mon = self._run_one_step(all_t[i])
        for k, v in mon.items():
          self.mon.item_contents[k].append(v)
      return None, {}

    return f_run
