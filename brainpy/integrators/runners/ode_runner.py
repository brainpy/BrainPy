# -*- coding: utf-8 -*-

import time

import numpy as np
import tqdm.auto
from jax.experimental.host_callback import id_tap

from brainpy import math
from brainpy.base.collector import Collector, TensorCollector
from brainpy.errors import RunningError, MonitorError
from brainpy.integrators.base import Integrator
from brainpy.running.runner import Runner

__all__ = [
  'IntegratorRunner',
]


class IntegratorRunner(Runner):
  def __init__(self, target, monitors=None, inits=None,
               args=None, dyn_args=None, dyn_vars=None,
               jit=True, dt=None, numpy_mon_after_run=True, progress_bar=True):
    super(IntegratorRunner, self).__init__(target=target,
                                           monitors=monitors,
                                           jit=jit,
                                           progress_bar=progress_bar,
                                           dyn_vars=dyn_vars,
                                           numpy_mon_after_run=numpy_mon_after_run)

    # parameters
    dt = math.get_dt() if dt is None else dt
    if not isinstance(dt, (int, float)):
      raise RunningError(f'"dt" must be scalar, but got {dt}')
    self.dt = dt

    # target
    if not isinstance(self.target, Integrator):
      raise RunningError(f'"target" must be an instance of {Integrator.__name__}, '
                         f'but we got {type(target)}: {target}')

    # arguments of the integral function
    self._static_args = Collector()
    if args is not None:
      assert isinstance(args, dict), f'"args" must be a dict, but we get {type(args)}: {args}'
      self._static_args.update(args)
    self._dyn_args = Collector()
    if dyn_args is not None:
      assert isinstance(dyn_args, dict), f'"dyn_args" must be a dict, but we get {type(dyn_args)}: {dyn_args}'
      sizes = np.unique([len(v) for v in dyn_args.values()])
      num_size = len(sizes)
      if num_size != 1:
        raise RunningError(f'All values in "dyn_args" should have the same length. But we got '
                           f'{num_size}: {sizes}')
      self._dyn_args.update(dyn_args)

    # monitors
    for k in self.mon.item_names:
      if k not in self.target.variables:
        raise MonitorError(f'Variable "{k}" to monitor is not defined in the integrator {self.target}.')

    # start simulation time
    self._start_t = None

    # dynamical changed variables
    self.dyn_vars.update(self.target.vars().unique())

    # Variables
    if inits is not None:
      if isinstance(inits, (list, tuple)):
        assert len(self.target.variables) == len(inits)
        inits = {k: inits[i] for i, k in enumerate(self.target.variables)}
      assert isinstance(inits, dict)
      sizes = np.unique([np.size(v) for v in list(inits.values())])
      max_size = np.max(sizes)
    else:
      max_size = 1
      inits = dict()
    self.variables = TensorCollector({v: math.Variable(math.zeros(max_size)) for v in self.target.variables})
    for k in inits.keys():
      self.variables[k][:] = inits[k]
    self.dyn_vars.update(self.variables)
    if len(self._dyn_args) > 0:
      self.idx = math.Variable(math.zeros(1, dtype=math.int_))
      self.dyn_vars['_idx'] = self.idx

    # build the update step
    if jit:
      _loop_func = math.make_loop(
        self._step,
        dyn_vars=self.dyn_vars,
        out_vars={k: self.variables[k] for k in self.mon.item_names}
        # out_vars={k: eval(k, self.variables) for k in self.mon.item_names}
      )
    else:
      def _loop_func(t_and_dt):
        out_vars = {k: [] for k in self.mon.item_names}
        times, dts = t_and_dt
        for i in range(len(times)):
          _t = times[i]
          _dt = dts[i]
          self._step([_t, _dt])
          for k in self.mon.item_names:
            out_vars[k].append(math.as_device_array(self.variables[k]))
        out_vars = {k: math.asarray(out_vars[k]) for k in self.mon.item_names}
        return out_vars
    self.step_func = _loop_func

  def _post(self, times, returns):  # monitor
    self.mon.ts = times
    for key in self.mon.item_names:
      self.mon.item_contents[key] = math.asarray(returns[key])

  def _step(self, t_and_dt):
    # arguments
    kwargs = dict()
    kwargs.update(self.variables)
    kwargs.update({'t': t_and_dt[0], 'dt': t_and_dt[1]})
    kwargs.update(self._static_args)
    if len(self._dyn_args) > 0:
      kwargs.update({k: v[self.idx] for k, v in self._dyn_args.items()})
      self.idx += 1
    # call integrator function
    update_values = self.target(**kwargs)
    for i, v in enumerate(self.target.variables):
      self.variables[v].update(update_values[i])
    if self.progress_bar:
      id_tap(lambda *args: self._pbar.update(), ())

  def run(self, duration, start_t=None):
    self.__call__(duration, start_t)

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
    if len(self._dyn_args) > 0:
      self.dyn_vars['_idx'][0] = 0

    # time step
    if start_t is None:
      if self._start_t is None:
        start_t = 0.
      else:
        start_t = float(self._start_t)
    end_t = float(start_t + duration)
    # times
    times = math.arange(start_t, end_t, self.dt)
    time_steps = math.ones_like(times) * self.dt
    # running
    if self.progress_bar:
      self._pbar = tqdm.auto.tqdm(total=times.size)
      self._pbar.set_description(f"Running a duration of {round(float(duration), 3)} ({times.size} steps)",
                                 refresh=True)
    t0 = time.time()
    hists = self.step_func([times.value, time_steps.value])
    running_time = time.time() - t0
    if self.progress_bar:
      self._pbar.close()
    # post-running
    self._post(times, hists)
    self._start_t = end_t
    if self.numpy_mon_after_run:
      self.mon.numpy()
    return running_time
