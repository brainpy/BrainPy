# -*- coding: utf-8 -*-

import time
from typing import Union, Dict, Sequence, Callable

import jax.numpy as jnp
import numpy as np
import tqdm.auto
from jax.experimental.host_callback import id_tap

from brainpy import math as bm
from brainpy.base.collector import Collector, TensorCollector
from brainpy.errors import RunningError, MonitorError
from brainpy.integrators.base import Integrator
from brainpy.running.runner import Runner


__all__ = [
  'IntegratorRunner',
]


class IntegratorRunner(Runner):
  """Structural runner for numerical integrators in brainpy.

  Examples
  --------

  Example to run an ODE integrator,

  .. plot::
    :include-source: True

    >>> import brainpy as bp
    >>> import brainpy.math as bm
    >>> a=0.7; b=0.8; tau=12.5
    >>> dV = lambda V, t, w, I: V - V * V * V / 3 - w + I
    >>> dw = lambda w, t, V, a, b: (V + a - b * w) / tau
    >>> integral = bp.odeint(bp.JointEq([dV, dw]), method='exp_auto')
    >>>
    >>> runner = bp.integrators.IntegratorRunner(
    >>>          integral,  # the simulation target
    >>>          monitors=['V', 'w'],  # the variables to monitor
    >>>          inits={'V': bm.random.rand(10),
    >>>                 'w': bm.random.normal(size=10)},  # the initial values
    >>>          args={'a': 1., 'b': 1.},  # update arguments
    >>>          dyn_args={'I': bp.inputs.ramp_input(0, 4, 200)},  # each time each current input
    >>> )
    >>> runner.run(100.)
    >>> bp.visualize.line_plot(runner.mon.ts, runner.mon.V, plot_ids=[0, 1, 4], show=True)

  Example to run an SDE intragetor,

  .. plot::
    :include-source: True

    >>> import brainpy as bp
    >>> import brainpy.math as bm
    >>> # stochastic Lorenz system
    >>> sigma=10; beta=8 / 3; rho=28
    >>> g = lambda x, y, z, t, p: (p * x, p * y, p * z)
    >>> f = lambda x, y, z, t, p: [sigma * (y - x), x * (rho - z) - y, x * y - beta * z]
    >>> lorenz = bp.sdeint(f, g, method='milstein')
    >>>
    >>> runner = bp.integrators.IntegratorRunner(
    >>>   lorenz,
    >>>   monitors=['x', 'y', 'z'],
    >>>   inits=[1., 1., 1.], # initialize all variable to 1.
    >>>   args={'p': 0.1},
    >>>   dt=0.01
    >>> )
    >>> runner.run(100.)
    >>>
    >>> import matplotlib.pyplot as plt
    >>> fig = plt.figure()
    >>> ax = fig.gca(projection='3d')
    >>> plt.plot(runner.mon.x.squeeze(), runner.mon.y.squeeze(), runner.mon.z.squeeze())
    >>> ax.set_xlabel('x')
    >>> ax.set_xlabel('y')
    >>> ax.set_xlabel('z')
    >>> plt.show()

  """

  def __init__(
      self,
      target: Integrator,

      # IntegratorRunner specific arguments
      inits: Union[Sequence, Dict] = None,
      args: Dict = None,
      dyn_args: Dict[str, Union[bm.ndarray, jnp.ndarray]] = None,
      dt: Union[float, int] = None,

      # regular/common arguments
      fun_monitors: Dict[str, Callable] = None,
      monitors: Sequence[str] = None,
      dyn_vars: Dict[str, bm.Variable] = None,
      jit: Union[bool, Dict[str, bool]] = True,
      numpy_mon_after_run: bool = True,
      progress_bar: bool = True
  ):
    """Initialization of structural runner for integrators.

    Parameters
    ----------
    target: Integrator
      The target to run.
    monitors: sequence of str
      The variables to monitor.
    fun_monitors: dict
      The monitors with callable functions.
    inits: sequence, dict
      The initial value of variables. With this parameter,
      you can easily control the number of variables to simulate.
      For example, if one of the variable has the shape of 10,
      then all variables will be an instance of :py:class:`brainpy.math.Variable`
      with the shape of :math:`(10,)`.
    args: dict
      The equation arguments to update.
      Note that if one of the arguments are heterogeneous (i.e., a tensor),
      it means we should run multiple trials. However, you can set the number
      of the elements in the variables so that each pair of variables can
      corresponds to one set of arguments.
    dyn_args: dict
      The dynamically changed arguments. This means this argument can control
      the argument dynamically changed. For example, if you want to inject a
      time varied currents into the HH neuron model, you can pack the currents
      into this ``dyn_args`` argument.
    dt: float, int
    dyn_vars: dict
    jit: bool
    progress_bar: bool
    numpy_mon_after_run: bool
    """

    if not isinstance(target, Integrator):
      raise TypeError(f'Target must be instance of {Integrator.__name__}, '
                      f'but we got {type(target)}')

    # get maximum size and initial variables
    if inits is not None:
      if isinstance(inits, (list, tuple, bm.JaxArray, jnp.ndarray)):
        assert len(target.variables) == len(inits)
        inits = {k: inits[i] for i, k in enumerate(target.variables)}
      assert isinstance(inits, dict), f'"inits" must be a dict, but we got {type(inits)}'
      sizes = np.unique([np.size(v) for v in list(inits.values())])
      max_size = np.max(sizes)
    else:
      max_size = 1
      inits = dict()

    # initialize variables
    self.variables = TensorCollector({v: bm.Variable(bm.zeros(max_size))
                                      for v in target.variables})
    for k in inits.keys():
      self.variables[k][:] = inits[k]

    # format string monitors
    monitors = self._format_seq_monitors(monitors)
    monitors = {k: (self.variables[k], i) for k, i in monitors}

    # initialize super class
    super(IntegratorRunner, self).__init__(target=target,
                                           monitors=monitors,
                                           fun_monitors=fun_monitors,
                                           jit=jit,
                                           progress_bar=progress_bar,
                                           dyn_vars=dyn_vars,
                                           numpy_mon_after_run=numpy_mon_after_run)

    # parameters
    dt = bm.get_dt() if dt is None else dt
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
      assert isinstance(args, dict), (f'"args" must be a dict, but '
                                      f'we got {type(args)}: {args}')
      self._static_args.update(args)
    self._dyn_args = Collector()
    if dyn_args is not None:
      assert isinstance(dyn_args, dict), (f'"dyn_args" must be a dict, but we get '
                                          f'{type(dyn_args)}: {dyn_args}')
      sizes = np.unique([len(v) for v in dyn_args.values()])
      num_size = len(sizes)
      if num_size != 1:
        raise RunningError(f'All values in "dyn_args" should have the same length. '
                           f'But we got {num_size}: {sizes}')
      self._dyn_args.update(dyn_args)

    # monitors
    for k in self.mon.var_names:
      if k not in self.target.variables and k not in self.fun_monitors:
        raise MonitorError(f'Variable "{k}" to monitor is not defined '
                           f'in the integrator {self.target}.')

    # start simulation time
    self._start_t = None

    # dynamical changed variables
    self.dyn_vars.update(self.target.vars().unique())

    # Variables
    self.dyn_vars.update(self.variables)
    if len(self._dyn_args) > 0:
      self.idx = bm.Variable(bm.zeros(1, dtype=jnp.int_))
      self.dyn_vars['_idx'] = self.idx

    # build the update step
    if self.jit['predict']:
      def _loop_func(times):
        return bm.for_loop(self._step, self.dyn_vars, times)
    else:
      def _loop_func(times):
        returns = {k: [] for k in self.fun_monitors.keys()}
        returns.update({k: [] for k in self.monitors.keys()})
        for i in range(len(times)):
          _t = times[i]
          _dt = self.dt
          # function monitor
          for k, v in self.fun_monitors.items():
            returns[k].append(v(_t, _dt))
          # step call
          self._step(_t)
          # variable monitors
          for k in self.monitors.keys():
            returns[k].append(bm.as_device_array(self.variables[k]))
        returns = {k: bm.asarray(returns[k]) for k in returns.keys()}
        return returns
    self.step_func = _loop_func

  def _step(self, t):
    # arguments
    kwargs = dict()
    kwargs.update(self.variables)
    kwargs.update({'t': t, 'dt': self.dt})
    kwargs.update(self._static_args)
    if len(self._dyn_args) > 0:
      kwargs.update({k: v[self.idx.value] for k, v in self._dyn_args.items()})
      self.idx += 1

    # call integrator function
    update_values = self.target(**kwargs)
    if len(self.target.variables) == 1:
      self.variables[self.target.variables[0]].update(update_values)
    else:
      for i, v in enumerate(self.target.variables):
        self.variables[v].update(update_values[i])

    # progress bar
    if self.progress_bar:
      id_tap(lambda *args: self._pbar.update(), ())

    # return of function monitors
    returns = dict()
    for key, func in self.fun_monitors.items():
      returns[key] = func(t, self.dt)
    for k in self.monitors.keys():
      returns[k] = self.variables[k].value
    return returns

  def run(self, duration, start_t=None, eval_time=False):
    """The running function.

    Parameters
    ----------
    duration : float, int, tuple, list
      The running duration.
    start_t : float, optional
      The start time to simulate.
    eval_time: bool
      Evaluate the running time or not?
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
    times = bm.arange(start_t, end_t, self.dt).value

    # running
    if self.progress_bar:
      self._pbar = tqdm.auto.tqdm(total=times.size)
      self._pbar.set_description(f"Running a duration of {round(float(duration), 3)} ({times.size} steps)",
                                 refresh=True)
    if eval_time:
      t0 = time.time()
    hists = self.step_func(times)
    if eval_time:
      running_time = time.time() - t0
    if self.progress_bar:
      self._pbar.close()

    # post-running
    times += self.dt
    if self.numpy_mon_after_run:
      times = np.asarray(times)
      for key in list(hists.keys()):
        hists[key] = np.asarray(hists[key])
    self.mon.ts = times
    for key in hists.keys():
      self.mon[key] = hists[key]
    self._start_t = end_t
    if eval_time:
      return running_time
