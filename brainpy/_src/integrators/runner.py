# -*- coding: utf-8 -*-

import time
import warnings
from functools import partial
from typing import Union, Dict, Sequence, Callable

import jax
import jax.numpy as jnp
import numpy as np
import tqdm.auto
from jax.experimental.host_callback import id_tap
from jax.tree_util import tree_flatten

from brainpy import math as bm
from brainpy._src.math.object_transform.base import Collector
from brainpy._src.running.runner import Runner
from brainpy.errors import RunningError
from .base import Integrator

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
    >>> runner = bp.IntegratorRunner(
    >>>          integral,  # the simulation target
    >>>          monitors=['V', 'w'],  # the variables to monitor
    >>>          inits={'V': bm.random.rand(10),
    >>>                 'w': bm.random.normal(size=10)},  # the initial values
    >>> )
    >>> runner.run(100.,
    >>>            args={'a': 1., 'b': 1.},  # update arguments
    >>>            dyn_args={'I': bp.inputs.ramp_input(0, 4, 100)},  # each time each current input
    >>> )
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
    >>> lorenz = bp.sdeint(f, g, method='milstein2')
    >>>
    >>> runner = bp.IntegratorRunner(
    >>>   lorenz,
    >>>   monitors=['x', 'y', 'z'],
    >>>   inits=[1., 1., 1.], # initialize all variable to 1.
    >>>   dt=0.01
    >>> )
    >>> runner.run(100., args={'p': 0.1},)
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

      # regular/common arguments
      dt: Union[float, int] = None,
      monitors: Sequence[str] = None,
      dyn_vars: Dict[str, bm.Variable] = None,
      jit: Union[bool, Dict[str, bool]] = True,
      numpy_mon_after_run: bool = True,
      progress_bar: bool = True,

      # deprecated
      args: Dict = None,
      dyn_args: Dict[str, Union[bm.ndarray, jnp.ndarray]] = None,
      fun_monitors: Dict[str, Callable] = None,
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
      .. deprecated:: 2.3.1
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
      correspond to one set of arguments.

      .. deprecated:: 2.3.1
         Will be removed after version 2.4.0.

    dyn_args: dict
      The dynamically changed arguments. This means this argument can control
      the argument dynamically changed. For example, if you want to inject a
      time varied currents into the HH neuron model, you can pack the currents
      into this ``dyn_args`` argument.

      .. deprecated:: 2.3.1
         Will be removed after version 2.4.0.

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
      if isinstance(inits, (list, tuple, bm.Array, jnp.ndarray)):
        assert len(target.variables) == len(inits)
        inits = {k: inits[i] for i, k in enumerate(target.variables)}
      assert isinstance(inits, dict), f'"inits" must be a dict, but we got {type(inits)}'
      sizes = np.unique([np.size(v) for v in list(inits.values())])
      max_size = np.max(sizes)
    else:
      max_size = 1
      inits = dict()

    # initialize variables
    self.variables = {v: bm.Variable(bm.zeros(max_size)) for v in target.variables}
    for k in inits.keys():
      self.variables[k][:] = inits[k]

    # format string monitors
    if isinstance(monitors, (tuple, list)):
      monitors = self._format_seq_monitors(monitors)
      monitors = {k: (self.variables[k], i) for k, i in monitors}
    elif isinstance(monitors, dict):
      monitors = self._format_dict_monitors(monitors)
      monitors = {k: ((self.variables[i], i) if isinstance(i, str) else i) for k, i in monitors.items()}
    else:
      raise ValueError

    # initialize super class
    super(IntegratorRunner, self).__init__(target=target,
                                           monitors=monitors,
                                           fun_monitors=fun_monitors,
                                           jit=jit,
                                           progress_bar=progress_bar,
                                           dyn_vars=dyn_vars,
                                           numpy_mon_after_run=numpy_mon_after_run)

    self.register_implicit_vars(self.variables)

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
    if args is not None:
      warnings.warn('Set "args" in `IntegratorRunner.run()` function, instead of __init__ function. '
                    'Will be removed since 2.4.0',
                    UserWarning)
      assert isinstance(args, dict), (f'"args" must be a dict, but '
                                      f'we got {type(args)}: {args}')
      self._static_args = args
    else:
      self._static_args = dict()
    if dyn_args is not None:
      warnings.warn('Set "dyn_args" in `IntegratorRunner.run()` function, instead of __init__ function. '
                    'Will be removed since 2.4.0',
                    UserWarning)
      assert isinstance(dyn_args, dict), (f'"dyn_args" must be a dict, but we get '
                                          f'{type(dyn_args)}: {dyn_args}')
      sizes = np.unique([len(v) for v in dyn_args.values()])
      num_size = len(sizes)
      if num_size != 1:
        raise RunningError(f'All values in "dyn_args" should have the same length. '
                           f'But we got {num_size}: {sizes}')
      self._dyn_args = dyn_args
    else:
      self._dyn_args = dict()

    # start simulation time and index
    self.start_t = bm.Variable(bm.zeros(1))
    self.idx = bm.Variable(bm.zeros(1, dtype=bm.int_))

  def _run_fun_integration(self, static_args, dyn_args, times, indices):
    return bm.for_loop(partial(self._step_fun_integrator, static_args),
                       (dyn_args, times, indices),
                       jit=self.jit['predict'])

  def _step_fun_integrator(self, static_args, dyn_args, t, i):
    # arguments
    kwargs = Collector(dt=self.dt, t=t)
    kwargs.update(static_args)
    kwargs.update(dyn_args)
    kwargs.update({k: v.value for k, v in self.variables.items()})

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
    shared = dict(t=t + self.dt, dt=self.dt, i=i)
    returns = dict()
    for k, v in self._monitors.items():
      if callable(v):
        returns[k] = bm.as_jax(v(shared))
      else:
        returns[k] = self.variables[k].value
    return returns

  def run(
      self,
      duration: float,
      start_t: float = None,
      eval_time: bool = False,
      args: Dict = None,
      dyn_args: Dict = None,
  ):
    """The running function.

    Parameters
    ----------
    duration : float, int, tuple, list
      The running duration.
    start_t : float, optional
      The start time to simulate.
    eval_time: bool
      Evaluate the running time or not?
    args: dict
      The equation arguments to update.
      .. versionadded:: 2.3.1

    dyn_args: dict
      The dynamically changed arguments over time. The size of first dimension should be
      equal to the running ``duration``.

      .. versionadded:: 2.3.1

    """
    args = dict() if args is None else args
    dyn_args = dict() if dyn_args is None else dyn_args
    assert isinstance(args, dict), f'"args" must be a dict, but we got {type(args)}: {args}'
    assert isinstance(dyn_args, dict), f'"dyn_args" must be a dict, but we got {type(dyn_args)}: {dyn_args}'
    args.update(self._static_args)
    dyn_args.update(self._dyn_args)

    # time step
    if start_t is None:
      start_t = self.start_t[0]
    end_t = start_t + duration
    # times
    times = bm.arange(start_t, end_t, self.dt).value
    indices = bm.arange(times.size).value + self.idx.value

    _dyn_args, _ = tree_flatten(dyn_args)
    for _d in _dyn_args:
      if jnp.shape(_d)[0] != times.size:
        raise ValueError(f'The shape of `dyn_args` does not match the given duration. '
                         f'{jnp.shape(_d)[0]} != {times.size} (duration={duration}, dt={self.dt}).')
      del _d
    del _dyn_args

    # running
    if self.progress_bar:
      self._pbar = tqdm.auto.tqdm(total=times.size)
      self._pbar.set_description(f"Running a duration of {round(float(duration), 3)} ({times.size} steps)",
                                 refresh=True)
    if eval_time:
      t0 = time.time()
    hists = self._run_fun_integration(args, dyn_args, times, indices)
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
    self.start_t[0] = end_t
    self.idx[0] += times.size
    if eval_time:
      return running_time
