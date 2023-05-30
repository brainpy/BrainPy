# -*- coding: utf-8 -*-

"""
This module provides numerical methods for integrating Caputo fractional derivative equations.

"""

from typing import Union, Dict, Sequence, Callable

import jax
from scipy.special import gamma, rgamma

import brainpy.math as bm
from brainpy import check
from brainpy._src.integrators.constants import DT
from brainpy._src.integrators.utils import check_inits, format_args
from brainpy.errors import UnsupportedError
from brainpy.types import ArrayType
from .base import FDEIntegrator
from .generic import register_fde_integrator

__all__ = [
  'CaputoEuler',
  'CaputoL1Schema',
]


class CaputoEuler(FDEIntegrator):
  r"""One-step Euler method for Caputo fractional differential equations.

  Given a fractional initial value problem,

  .. math::

     D_{*}^{\alpha} y(t)=f(t, y(t)), \quad y^{(k)}(0)=y_{0}^{(k)}, \quad k=0,1, \ldots,\lceil\alpha\rceil-1

  where the :math:`y_0^{(k)}` ay be arbitrary real numbers and where :math:`\alpha>0`.
  :math:`D_{*}^{\alpha}` denotes the differential operator in the sense of Caputo, defined
  by

  .. math::

     D_{*}^{\alpha} z(t)=J^{n-\alpha} D^{n} z(t)

  where :math:`n:=\lceil\alpha\rceil` is the smallest integer :math:`\geqslant \alpha`,
  Here :math:`D^n` is the usual differential operator of (integer) order :math:`n`,
  and for :math:`\mu > 0`, :math:`J^{\mu}` is the Riemann–Liouville integral operator
  of order :math:`\mu`, defined by

  .. math::

     J^{\mu} z(t)=\frac{1}{\Gamma(\mu)} \int_{0}^{t}(t-u)^{\mu-1} z(u) \mathrm{d} u

  The one-step Euler method for fractional differential equation is defined as

  .. math::

     y_{k+1} = y_0 + \frac{1}{\Gamma(\alpha)} \sum_{j=0}^{k} b_{j, k+1} f\left(t_{j}, y_{j}\right).

  where

  .. math::

     b_{j, k+1}=\frac{h^{\alpha}}{\alpha}\left((k+1-j)^{\alpha}-(k-j)^{\alpha}\right).


  Examples
  --------

  >>> import brainpy as bp
  >>>
  >>> a, b, c = 10, 28, 8 / 3
  >>> def lorenz(x, y, z, t):
  >>>   dx = a * (y - x)
  >>>   dy = x * (b - z) - y
  >>>   dz = x * y - c * z
  >>>   return dx, dy, dz
  >>>
  >>> duration = 30.
  >>> dt = 0.005
  >>> inits = [1., 0., 1.]
  >>> f = bp.fde.CaputoEuler(lorenz, alpha=0.97, num_memory=int(duration / dt), inits=inits)
  >>> runner = bp.integrators.IntegratorRunner(f, monitors=list('xyz'), dt=dt, inits=inits)
  >>> runner.run(duration)
  >>>
  >>> import matplotlib.pyplot as plt
  >>> plt.plot(runner.mon.x.flatten(), runner.mon.z.flatten())
  >>> plt.show()


  Parameters
  ----------
  f : callable
    The derivative function.
  alpha: int, float, jnp.ndarray, bm.ndarray, sequence
    The fractional-order of the derivative function. Should be in the range of ``(0., 1.)``.
  num_memory: int
    The total time step of the simulation.
  inits: sequence
    A sequence of the initial values for variables.
  dt: float, int
    The numerical precision.
  name: str
    The integrator name.

  References
  ----------
  .. [1] Li, Changpin, and Fanhai Zeng. "The finite difference methods for fractional
         ordinary differential equations." Numerical Functional Analysis and
         Optimization 34.2 (2013): 149-179.
  .. [2] Diethelm, Kai, Neville J. Ford, and Alan D. Freed. "Detailed error analysis
         for a fractional Adams method." Numerical algorithms 36.1 (2004): 31-52.
  """

  def __init__(
      self,
      f: Callable,
      alpha: Union[float, Sequence[float], ArrayType],
      num_memory: int,
      inits: Union[ArrayType, Sequence[ArrayType], Dict[str, ArrayType]],
      dt: float = None,
      name: str = None,
      state_delays: Dict[str, Union[bm.LengthDelay, bm.TimeDelay]] = None,
  ):
    super(CaputoEuler, self).__init__(f=f,
                                      alpha=alpha,
                                      dt=dt,
                                      name=name,
                                      num_memory=num_memory,
                                      state_delays=state_delays)

    # fractional order
    if not bm.all(bm.logical_and(self.alpha < 1, self.alpha > 0)):
      raise UnsupportedError(f'Only support the fractional order in (0, 1), '
                             f'but we got {self.alpha}.')

    # initial values
    self.inits = check_inits(inits, self.variables)

    # coefficients
    rgamma_alpha = bm.asarray(rgamma(bm.as_numpy(self.alpha)))
    ranges = bm.asarray([bm.arange(num_memory + 1) for _ in self.variables]).T
    coef = rgamma_alpha * bm.diff(bm.power(ranges, self.alpha), axis=0)
    self.coef = bm.flip(coef, axis=0)

    # variable states
    self.f_states = {v: bm.Variable(bm.zeros((num_memory,) + self.inits[v].shape))
                     for v in self.variables}
    self.register_implicit_vars(self.f_states)
    self.idx = bm.Variable(bm.asarray([1]))

    self.set_integral(self._integral_func)

  def _check_step(self, args):
    dt, t = args
    raise ValueError(f'The maximum number of step is {self.num_memory}, '
                     f'however, the current time {t} require a time '
                     f'step number {t / dt}.')

  def _integral_func(self, *args, **kwargs):
    # format arguments
    all_args = format_args(args, kwargs, self.arg_names)
    t = all_args['t']
    dt = all_args.pop(DT, self.dt)
    if check.is_checking():
      check.jit_error(self.num_memory * dt < t, self._check_step, (dt, t))

    # derivative values
    devs = self.f(**all_args)
    if len(self.variables) == 1:
      if not isinstance(devs, (bm.ndarray, jax.Array)):
        raise ValueError('Derivative values must be a tensor when there '
                         'is only one variable in the equation.')
      devs = {self.variables[0]: devs}
    else:
      if not isinstance(devs, (tuple, list)):
        raise ValueError('Derivative values must be a list/tuple of tensors '
                         'when there are multiple variables in the equation.')
      devs = {var: devs[i] for i, var in enumerate(self.variables)}

    # function states
    for key in self.variables:
      self.f_states[key][self.idx[0]] = devs[key]

    # integral results
    integrals = []
    idx = ((self.num_memory - 1 - self.idx) + bm.arange(self.num_memory)) % self.num_memory
    for i, key in enumerate(self.variables):
      integral = self.inits[key] + self.coef[idx, i] @ self.f_states[key]
      integrals.append(integral * (dt ** self.alpha[i] / self.alpha[i]))
    self.idx.value = (self.idx + 1) % self.num_memory

    # return integrals
    if len(self.variables) == 1:
      return integrals[0]
    else:
      return integrals


register_fde_integrator(name='euler', integrator=CaputoEuler)


class CaputoL1Schema(FDEIntegrator):
  r"""The L1 scheme method for the numerical approximation of the Caputo
  fractional-order derivative equations [3]_.

  For the fractional order :math:`0<\alpha<1`, let the fractional derivative of variable
  :math:`x(t)` be

  .. math::

     \frac{d^{\alpha} x}{d t^{\alpha}}=F(x, t)

  The Caputo definition of the fractional derivative for variable :math:`x` is

  .. math::

     \frac{d^{\alpha} x}{d t^{\alpha}}=\frac{1}{\Gamma(1-\alpha)} \int_{0}^{t} \frac{x^{\prime}(u)}{(t-u)^{\alpha}} d u

  where :math:`\Gamma` is the Gamma function.

  The fractional-order derivative is capable of integrating the activity of the
  function over all past activities weighted by a function that follows a power-law.
  Using one of the numerical methods, the L1 scheme method [3]_, the numerical
  approximation of the fractional-order derivative of :math:`x` is

  .. math::

     \frac{d^{\alpha} \chi}{d t^{\alpha}} \approx \frac{(d t)^{-\alpha}}{\Gamma(2-\alpha)}\left[\sum_{k=0}^{N-1}\left[x\left(t_{k+1}\right)-
     \mathrm{x}\left(t_{k}\right)\right]\left[(N-k)^{1-\alpha}-(N-1-k)^{1-\alpha}\right]\right]

  Therefore, the numerical solution of original system is given by

  .. math::

     x\left(t_{N}\right) \approx d t^{\alpha} \Gamma(2-\alpha) F(x, t)+x\left(t_{N-1}\right)-
     \left[\sum_{k=0}^{N-2}\left[x\left(t_{k+1}\right)-x\left(t_{k}\right)\right]\left[(N-k)^{1-\alpha}-(N-1-k)^{1-\alpha}\right]\right]

  Hence, the solution of the fractional-order derivative can be described as the
  difference between the *Markov term* and the *memory trace*. The *Markov term*
  weighted by the gamma function is

  .. math::

     \text { Markov term }=d t^{\alpha} \Gamma(2-\alpha) F(x, t)+x\left(t_{N-1}\right)

  The memory trace (:math:`x`-memory trace since it is related to variable :math:`x`) is

  .. math::

     \text { Memory trace }=\sum_{k=0}^{N-2}\left[x\left(t_{k+1}\right)-x\left(t_{k}\right)\right]\left[(N-k)^{1-\alpha}-(N-(k+1))^{1-\alpha}\right]

  The memory trace integrates all the past activity and captures the long-term
  history of the system. For :math:`\alpha=1`, the memory trace is 0 for any
  time :math:`t`. When the fractional order :math:`\alpha` is decreased from 1,
  the memory trace non-linearly increases from 0, and its dynamics strongly
  depends on time. Thus, the fractional order dynamics strongly deviates
  from the first order dynamics.


  Examples
  --------

  >>> import brainpy as bp
  >>>
  >>> a, b, c = 10, 28, 8 / 3
  >>> def lorenz(x, y, z, t):
  >>>   dx = a * (y - x)
  >>>   dy = x * (b - z) - y
  >>>   dz = x * y - c * z
  >>>   return dx, dy, dz
  >>>
  >>> duration = 30.
  >>> dt = 0.005
  >>> inits = [1., 0., 1.]
  >>> f = bp.fde.CaputoL1Schema(lorenz, alpha=0.99, num_memory=int(duration / dt), inits=inits)
  >>> runner = bp.IntegratorRunner(f, monitors=list('xz'), dt=dt, inits=inits)
  >>> runner.run(duration)
  >>>
  >>> import matplotlib.pyplot as plt
  >>> plt.plot(runner.mon.x.flatten(), runner.mon.z.flatten())
  >>> plt.show()


  Parameters
  ----------
  f : callable
    The derivative function.
  alpha: int, float, jnp.ndarray, bm.ndarray, sequence
    The fractional-order of the derivative function. Should be in the range of ``(0., 1.]``.
  num_memory: int
    The total time step of the simulation.
  inits: sequence
    A sequence of the initial values for variables.
  dt: float, int
    The numerical precision.
  name: str
    The integrator name.

  References
  ----------
  .. [3] Oldham, K., & Spanier, J. (1974). The fractional calculus theory
         and applications of differentiation and integration to arbitrary
         order. Elsevier.
  """

  def __init__(
      self,
      f: Callable,
      alpha: Union[float, Sequence[float], ArrayType],
      num_memory: int,
      inits: Union[ArrayType, Sequence[ArrayType], Dict[str, ArrayType]],
      state_delays: Dict[str, Union[bm.LengthDelay, bm.TimeDelay]] = None,
      dt: float = None,
      name: str = None,
  ):
    super(CaputoL1Schema, self).__init__(f=f,
                                         alpha=alpha,
                                         dt=dt,
                                         name=name,
                                         num_memory=num_memory,
                                         state_delays=state_delays)

    # fractional order
    if not bm.all(bm.logical_and(self.alpha <= 1, self.alpha > 0)):
      raise UnsupportedError(f'Only support the fractional order in (0, 1), '
                             f'but we got {self.alpha}.')
    self.gamma_alpha = bm.asarray(gamma(bm.as_numpy(2 - self.alpha)))

    # initial values
    inits = check_inits(inits, self.variables)
    self.inits = bm.VarDict({v: bm.Variable(inits[v]) for v in self.variables})

    # coefficients
    ranges = bm.asarray([bm.arange(1, num_memory + 2) for _ in self.variables]).T
    coef = bm.diff(bm.power(ranges, 1 - self.alpha), axis=0)
    self.coef = bm.flip(coef, axis=0)

    # used to save the difference of two adjacent states
    self.diff_states = bm.VarDict({v + "_diff": bm.Variable(bm.zeros((num_memory,) + self.inits[v].shape,
                                                            dtype=self.inits[v].dtype))
                                   for v in self.variables})
    self.idx = bm.Variable(bm.asarray([self.num_memory - 1]))

    # integral function
    self.set_integral(self._integral_func)

  def reset(self, inits):
    """Reset function."""
    self.idx.value = bm.asarray([self.num_memory - 1])
    inits = check_inits(inits, self.variables)
    for key, value in inits.items():
      self.inits[key] = value
    for key, val in inits.items():
      self.diff_states[key + "_diff"] = bm.zeros((self.num_memory,) + val.shape, dtype=val.dtype)

  def hists(self, var=None, numpy=True):
    """Get the recorded history values."""
    if var is None:
      hists_ = {k: bm.vstack([self.inits[k], self.diff_states[k + '_diff']])
                for k in self.variables}
      hists_ = {k: bm.cumsum(v, axis=0) for k, v in hists_.items()}
      if numpy:
        hists_ = {k: v.numpy() for k, v in hists_}
      return hists_
    else:
      assert var in self.variables, (f'"{var}" is not defined in equation '
                                     f'variables: {self.variables}')
      hists_ = bm.vstack([self.inits[var], self.diff_states[var + '_diff']])
      hists_ = bm.cumsum(hists_, axis=0)
      if numpy:
        hists_ = hists_.numpy()
      return hists_

  def _check_step(self, args):
    dt, t = args
    raise ValueError(f'The maximum number of step is {self.num_memory}, '
                     f'however, the current time {t} require a time '
                     f'step number {t / dt}.')

  def _integral_func(self, *args, **kwargs):
    # format arguments
    all_args = format_args(args, kwargs, self.arg_names)
    t = all_args['t']
    dt = all_args.pop(DT, self.dt)
    if check.is_checking():
      check.jit_error(self.num_memory * dt < t, self._check_step, (dt, t))

    # derivative values
    devs = self.f(**all_args)
    if len(self.variables) == 1:
      if not isinstance(devs, (bm.Array, jax.Array)):
        raise ValueError('Derivative values must be a tensor when there '
                         'is only one variable in the equation.')
      devs = {self.variables[0]: devs}
    else:
      if not isinstance(devs, (tuple, list)):
        raise ValueError('Derivative values must be a list/tuple of tensors '
                         'when there are multiple variables in the equation.')
      devs = {var: devs[i] for i, var in enumerate(self.variables)}

    # integral results
    integrals = []
    idx = ((self.num_memory - 1 - self.idx) + bm.arange(self.num_memory)) % self.num_memory
    for i, key in enumerate(self.variables):
      self.diff_states[key + '_diff'][self.idx[0]] = all_args[key] - self.inits[key]
      self.inits[key].value = all_args[key]
      markov_term = dt ** self.alpha[i] * self.gamma_alpha[i] * devs[key] + all_args[key]
      memory_trace = self.coef[idx, i] @ self.diff_states[key + '_diff']
      integral = markov_term - memory_trace
      integrals.append(integral)
    self.idx.value = (self.idx + 1) % self.num_memory

    # return integrals
    if len(self.variables) == 1:
      return integrals[0]
    else:
      return integrals


register_fde_integrator(name='l1', integrator=CaputoL1Schema)
