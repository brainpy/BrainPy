# -*- coding: utf-8 -*-

"""
This module provides numerical solvers for Grünwald–Letnikov derivative FDEs.
"""

from typing import Dict, Union, Callable, Any

import jax.numpy as jnp

import brainpy.math as bm
from brainpy.errors import UnsupportedError
from brainpy.integrators.constants import DT
from brainpy.integrators.utils import check_inits, format_args
from .base import FDEIntegrator
from .generic import register_fde_integrator

__all__ = [
  'GLShortMemory'
]


class GLShortMemory(FDEIntegrator):
  r"""Efficient Computation of the Short-Memory Principle in Grünwald-Letnikov Method [1]_.

  According to the explicit numerical approximation of Grünwald-Letnikov, the
  fractional-order derivative :math:`q` for a discrete function :math:`f(t_K)`
  can be described as follows:

  .. math::

     {{}_{k-\frac{L_{m}}{h}}D_{t_{k}}^{q}}f(t_{k})\approx h^{-q}
     \sum\limits_{j=0}^{k}C_{j}^{q}f(t_{k-j})

  where :math:`L_{m}` is the memory lenght, :math:`h` is the integration step size,
  and :math:`C_{j}^{q}` are the binomial coefficients which are calculated recursively with

  .. math::

     C_{0}^{q}=1,\ C_{j}^{q}=\left(1- \frac{1+q}{j}\right)C_{j-1}^{q},\ j=1,2, \ldots k.

  Then, the numerical solution for a fractional-order differential equation (FODE) expressed
  in the form

  .. math::

     D_{t_{k}}^{q}x(t_{k})=f(x(t_{k}))

  can be obtained by

  .. math::

     x(t_{k})=f(x(t_{k-1}))h^{q}- \sum\limits_{j=1}^{k}C_{j}^{q}x(t_{k-j}).

  for :math:`0 < q < 1`. The above expression requires infinity memory length
  for numerical solution since the summation term depends on the discritized
  time :math:`t_k`. This implies relatively high simulation times.

  To reduce the computational time, the upper bound of summation needs to be modified by
  :math:`k=v`, where

  .. math::

     v=\begin{cases} k, & k\leq M,\\ L_{m}, & k > M. \end{cases}

  This is known as the short-memory principle, where :math:`M`
  is the memory window with a width defined by :math:`M=\frac{L_{m}}{h}`.
  As was reported in [2]_, the accuracy increases by increaing the width of memory window.

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
  >>> integral = bp.fde.GLShortMemory(lorenz,
  >>>                                 alpha=0.96,
  >>>                                 num_step=500,
  >>>                                 inits=[1., 0., 1.])
  >>> runner = bp.integrators.IntegratorRunner(integral,
  >>>                                          monitors=list('xyz'),
  >>>                                          inits=[1., 0., 1.],
  >>>                                          dt=0.005)
  >>> runner.run(100.)
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
    The length of the short memory.

    .. versionchanged:: 2.1.11

  inits: sequence
    A sequence of the initial values for variables.
  dt: float, int
    The numerical precision.
  name: str
    The integrator name.

  References
  ----------
  .. [1] Clemente-López, D., et al. "Efficient computation of the
         Grünwald-Letnikov method for arm-based implementations of
         fractional-order chaotic systems." 2019 8th International
         Conference on Modern Circuits and Systems Technologies (MOCAST). IEEE, 2019.
  .. [2] M. F. Tolba, A. M. AbdelAty, N. S. Soliman, L. A. Said, A. H.
         Madian, A. T. Azar, et al., "FPGA implementation of two fractional
         order chaotic systems", International Journal of Electronics and
         Communications, vol. 78, pp. 162-172, 2017.
  """

  def __init__(
      self,
      f: Callable,
      alpha: Any,
      inits: Any,
      num_memory: int,
      dt: float = None,
      name: str = None,
      state_delays: Dict[str, Union[bm.LengthDelay, bm.TimeDelay]] = None,
  ):
    super(GLShortMemory, self).__init__(f=f,
                                        alpha=alpha,
                                        dt=dt,
                                        name=name,
                                        num_memory=num_memory,
                                        state_delays=state_delays)

    # fractional order
    if not bm.all(bm.logical_and(self.alpha <= 1, self.alpha > 0)):
      raise UnsupportedError(f'Only support the fractional order in (0, 1), '
                             f'but we got {self.alpha}.')

    # initial values
    inits = check_inits(inits, self.variables)

    # delays
    self.delays = {}
    for key, val in inits.items():
      delay = bm.Variable(bm.zeros((self.num_memory,) + val.shape, dtype=val.dtype))
      delay[0] = val
      self.delays[key+'_delay'] = delay
    self._idx = bm.Variable(bm.asarray([1]))
    self.register_implicit_vars(self.delays)

    # binomial coefficients
    bc = (1 - (1 + self.alpha.reshape((-1, 1))) / jnp.arange(1, num_memory + 1))
    bc = bm.cumprod(bm.vstack([bm.ones_like(self.alpha), bc.T]), axis=0)
    self._binomial_coef = bm.flip(bc[1:], axis=0)

    # integral function
    self.set_integral(self._integral_func)

  def reset(self, inits):
    """Reset function of the delay variables."""
    self._idx.value = bm.asarray([1])
    inits = check_inits(inits, self.variables)
    for key, val in inits.items():
      delay = bm.zeros((self.num_memory,) + val.shape, dtype=val.dtype)
      delay[0] = val
      self.delays[key].value = delay

  @property
  def binomial_coef(self):
    return bm.as_numpy(bm.flip(self._binomial_coef, axis=0))

  def _integral_func(self, *args, **kwargs):
    # format arguments
    all_args = format_args(args, kwargs, self.arg_names)
    dt = all_args.pop(DT, self.dt)

    # derivative values
    devs = self.f(**all_args)
    if len(self.variables) == 1:
      if not isinstance(devs, (bm.ndarray, jnp.ndarray)):
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
    idx = (self._idx + bm.arange(self.num_memory)) % self.num_memory
    for i, var in enumerate(self.variables):
      delay_var = var + '_delay'
      summation = self._binomial_coef[:, i] @ self.delays[delay_var][idx]
      integral = (dt ** self.alpha[i]) * devs[var] - summation
      self.delays[delay_var][self._idx[0]] = integral
      integrals.append(integral)
    self._idx.value = (self._idx + 1) % self.num_memory

    # return integrals
    if len(self.variables) == 1:
      return integrals[0]
    else:
      return integrals


register_fde_integrator(name='short-memory', integrator=GLShortMemory)
