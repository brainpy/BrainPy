# -*- coding: utf-8 -*-

r"""This module provides exponential integrators for ODEs.

Exponential integrators are a large class of methods from numerical analysis is based on
the exact integration of the linear part of the initial value problem. Because the linear
part is integrated exactly, this can help to mitigate the stiffness of a differential
equation.

We consider initial value problems of the form,

.. math:: u'(t)=f(u(t)),\qquad u(t_{0})=u_{0},

which can be decomposed of

.. math:: u'(t)=Lu(t)+N(u(t)),\qquad u(t_{0})=u_{0},

where :math:`L={\frac {\partial f}{\partial u}}` (the Jacobian of f) is composed of
linear terms, and :math:`N=f(u)-Lu` is composed of the non-linear terms.

This procedure enjoys the advantage, in each step, that
:math:`{\frac {\partial N_{n}}{\partial u}}(u_{n})=0`.
This considerably simplifies the derivation of the order conditions and improves the
stability when integrating the nonlinearity :math:`N(u(t))`.

Exact integration of this problem from time 0 to a later time :math:`t` can be performed
using `matrix exponentials <https://en.wikipedia.org/wiki/Matrix_exponential>`_ to define
an integral equation for the exact solution:

.. math:: u(t)=e^{Lt}u_{0}+\int _{0}^{t}e^{L(t-\tau )}N\left(t+\tau, u\left(\tau \right)\right)\,d\tau .

This representation of the exact solution is also called as *variation-of-constant formula*.
In the case of :math:`N\equiv 0`, this formulation is the exact solution to the linear
differential equation.


**Exponential Rosenbrock methods**

Exponential Rosenbrock methods were shown to be very efficient in solving large systems
of stiff ODEs. Applying the variation-of-constants formula gives the exact solution at
time :math:`t_{n+1}` with the numerical solution :math:`u_n` as

.. math::
    u(t_{n+1})=e^{h_{n}L}u(t_{n})+\int _{0}^{h_{n}}e^{(h_{n}-\tau )L}N(t_n+\tau, u(t_{n}+\tau ))d\tau .
    :label: discrete-variation-of-constants-formula

where :math:`h_n=t_{n+1}-t_n`.

The idea now is to approximate the integral in :eq:`discrete-variation-of-constants-formula`
by some quadrature rule with nodes :math:`c_{i}` and weights :math:`b_{i}(h_{n}L)`
(:math:`1\leq i\leq s`). This yields the following class of *s-stage* explicit exponential
Rosenbrock methods:

.. math::
    \begin{align}
    U_{ni}=&e^{c_{i}h_{n}L}u_n+h_{n}\sum_{j=1}^{i-1}a_{ij}(h_{n}L)N(U_{nj}),  \\
    u_{n+1}=&e^{h_{n}L}u_n+h_{n}\sum_{i=1}^{s}b_{i}(h_{n}L)N(U_{ni})
    \end{align}

where :math:`U_{ni}\approx u(t_{n}+c_{i}h_{n})`.

The coefficients :math:`a_{ij}(z),b_{i}(z)` are usually chosen as linear combinations of
the entire functions :math:`\varphi _{k}(c_{i}z),\varphi _{k}(z)`, respectively, where

.. math::
    \begin{align}
    \varphi _{k}(z)=&\int _{0}^{1}e^{(1-\theta )z}{\frac {\theta ^{k-1}}{(k-1)!}}d\theta ,\quad k\geq 1, \\
    \varphi _{0}(z)=&e^{z},\\
    \varphi _{k+1}(z)=&{\frac {\varphi_{k}(z)-\varphi _{k}(0)}{z}},\ k\geq 0.
    \end{align}

By introducing the difference :math:`D_{ni}=N(U_{ni})-N(u_{n})`, they can be reformulated
in a more efficient way for implementation as

.. math::
    \begin{align}
    U_{ni}=&u_{n}+c_{i}h_{n}\varphi _{1}(c_{i}h_{n}L)f(u_{n})+h_{n}\sum _{j=2}^{i-1}a_{ij}(h_{n}L)D_{nj}, \\
    u_{n+1}=&u_{n}+h_{n}\varphi _{1}(h_{n}L)f(u_{n})+h_{n}\sum _{i=2}^{s}b_{i}(h_{n}L)D_{ni}.
    \end{align}

where :math:`\varphi_{1}(z)=\frac{e^z-1}{z}`.

In order to implement this scheme with adaptive step size, one can consider, for the purpose
of local error estimation, the following embedded methods

.. math:: {\bar {u}}_{n+1}=u_{n}+h_{n}\varphi _{1}(h_{n}L)f(u_{n})+h_{n}\sum _{i=2}^{s}{\bar {b}}_{i}(h_{n}L)D_{ni},

which use the same stages :math:`U_{ni}` but with weights :math:`{\bar {b}}_{i}`.

For convenience, the coefficients of the explicit exponential Rosenbrock methods together
with their embedded methods can be represented by using the so-called reduced Butcher
tableau as follows:

.. math::
    \begin{array}{c|ccccc}
    c_{2} & & & & & \\
    c_{3} & a_{32} & & & & \\
    \vdots & \vdots & & \ddots & & \\
    c_{s} & a_{s 2} & a_{s 3} & \cdots & a_{s, s-1} \\
    \hline & b_{2} & b_{3} & \cdots & b_{s-1} & b_{s} \\
    & \bar{b}_{2} & \bar{b}_{3} & \cdots & \bar{b}_{s-1} & \bar{b}_{s}
    \end{array}

.. [1] https://en.wikipedia.org/wiki/Exponential_integrator
.. [2] Hochbruck, M., & Ostermann, A. (2010). Exponential integrators. Acta Numerica, 19, 209-286.
"""

import logging

from functools import wraps
from brainpy import errors
from brainpy._src import math as bm
from brainpy._src.integrators import constants as C, utils, joint_eq
from brainpy._src.integrators.ode.base import ODEIntegrator
from .generic import register_ode_integrator


__all__ = [
  'ExponentialEuler',
]


class ExponentialEuler(ODEIntegrator):
  """Exponential Euler method using automatic differentiation.

  This method uses `brainpy.math.vector_grad <../../math/generated/brainpy.math.autograd.vector_grad.html>`_
  to automatically infer the linear part of the given function. Therefore, it has minimal constraints
  on your derivative function. Arbitrary complex functions can be numerically integrated with this method.

  Examples
  --------

  Here is an example uses ``ExponentialEuler`` to implement HH neuron model.

  .. plot::
    :include-source: True

    >>> import brainpy as bp
    >>> import brainpy.math as bm
    >>>
    >>> class HH(bp.dyn.NeuDyn):
    >>>   def __init__(self, size, ENa=55., EK=-90., EL=-65, C=1.0, gNa=35., gK=9.,
    >>>                gL=0.1, V_th=20., phi=5.0, name=None):
    >>>     super(HH, self).__init__(size=size, name=name)
    >>>
    >>>     # parameters
    >>>     self.ENa = ENa
    >>>     self.EK = EK
    >>>     self.EL = EL
    >>>     self.C = C
    >>>     self.gNa = gNa
    >>>     self.gK = gK
    >>>     self.gL = gL
    >>>     self.V_th = V_th
    >>>     self.phi = phi
    >>>
    >>>     # variables
    >>>     self.V = bm.Variable(bm.ones(size) * -65.)
    >>>     self.h = bm.Variable(bm.ones(size) * 0.6)
    >>>     self.n = bm.Variable(bm.ones(size) * 0.32)
    >>>     self.spike = bm.Variable(bm.zeros(size, dtype=bool))
    >>>     self.input = bm.Variable(bm.zeros(size))
    >>>
    >>>     # functions
    >>>     self.int_h = bp.ode.ExponentialEuler(self.dh)
    >>>     self.int_n = bp.ode.ExponentialEuler(self.dn)
    >>>     self.int_V = bp.ode.ExponentialEuler(self.dV)
    >>>
    >>>   def dh(self, h, t, V):
    >>>     alpha = 0.07 * bm.exp(-(V + 58) / 20)
    >>>     beta = 1 / (bm.exp(-0.1 * (V + 28)) + 1)
    >>>     dhdt = self.phi * (alpha * (1 - h) - beta * h)
    >>>     return dhdt
    >>>
    >>>   def dn(self, n, t, V):
    >>>     alpha = -0.01 * (V + 34) / (bm.exp(-0.1 * (V + 34)) - 1)
    >>>     beta = 0.125 * bm.exp(-(V + 44) / 80)
    >>>     dndt = self.phi * (alpha * (1 - n) - beta * n)
    >>>     return dndt
    >>>
    >>>   def dV(self, V, t, h, n, Iext):
    >>>     m_alpha = -0.1 * (V + 35) / (bm.exp(-0.1 * (V + 35)) - 1)
    >>>     m_beta = 4 * bm.exp(-(V + 60) / 18)
    >>>     m = m_alpha / (m_alpha + m_beta)
    >>>     INa = self.gNa * m ** 3 * h * (V - self.ENa)
    >>>     IK = self.gK * n ** 4 * (V - self.EK)
    >>>     IL = self.gL * (V - self.EL)
    >>>     dVdt = (- INa - IK - IL + Iext) / self.C
    >>>
    >>>     return dVdt
    >>>
    >>>   def update(self, tdi):
    >>>     h = self.int_h(self.h, tdi.t, self.V, dt=tdi.dt)
    >>>     n = self.int_n(self.n, tdi.t, self.V, dt=tdi.dt)
    >>>     V = self.int_V(self.V, tdi.t,  self.h, self.n, self.input, dt=tdi.dt)
    >>>     self.spike.value = bm.logical_and(self.V < self.V_th, V >= self.V_th)
    >>>     self.V.value = V
    >>>     self.h.value = h
    >>>     self.n.value = n
    >>>     self.input[:] = 0.
    >>>
    >>> run = bp.dyn.DSRunner(HH(1), inputs=('input', 2.), monitors=['V'], dt=0.05)
    >>> run(100)
    >>> bp.visualize.line_plot(run.mon.ts, run.mon.V, legend='V', show=True)

  The above example can also be defined with ``brainpy.JointEq``.

  .. plot::
    :include-source: True

    >>> import brainpy as bp
    >>> import brainpy.math as bm
    >>>
    >>> class HH(bp.dyn.NeuDyn):
    >>>   def __init__(self, size, ENa=55., EK=-90., EL=-65, C=1.0, gNa=35., gK=9.,
    >>>                gL=0.1, V_th=20., phi=5.0, name=None):
    >>>     super(HH, self).__init__(size=size, name=name)
    >>>
    >>>     # parameters
    >>>     self.ENa = ENa
    >>>     self.EK = EK
    >>>     self.EL = EL
    >>>     self.C = C
    >>>     self.gNa = gNa
    >>>     self.gK = gK
    >>>     self.gL = gL
    >>>     self.V_th = V_th
    >>>     self.phi = phi
    >>>
    >>>     # variables
    >>>     self.V = bm.Variable(bm.ones(size) * -65.)
    >>>     self.h = bm.Variable(bm.ones(size) * 0.6)
    >>>     self.n = bm.Variable(bm.ones(size) * 0.32)
    >>>     self.spike = bm.Variable(bm.zeros(size, dtype=bool))
    >>>     self.input = bm.Variable(bm.zeros(size))
    >>>
    >>>     # functions
    >>>     derivative = bp.JointEq([self.dh, self.dn, self.dV])
    >>>     self.integral = bp.ode.ExponentialEuler(derivative)
    >>>
    >>>   def dh(self, h, t, V):
    >>>     alpha = 0.07 * bm.exp(-(V + 58) / 20)
    >>>     beta = 1 / (bm.exp(-0.1 * (V + 28)) + 1)
    >>>     dhdt = self.phi * (alpha * (1 - h) - beta * h)
    >>>     return dhdt
    >>>
    >>>   def dn(self, n, t, V):
    >>>     alpha = -0.01 * (V + 34) / (bm.exp(-0.1 * (V + 34)) - 1)
    >>>     beta = 0.125 * bm.exp(-(V + 44) / 80)
    >>>     dndt = self.phi * (alpha * (1 - n) - beta * n)
    >>>     return dndt
    >>>
    >>>   def dV(self, V, t, h, n, Iext):
    >>>     m_alpha = -0.1 * (V + 35) / (bm.exp(-0.1 * (V + 35)) - 1)
    >>>     m_beta = 4 * bm.exp(-(V + 60) / 18)
    >>>     m = m_alpha / (m_alpha + m_beta)
    >>>     INa = self.gNa * m ** 3 * h * (V - self.ENa)
    >>>     IK = self.gK * n ** 4 * (V - self.EK)
    >>>     IL = self.gL * (V - self.EL)
    >>>     dVdt = (- INa - IK - IL + Iext) / self.C
    >>>
    >>>     return dVdt
    >>>
    >>>   def update(self, tdi):
    >>>     h, n, V = self.integral(self.h, self.n, self.V, tdi.t, self.input, dt=tdi.dt)
    >>>     self.spike.value = bm.logical_and(self.V < self.V_th, V >= self.V_th)
    >>>     self.V.value = V
    >>>     self.h.value = h
    >>>     self.n.value = n
    >>>     self.input[:] = 0.
    >>>
    >>> run = bp.dyn.DSRunner(HH(1), inputs=('input', 2.), monitors=['V'], dt=0.05)
    >>> run(100)
    >>> bp.visualize.line_plot(run.mon.ts, run.mon.V, legend='V', show=True)

  Parameters
  ----------
  f : function, joint_eq.JointEq
    The derivative function.
  var_type : optional, str
    The variable type.
  dt : optional, float
    The default numerical integration step.
  name : optional, str
    The integrator name.
  """

  def __init__(
      self,
      f,
      var_type=None,
      dt=None,
      name=None,
      show_code=False,
      state_delays=None,
      neutral_delays=None
  ):
    super(ExponentialEuler, self).__init__(f=f,
                                           var_type=var_type,
                                           dt=dt,
                                           name=name,
                                           show_code=show_code,
                                           state_delays=state_delays,
                                           neutral_delays=neutral_delays)

    if var_type == C.SYSTEM_VAR:
      raise NotImplementedError(f'{self.__class__.__name__} does not support {C.SYSTEM_VAR}, '
                                f'because the auto-differentiation ')

    # build the integrator
    self.code_lines = []
    self.code_scope = {}
    self.integral = self.build()

  def build(self):
    parses = self._build_integrator(self.f)
    all_vps = self.variables + self.parameters

    @wraps(self.f)
    def integral_func(*args, **kwargs):
      # format arguments
      params_in = bm.Collector()
      for i, arg in enumerate(args):
        params_in[all_vps[i]] = arg
      params_in.update(kwargs)
      if C.DT not in params_in:
        params_in[C.DT] = self.dt

      # call integrals
      results = []
      for i, parse in enumerate(parses):
        f_integral, vars_, pars_ = parse
        vps = vars_ + pars_ + [C.DT]
        r = f_integral(params_in[vps[0]], **{arg: params_in[arg] for arg in vps[1:] if arg in params_in})
        results.append(r)
      return results if len(self.variables) > 1 else results[0]

    return integral_func

  def _build_integrator(self, eq):
    if isinstance(eq, joint_eq.JointEq):
      results = []
      for sub_eq in eq.eqs:
        results.extend(self._build_integrator(sub_eq))
      return results
    else:
      vars, pars, _ = utils.get_args(eq)

      # checking
      if len(vars) != 1:
        raise errors.DiffEqError(C.multi_vars_msg.format(cls=self.__class__.__name__,
                                                         vars=str(vars),
                                                         eq=str(eq)))

      # gradient function
      value_and_grad = bm.vector_grad(eq, argnums=0, return_value=True)

      # integration function
      def integral(*args, **kwargs):
        assert len(args) > 0
        dt = kwargs.pop(C.DT, self.dt)
        linear, derivative = value_and_grad(*args, **kwargs)
        phi = bm.where(linear == 0.,
                       bm.ones_like(linear),
                       (bm.exp(dt * linear) - 1) / (dt * linear))
        return args[0] + dt * phi * derivative

      return [(integral, vars, pars), ]


register_ode_integrator('exponential_euler', ExponentialEuler)
register_ode_integrator('exp_euler', ExponentialEuler)
register_ode_integrator('exp_euler_auto', ExponentialEuler)
register_ode_integrator('exp_auto', ExponentialEuler)
