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

from brainpy import math
from brainpy.integrators import constants
from brainpy.integrators.ode.wrapper import exp_euler_wrapper

__all__ = [
  'exponential_euler',
  'exponential_midpoint',
]


def exponential_euler(f=None, show_code=None, dt=None, var_type=None):
  r"""The exponential Euler method for ODEs.

  The simplest exponential Rosenbrock method is the exponential
  Rosenbrock–Euler scheme, which has order 2.

  For an ODE equation of the form

  .. math::

      u^{\prime}=f(u), \quad u(0)=u_{0}

  its schema is given by

  .. math::

      u_{n+1}= u_{n}+h \varphi(hL) f (u_{n})

  where :math:`L=f^{\prime}(u_{n})` and :math:`\varphi(z)=\frac{e^{z}-1}{z}`.

  For a linear ODE system: :math:`u^{\prime} = Ay + B`,
  the above equation is equal to :math:`u_{n+1}= u_{n}e^{hA}-B/A(1-e^{hA})`,
  which is the exact solution for this ODE system.

  Parameters
  ----------
  f : function
    The derivative function.
  dt : optional, float
    The numerical precision.
  var_type : optional, str
    The variable type.
  show_code : bool
    Whether show the code.

  Returns
  -------
  func : function
      The one-step numerical integrator function.
  """

  dt = math.get_dt() if dt is None else dt
  show_code = False if show_code is None else show_code
  var_type = constants.SCALAR_VAR if var_type is None else var_type

  if f is None:
    return lambda f: exp_euler_wrapper(f,
                                       show_code=show_code,
                                       dt=dt,
                                       var_type=var_type)
  else:
    return exp_euler_wrapper(f,
                             show_code=show_code,
                             dt=dt,
                             var_type=var_type)


def exponential_midpoint(f=None, show_code=None, dt=None, var_type=None):
  r"""The exponential midpoint method for ODEs.

  Borgers and Nectow [1]_ [2]_ considered an explicit second-order method, called
  the exponential midpoint method, which is given by:

  .. math::
      \begin{align}
      u_{n+1/2} = &e^{1/2hL(u_n)}u_n + \frac{e^{1/2hL}-1}{1/2hL(u_n)} \frac{1}{2}hN(U_n) \\
      u_{n+1} = & e^{hL(u_{n+1/2})}u_n + \frac{e^{hL(u_{n+1/2})}-1}{hL(u_{n+1/2})}hN(u_{n+1/2})
      \end{align}

  The second line is similar to exponential Euler, except :math:`L` and :math:`N` are
  evaluated at the approximate midpoint :math:`u_{n+1/2}`, obtained by first taking
  a half-step of exponential Euler. Note that each nonlinear function is evaluated
  twice, first at :math:`u_{n}` and again at :math:`u_{n+1/2}`, so this method is
  twice as expensive per step as an Euler-type method.

  Parameters
  ----------
  f : function
    The derivative function.
  dt : optional, float
    The numerical precision.
  var_type : optional, str
    The variable type.
  show_code : bool
    Whether show the code.

  Returns
  -------
  func : function
      The one-step numerical integrator function.

  References
  ----------
  .. [1] Borgers, C., & Nectow, A. R. (2013). Exponential Time Differencing for
         Hodgkin--Huxley-like ODEs. SIAM Journal on Scientific Computing, 35(3),
         B623-B643.
  .. [2] Chen, Zhengdao, Baranidharan Raman, and Ari Stern. "Structure-Preserving
         Numerical Integrators for Hodgkin--Huxley-Type Systems." SIAM Journal on
         Scientific Computing 42, no. 1 (2020): B273-B298.
  """
  raise NotImplementedError
