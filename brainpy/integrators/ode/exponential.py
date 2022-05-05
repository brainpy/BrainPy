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

import inspect
import logging

from brainpy import math, errors, tools
from brainpy.base.collector import Collector
from brainpy.integrators import constants as C, utils, joint_eq
from brainpy.integrators.analysis_by_ast import separate_variables
from brainpy.integrators.ode.base import ODEIntegrator
from .generic import register_ode_integrator

try:
  import sympy
  from brainpy.integrators import analysis_by_sympy
except (ModuleNotFoundError, ImportError):
  sympy = analysis_by_sympy = None

logger = logging.getLogger('brainpy.integrators.ode.exponential')

__all__ = [
  'ExponentialEuler',
  'ExpEulerAuto',
]


class ExponentialEuler(ODEIntegrator):
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

  Examples
  --------

  Linear system example: HH model's derivative function.

  >>> def derivative(V, m, h, n, t, Iext, gNa, ENa, gK, EK, gL, EL, C):
  >>>    alpha = 0.1 * (V + 40) / (1 - bm.exp(-(V + 40) / 10))
  >>>    beta = 4.0 * bm.exp(-(V + 65) / 18)
  >>>    dmdt = alpha * (1 - m) - beta * m
  >>>
  >>>    alpha = 0.07 * bm.exp(-(V + 65) / 20.)
  >>>    beta = 1 / (1 + bm.exp(-(V + 35) / 10))
  >>>    dhdt = alpha * (1 - h) - beta * h
  >>>
  >>>    alpha = 0.01 * (V + 55) / (1 - bm.exp(-(V + 55) / 10))
  >>>    beta = 0.125 * bm.exp(-(V + 65) / 80)
  >>>    dndt = alpha * (1 - n) - beta * n
  >>>
  >>>    I_Na = (gNa * m ** 3.0 * h) * (V - ENa)
  >>>    I_K = (gK * n ** 4.0) * (V - EK)
  >>>    I_leak = gL * (V - EL)
  >>>    dVdt = (- I_Na - I_K - I_leak + Iext) / C
  >>>    return dVdt, dmdt, dhdt, dndt
  >>>
  >>> ExponentialEuler(f=derivative, show_code=True)
  def brainpy_itg_of_ode0_drivative(V, m, h, n, t, Iext, gNa, ENa, gK, EK, gL, EL, C, dt=0.01):
    _dfV_dt = EK*gK*n**4.0/C + EL*gL/C + ENa*gNa*h*m**3.0/C + Iext/C - V*gK*n**4.0/C - V*gL/C - V*gNa*h*m**3.0/C
    _V_linear = -gK*n**4.0/C - gL/C - gNa*h*m**3.0/C
    _V_linear_exp = math.exp(_V_linear * dt)
    _V_df_part = _dfV_dt*(_V_linear_exp - 1)/_V_linear
    V_new = V + _V_df_part
    #
    alpha = 0.1 * (V + 40) / (1 - bm.exp(-(V + 40) / 10))
    beta = 4.0 * bm.exp(-(V + 65) / 18)
    _dfm_dt = -alpha*m + 1.0*alpha - beta*m
    _m_linear = -alpha - beta
    _m_linear_exp = math.exp(_m_linear * dt)
    _m_df_part = _dfm_dt*(_m_linear_exp - 1)/_m_linear
    m_new = _m_df_part + m
    #
    alpha = 0.07 * bm.exp(-(V + 65) / 20.0)
    beta = 1 / (1 + bm.exp(-(V + 35) / 10))
    _dfh_dt = -alpha*h + 1.0*alpha - beta*h
    _h_linear = -alpha - beta
    _h_linear_exp = math.exp(_h_linear * dt)
    _h_df_part = _dfh_dt*(_h_linear_exp - 1)/_h_linear
    h_new = _h_df_part + h
    #
    alpha = 0.01 * (V + 55) / (1 - bm.exp(-(V + 55) / 10))
    beta = 0.125 * bm.exp(-(V + 65) / 80)
    _dfn_dt = -alpha*n + 1.0*alpha - beta*n
    _n_linear = -alpha - beta
    _n_linear_exp = math.exp(_n_linear * dt)
    _n_df_part = _dfn_dt*(_n_linear_exp - 1)/_n_linear
    n_new = _n_df_part + n
    #
    return V_new, m_new, h_new, n_new

  Nonlinear system example: Van der Pol oscillator.

  >>> def vdp_derivative(x, y, t, mu):
  >>>    dx = mu * (x - x ** 3 / 3 - y)
  >>>    dy = x / mu
  >>>    return dx, dy
  >>>
  >>> ExponentialEuler(f=vdp_derivative, show_code=True)
  def brainpy_itg_of_ode0_vdp_derivative(x, y, t, mu, dt=0.01):
    _dfx_dt = mu*x - 0.333333333333333*mu*x**3.0 - mu*y
    _x_linear = -0.999999999999999*mu*x**2.0 + mu
    _x_linear_exp = math.exp(_x_linear * dt)
    _x_df_part = _dfx_dt*(_x_linear_exp - 1)/_x_linear
    x_new = _x_df_part + x
    #
    dy = x / mu
    _dfy_dt = dy
    _y_df_part = _dfy_dt * dt
    y_new = _y_df_part + y
    #
    return x_new, y_new

  However, ExponentialEuler method has severve constraints (see below).

  .. note::
    Many constraints are involved when using this ExponentialEuler method, because
    it uses SymPy to make symbolic differentiation for your codes.
    If you want to use get a more flexible method to do exponential euler integration,
    please refer to ``ExpEulerAuto`` method.

    The mechanism of ExponentialEuler method is :
    First, the user's codes are transformed into the SymPy expressions
    by using AST. Then, infer the derivative of the sympy expression by
    using ``sympy.diff()``.

    The constains of using ExponentialEuler method are:

    1. Must use latex-like codes to write equations.

    For your derivative variable, if there are general Python functions are using
    these varaibles, the SymPy parsing will not work.
    For instance, the following codes can not be recognized by SymPy, because the
    unsupported function ``bm.power`` are performing on the derivative variable ``x``.

    >>> import brainpy as bp
    >>> import brainpy.math as bm
    >>>
    >>> def derivative(x, t):
    >>>   dx = bm.power(x, 3) - bm.power(x, 2)
    >>>   return dx
    >>>
    >>> f = bp.ode.ExponentialEuler(derivative, show_code=True)
    def brainpy_itg_of_ode1_dev(x, t, dt=0.1):
      _dfx_dt = -bm.power(x, 2.0) + bm.power(x, 3.0)
      _x_linear = -Derivative(bm.power(x, 2.0), x) + Derivative(bm.power(x, 3.0), x)
      _x_linear_exp = math.exp(_x_linear * dt)
      _x_df_part = _dfx_dt*(_x_linear_exp - 1)/_x_linear
      x_new = _x_df_part + x
      return x_new

    As you see, SymPy cannot parse ``bm.power()``. Therefore, the linear part
    (``_x_linear``) of the system contains a unknown variable ``Derivative``. This will
    cause an error in future. Instead, you must define the derivative as:

    >>> def derivative(x, t):
    >>>   dx = x ** 3 - x ** 2
    >>>   return dx
    >>>
    >>> f = bp.ode.ExponentialEuler(derivative, show_code=True)
    def brainpy_itg_of_ode3_dev(x, t, dt=0.1):
      _dfx_dt = -x**2.0 + x**3.0
      _x_linear = -2.0*x**1.0 + 3.0*x**2.0
      _x_linear_exp = math.exp(_x_linear * dt)
      _x_df_part = _dfx_dt*(_x_linear_exp - 1)/_x_linear
      x_new = _x_df_part + x
      return x_new

    So, what ``brainpy.math`` functions are supported on derivative variables?
    You can inspect them by:

    >>> bp.integrators.analysis_by_sympy.FUNCTION_MAPPING.keys()
    {'abs': Abs,
     'sign': sign,
     'sinc': sinc,
     'arcsin': asin,
     'arccos': acos,
     'arctan': atan,
     'arctan2': atan2,
     'arcsinh': asinh,
     'arccosh': acosh,
     'arctanh': atanh,
     'log2': log2,
     'log1p': log1p,
     'expm1': expm1,
     'exp2': exp2,
     'asin': asin,
     'acos': acos,
     'atan': atan,
     'atan2': atan2,
     'asinh': asinh,
     'acosh': acosh,
     'atanh': atanh,
     'cos': cos,
     'sin': sin,
     'tan': tan,
     'cosh': cosh,
     'sinh': sinh,
     'tanh': tanh,
     'log': log,
     'log10': log10,
     'sqrt': sqrt,
     'exp': exp,
     'hypot': hypot,
     'ceil': ceiling,
     'floor': floor}

    However, if your general Python functions are used in variables not related to
    the derivative variable, it will work. For instance,

    >>> def derivative(x, t, y):
    >>>   alpha = bm.power(y, 3) - bm.power(y, 2)
    >>>   beta = bm.cumsum(y)
    >>>   dx = alpha * x - beta * (1 - x)
    >>>   return dx
    >>>
    >>> bp.ode.ExponentialEuler(derivative, show_code=True)

    2. Functional return only support symbols, not expressions.

    For example, this derivative will cause an error:

    >>> def derivative(x, t, tau):
    >>>   return -x / tau
    >>>
    >>> bp.ode.ExponentialEuler(derivative)
    brainpy.errors.DiffEqError: Cannot analyze differential equation with expression return. Like:
    def df(v, t):
        return -v + 1.
    We only support return variables. Therefore, the above should be coded as:
    def df(v, t):
        dv = -v + 1.
        return dv

    Instead, you should wrap the expression ``-x / tau`` as a symbol ``dx``, then return ``dx``.

    >>> def derivative(x, t, tau):
    >>>   dx = -x / tau
    >>>   return dx
    >>>
    >>> bp.ode.ExponentialEuler(derivative)
    <brainpy.integrators.ode.exponential.ExponentialEuler at 0x29daa298d60>

  See Also
  --------
  ExpEulerAuto

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
  timeout : float
    The timeout limit to use sympy solver.
  """

  def __init__(self,
               f,
               var_type=None,
               dt=None,
               name=None,
               show_code=False,
               timeout=5,
               state_delays=None,
               neutral_delays=None):
    super(ExponentialEuler, self).__init__(f=f,
                                           var_type=var_type,
                                           dt=dt,
                                           name=name,
                                           show_code=show_code,
                                           state_delays=state_delays,
                                           neutral_delays=neutral_delays)

    self.timeout = timeout

    # keyword checking
    keywords = {
      C.F: 'the derivative function',
      # C.DT: 'the precision of numerical integration',
      'exp': 'the exponential function',
      'math': 'the math module',
    }
    for v in self.variables:
      keywords[f'{v}_new'] = 'the intermediate value'
    utils.check_kws(self.arg_names, keywords)

    # build the integrator
    self.build()

  def build(self):
    if analysis_by_sympy is None or sympy is None:
      raise errors.PackageMissingError(f'Package "sympy" must be installed when the users '
                                       f'want to utilize {ExponentialEuler.__name__}. ')

    # check bound method
    if hasattr(self.f, '__self__'):
      self.code_lines = [f'def {self.func_name}({", ".join(["self"] + list(self.arguments))}):']

    # code scope
    closure_vars = inspect.getclosurevars(self.f)
    self.code_scope.update(closure_vars.nonlocals)
    self.code_scope.update(dict(closure_vars.globals))
    self.code_scope['math'] = math

    analysis = separate_variables(self.f)
    variables_for_returns = analysis['variables_for_returns']
    expressions_for_returns = analysis['expressions_for_returns']
    for vi, (key, all_var) in enumerate(variables_for_returns.items()):
      # separate variables
      sd_variables = []
      for v in all_var:
        if len(v) > 1:
          raise ValueError(f'Cannot analyze multi-assignment code line: {v}.')
        sd_variables.append(v[0])
      expressions = expressions_for_returns[key]
      var_name = self.variables[vi]
      diff_eq = analysis_by_sympy.SingleDiffEq(var_name=var_name,
                                               variables=sd_variables,
                                               expressions=expressions,
                                               derivative_expr=key,
                                               scope=self.code_scope,
                                               func_name=self.func_name)
      var = sympy.Symbol(diff_eq.var_name, real=True)
      try:
        s_df_part = tools.timeout(self.timeout)(self.solve)(diff_eq, var)
      except KeyboardInterrupt:
        raise errors.DiffEqError(
          f'{self.__class__} solve {self.f} failed, because '
          f'symbolic differentiation of SymPy timeout due to {self.timeout} s limit. '
          f'Instead, you can use {ExpEulerAuto} to make Exponential Euler '
          f'integration due to due to it is capable of '
          f'performing automatic differentiation.'
        )
      # update expression
      update = var + s_df_part

      # The actual update step
      self.code_lines.append(f'  {diff_eq.var_name}_new = {analysis_by_sympy.sympy2str(update)}')
      self.code_lines.append('')

    self.code_lines.append(f'  return {", ".join([f"{v}_new" for v in self.variables])}')
    self.integral = utils.compile_code(code_scope={k: v for k, v in self.code_scope.items()},
                                       code_lines=self.code_lines,
                                       show_code=self.show_code,
                                       func_name=self.func_name)

    if hasattr(self.f, '__self__'):
      host = self.f.__self__
      self.integral = self.integral.__get__(host, host.__class__)

  def solve(self, diff_eq, var):
    if analysis_by_sympy is None or sympy is None:
      raise errors.PackageMissingError(f'Package "sympy" must be installed when the users '
                                       f'want to utilize {ExponentialEuler.__name__}. ')

    f_expressions = diff_eq.get_f_expressions(substitute_vars=diff_eq.var_name)

    # code lines
    self.code_lines.extend([f"  {str(expr)}" for expr in f_expressions[:-1]])

    # get the linear system using sympy
    f_res = f_expressions[-1]
    if len(f_res.code) > 500:
      raise errors.DiffEqError(
        f'Too complex differential equation:\n\n'
        f'{f_res.code}\n\n'
        f'SymPy cannot analyze. Please use {ExpEulerAuto} to '
        f'make Exponential Euler integration due to it is capable of '
        f'performing automatic differentiation.'
      )
    df_expr = analysis_by_sympy.str2sympy(f_res.code).expr.expand()
    s_df = sympy.Symbol(f"{f_res.var_name}")
    self.code_lines.append(f'  {s_df.name} = {analysis_by_sympy.sympy2str(df_expr)}')

    # get df part
    s_linear = sympy.Symbol(f'_{diff_eq.var_name}_linear')
    s_linear_exp = sympy.Symbol(f'_{diff_eq.var_name}_linear_exp')
    s_df_part = sympy.Symbol(f'_{diff_eq.var_name}_df_part')
    if df_expr.has(var):
      # linear
      linear = sympy.diff(df_expr, var, evaluate=True)
      # TODO: linear has unknown symbol
      self.code_lines.append(f'  {s_linear.name} = {analysis_by_sympy.sympy2str(linear)}')
      # linear exponential
      self.code_lines.append(f'  {s_linear_exp.name} = math.exp({s_linear.name} * {C.DT})')
      # df part
      df_part = (s_linear_exp - 1) / s_linear * s_df
      self.code_lines.append(f'  {s_df_part.name} = {analysis_by_sympy.sympy2str(df_part)}')
    else:
      # df part
      self.code_lines.append(f'  {s_df_part.name} = {s_df.name} * {C.DT}')
    return s_df_part


register_ode_integrator('exponential_euler', ExponentialEuler)
register_ode_integrator('exp_euler', ExponentialEuler)


class ExpEulerAuto(ODEIntegrator):
  """Exponential Euler method using automatic differentiation.

  This method uses `brainpy.math.vector_grad <../../math/generated/brainpy.math.autograd.vector_grad.html>`_
  to automatically infer the linear part of the given function. Therefore, it has minimal constraints
  on your derivative function. Arbitrary complex functions can be numerically integrated with this method.

  Examples
  --------

  Here is an example uses ``ExpEulerAuto`` to implement HH neuron model.

  .. plot::
    :include-source: True

    >>> import brainpy as bp
    >>> import brainpy.math as bm
    >>>
    >>> class HH(bp.dyn.NeuGroup):
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
    >>>     self.int_h = bp.ode.ExpEulerAuto(self.dh)
    >>>     self.int_n = bp.ode.ExpEulerAuto(self.dn)
    >>>     self.int_V = bp.ode.ExpEulerAuto(self.dV)
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
    >>>   def update(self, _t, _dt):
    >>>     h = self.int_h(self.h, _t, self.V, dt=_dt)
    >>>     n = self.int_n(self.n, _t, self.V, dt=_dt)
    >>>     V = self.int_V(self.V, _t,  self.h, self.n, self.input, dt=_dt)
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
    >>> class HH(bp.dyn.NeuGroup):
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
    >>>     self.integral = bp.ode.ExpEulerAuto(derivative)
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
    >>>   def update(self, _t, _dt):
    >>>     h, n, V = self.integral(self.h, self.n, self.V, _t, self.input, dt=_dt)
    >>>     self.spike.value = bm.logical_and(self.V < self.V_th, V >= self.V_th)
    >>>     self.V.value = V
    >>>     self.h.value = h
    >>>     self.n.value = n
    >>>     self.input[:] = 0.
    >>>
    >>> run = bp.dyn.DSRunner(HH(1), inputs=('input', 2.), monitors=['V'], dt=0.05)
    >>> run(100)
    >>> bp.visualize.line_plot(run.mon.ts, run.mon.V, legend='V', show=True)

  See Also
  --------
  ExponentialEuler

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
  show_code : bool
  dyn_var : optional, dict, sequence of JaxArray, JaxArray
  """

  def __init__(self,
               f,
               var_type=None,
               dt=None,
               name=None,
               show_code=False,
               dyn_var=None,
               state_delays=None,
               neutral_delays=None):
    super(ExpEulerAuto, self).__init__(f=f,
                                       var_type=var_type,
                                       dt=dt,
                                       name=name,
                                       show_code=show_code,
                                       state_delays=state_delays,
                                       neutral_delays=neutral_delays)

    self.dyn_var = dyn_var

    # keyword checking
    keywords = {
      C.F: 'the derivative function',
      # C.DT: 'the precision of numerical integration',
    }
    utils.check_kws(self.arg_names, keywords)

    # build the integrator
    self.code_lines = []
    self.code_scope = {}
    self.integral = self.build()

  def build(self):
    all_vars, all_pars = [], []
    integrals, arg_names = [], []
    a = self._build_integrator(self.f)
    for integral, vars, _ in a:
      integrals.append(integral)
      for var in vars:
        if var not in all_vars:
          all_vars.append(var)
    for _, vars, pars in a:
      for par in pars:
        if (par not in all_vars) and (par not in all_pars):
          all_pars.append(par)
      arg_names.append(vars + pars + ['dt'])
    all_pars.append('dt')
    all_vps = all_vars + all_pars

    def integral_func(*args, **kwargs):
      # format arguments
      params_in = Collector()
      for i, arg in enumerate(args):
        params_in[all_vps[i]] = arg
      params_in.update(kwargs)
      if 'dt' not in params_in:
        params_in['dt'] = math.get_dt()

      # call integrals
      results = []
      for i, int_fun in enumerate(integrals):
        _key = arg_names[i][0]
        r = int_fun(params_in[_key], **{arg: params_in[arg] for arg in arg_names[i][1:] if arg in params_in})
        results.append(r)
      return results if isinstance(self.f, joint_eq.JointEq) else results[0]

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
        raise errors.DiffEqError(f'{self.__class__} only supports numerical integration '
                                 f'for one variable once, while we got {vars} in {eq}. '
                                 f'Please split your multiple variables into multiple '
                                 f'derivative functions.')

      # gradient function
      value_and_grad = math.vector_grad(eq, argnums=0, dyn_vars=self.dyn_var, return_value=True)

      # integration function
      def integral(*args, **kwargs):
        assert len(args) > 0
        dt = kwargs.pop('dt', math.get_dt())
        linear, derivative = value_and_grad(*args, **kwargs)
        phi = math.where(linear == 0., math.ones_like(linear),
                         (math.exp(dt * linear) - 1) / (dt * linear))
        return args[0] + dt * phi * derivative

      return [(integral, vars, pars), ]


register_ode_integrator('exp_euler_auto', ExpEulerAuto)
register_ode_integrator('exp_auto', ExpEulerAuto)
