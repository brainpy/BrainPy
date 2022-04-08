# -*- coding: utf-8 -*-


r"""This module provides adaptive Runge-Kutta methods for ODEs.

Adaptive methods are designed to produce an estimate of the local truncation
error of a single Runge–Kutta step. This is done by having two methods,
one with order :math:`p` and one with order :math:`p-1`. These methods are
interwoven, i.e., they have common intermediate steps. Thanks to this, estimating
the error has little or negligible computational cost compared to a step with
the higher-order method.

During the integration, the step size is adapted such that the estimated error
stays below a user-defined threshold: If the error is too high, a step is repeated
with a lower step size; if the error is much smaller, the step size is increased
to save time. This results in an (almost) optimal step size, which saves computation
time. Moreover, the user does not have to spend time on finding an appropriate step size.

The lower-order step is given by

.. math::
    y_{n+1}^{*}=y_{n}+h\sum _{i=1}^{s}b_{i}^{*}k_{i},

where :math:`k_{i}` are the same as for the higher-order method. Then the error is

.. math::
    e_{n+1}=y_{n+1}-y_{n+1}^{*}=h\sum _{i=1}^{s}(b_{i}-b_{i}^{*})k_{i},

which is (:math:`O(h^{p}`).

The Butcher tableau for this kind of method is extended to give the values of
:math:`b_{i}^{*}`:

.. math::
    \begin{array}{c|llll}
    0 & & & & & \\
    c_{2} & a_{21} & & & & \\
    c_{3} & a_{31} & a_{32} & & & \\
    \vdots & \vdots & & \ddots & \\
    c_{s} & a_{s 1} & a_{s 2} & \cdots & a_{s, s-1} \\
    \hline & b_{1} & b_{2} & \cdots & b_{s-1} & b_{s} \\
        & b_{1}^{*} & b_{2}^{*} & \cdots & b_{s-1}^{*} & b_{s}^{*}
    \end{array}

More details please check [1]_ [2]_ [3]_.


.. [1] https://en.wikipedia.org/wiki/Runge%E2%80%93Kutta_methods
.. [2] Press, W.H., Press, W.H., Flannery, B.P., Teukolsky, S.A., Vetterling, W.T.,
       Flannery, B.P. and Vetterling, W.T., 1989. Numerical recipes in Pascal: the
       art of scientific computing (Vol. 1). Cambridge university press.
.. [3] Press, W. H., & Teukolsky, S. A. (1992). Adaptive Stepsize Runge‐Kutta Integration.
       Computers in Physics, 6(2), 188-191.
"""

import brainpy.math as bm
from brainpy import errors
from brainpy.integrators import constants as C, utils
from brainpy.integrators.ode import common
from brainpy.integrators.ode.base import ODEIntegrator
from .generic import register_ode_integrator

__all__ = [
  'AdaptiveRKIntegrator',
  'RKF12',
  'RKF45',
  'DormandPrince',
  'CashKarp',
  'BogackiShampine',
  'HeunEuler',
]


class AdaptiveRKIntegrator(ODEIntegrator):
  r"""Adaptive Runge-Kutta method for ordinary differential equations.

  The embedded methods are designed to produce an estimate of the local
  truncation error of a single Runge-Kutta step, and as result, allow to
  control the error with adaptive step-size. This is done by having two
  methods in the tableau, one with order p and one with order :math:`p-1`.

  The lower-order step is given by

  .. math::

      y^*_{n+1} = y_n + h\sum_{i=1}^s b^*_i k_i,

  where the :math:`k_{i}` are the same as for the higher order method. Then the error is

  .. math::

      e_{n+1} = y_{n+1} - y^*_{n+1} = h\sum_{i=1}^s (b_i - b^*_i) k_i,


  which is :math:`O(h^{p})`. The Butcher Tableau for this kind of method is extended to
  give the values of :math:`b_{i}^{*}`

  .. math::

      \begin{array}{c|cccc}
          c_1    & a_{11} & a_{12}& \dots & a_{1s}\\
          c_2    & a_{21} & a_{22}& \dots & a_{2s}\\
          \vdots & \vdots & \vdots& \ddots& \vdots\\
          c_s    & a_{s1} & a_{s2}& \dots & a_{ss} \\
      \hline & b_1    & b_2   & \dots & b_s\\
             & b_1^*    & b_2^*   & \dots & b_s^*\\
      \end{array}

  Parameters
  ----------
  f : callable
    The derivative function.
  show_code : bool
    Whether show the formatted code.
  dt : float
    The numerical precision.
  adaptive : bool
    Whether use the adaptive updating.
  tol : float
    The error tolerence.
  var_type : str
    The variable type.
  """

  A = []  # The A matrix in the Butcher tableau.
  B1 = []  # The B1 vector in the Butcher tableau.
  B2 = []  # The B2 vector in the Butcher tableau.
  C = []  # The C vector in the Butcher tableau.

  def __init__(self,
               f,
               var_type=None,
               dt=None,
               name=None,
               adaptive=None,
               tol=None,
               show_code=False,
               state_delays=None,
               neutral_delays=None):
    super(AdaptiveRKIntegrator, self).__init__(f=f,
                                               var_type=var_type,
                                               dt=dt,
                                               name=name,
                                               show_code=show_code,
                                               state_delays=state_delays,
                                               neutral_delays=neutral_delays)

    # check parameters
    self.adaptive = False if (adaptive is None) else adaptive
    self.tol = 0.1 if tol is None else tol
    self.var_type = C.POP_VAR if var_type is None else var_type
    if self.var_type not in C.SUPPORTED_VAR_TYPE:
      raise errors.IntegratorError(f'"var_type" only supports {C.SUPPORTED_VAR_TYPE}, '
                                   f'not {self.var_type}.')

    # integrator keywords
    keywords = {
      C.F: 'the derivative function',
      # C.DT: 'the precision of numerical integration'
    }
    for v in self.variables:
      keywords[f'{v}_new'] = 'the intermediate value'
      for i in range(1, len(self.A) + 1):
        keywords[f'd{v}_k{i}'] = 'the intermediate value'
      for i in range(2, len(self.A) + 1):
        keywords[f'k{i}_{v}_arg'] = 'the intermediate value'
        keywords[f'k{i}_t_arg'] = 'the intermediate value'
    if adaptive:
      keywords['dt_new'] = 'the new numerical precision "dt"'
      keywords['tol'] = 'the tolerance for the local truncation error'
      keywords['error'] = 'the local truncation error'
      for v in self.variables:
        keywords[f'{v}_te'] = 'the local truncation error'
      self.code_scope['tol'] = tol
      self.code_scope['math'] = bm
    utils.check_kws(self.arg_names, keywords)

    # build the integrator
    self.build()

  def build(self):
    # step stage
    common.step(self.variables, C.DT,
                self.A, self.C, self.code_lines, self.parameters)
    # variable update
    return_args = common.update(self.variables, C.DT, self.B1, self.code_lines)
    # error adaptive item
    if self.adaptive:
      errors_ = []
      for v in self.variables:
        result = []
        for i, (b1, b2) in enumerate(zip(self.B1, self.B2)):
          if isinstance(b1, str):
            b1 = eval(b1)
          if isinstance(b2, str):
            b2 = eval(b2)
          diff = b1 - b2
          if diff != 0.:
            result.append(f'd{v}_k{i + 1} * {C.DT} * {diff}')
        if len(result) > 0:
          if self.var_type == C.SCALAR_VAR:
            self.code_lines.append(f'  {v}_te = abs({" + ".join(result)})')
          else:
            self.code_lines.append(f'  {v}_te = sum(abs({" + ".join(result)}))')
          errors_.append(f'{v}_te')
      if len(errors_) > 0:
        self.code_lines.append(f'  error = {" + ".join(errors_)}')
        self.code_lines.append(f'  {C.DT}_new = math.where(error > tol, 0.9*{C.DT}*(tol/error)**0.2, {C.DT})')
        return_args.append(f'{C.DT}_new')
    # returns
    self.code_lines.append(f'  return {", ".join(return_args)}')
    # compile
    self.integral = utils.compile_code(
      code_scope={k: v for k, v in self.code_scope.items()},
      code_lines=self.code_lines,
      show_code=self.show_code,
      func_name=self.func_name)


class RKF12(AdaptiveRKIntegrator):
  r"""The Fehlberg RK1(2) method for ODEs.

  The Fehlberg method has two methods of orders 1 and 2.

  It has the characteristics of:

      - method stage = 2
      - method order = 1
      - Butcher Tables:

  .. math::

      \begin{array}{l|ll}
          0 & & \\
          1 / 2 & 1 / 2 & \\
          1 & 1 / 256 & 255 / 256 & \\
          \hline & 1 / 512 & 255 / 256 & 1 / 512 \\
          & 1 / 256 & 255 / 256 & 0
      \end{array}

  References
  ----------

  .. [1] Fehlberg, E. (1969-07-01). "Low-order classical Runge-Kutta
          formulas with stepsize control and their application to some heat
          transfer problems"

  """

  A = [(),
       (0.5,),
       ('1/256', '255/256')]
  B1 = ['1/512', '255/256', '1/512']
  B2 = ['1/256', '255/256', 0]
  C = [0, 0.5, 1]


register_ode_integrator('rkf12', RKF12)


class RKF45(AdaptiveRKIntegrator):
  r"""The Runge–Kutta–Fehlberg method for ODEs.

  The method presented in Fehlberg's 1969 paper has been dubbed the
  RKF45 method, and is a method of order :math:`O(h^4)` with an error
  estimator of order :math:`O(h^5)`. The novelty of Fehlberg's method is
  that it is an embedded method from the Runge–Kutta family, meaning that
  identical function evaluations are used in conjunction with each other
  to create methods of varying order and similar error constants.

  Its Butcher table is:

  .. math::

      \begin{array}{l|lllll}
          0 & & & & & & \\
          1 / 4 & 1 / 4 & & & & \\
          3 / 8 & 3 / 32 & 9 / 32 & & \\
          12 / 13 & 1932 / 2197 & -7200 / 2197 & 7296 / 2197 & \\
          1 & 439 / 216 & -8 & 3680 / 513 & -845 / 4104 & & \\
          1 / 2 & -8 / 27 & 2 & -3544 / 2565 & 1859 / 4104 & -11 / 40 & \\
          \hline & 16 / 135 & 0 & 6656 / 12825 & 28561 / 56430 & -9 / 50 & 2 / 55 \\
          & 25 / 216 & 0 & 1408 / 2565 & 2197 / 4104 & -1 / 5 & 0
      \end{array}

  References
  ----------

  .. [1] https://en.wikipedia.org/wiki/Runge%E2%80%93Kutta%E2%80%93Fehlberg_method
  .. [2] Erwin Fehlberg (1969). Low-order classical Runge-Kutta formulas with step
         size control and their application to some heat transfer problems . NASA Technical Report 315.
         https://ntrs.nasa.gov/api/citations/19690021375/downloads/19690021375.pdf

  """

  A = [(),
       (0.25,),
       (0.09375, 0.28125),
       ('1932/2197', '-7200/2197', '7296/2197'),
       ('439/216', -8, '3680/513', '-845/4104'),
       ('-8/27', 2, '-3544/2565', '1859/4104', -0.275)]
  B1 = ['16/135', 0, '6656/12825', '28561/56430', -0.18, '2/55']
  B2 = ['25/216', 0, '1408/2565', '2197/4104', -0.2, 0]
  C = [0, 0.25, 0.375, '12/13', 1, '1/3']


register_ode_integrator('rkf45', RKF45)


class DormandPrince(AdaptiveRKIntegrator):
  r"""The Dormand–Prince method for ODEs.

  The DOPRI method, is an explicit method for solving ordinary differential equations
  (Dormand & Prince 1980). The Dormand–Prince method has seven stages, but it uses only
  six function evaluations per step because it has the FSAL (First Same As Last) property:
  the last stage is evaluated at the same point as the first stage of the next step.
  Dormand and Prince chose the coefficients of their method to minimize the error of
  the fifth-order solution. This is the main difference with the Fehlberg method, which
  was constructed so that the fourth-order solution has a small error. For this reason,
  the Dormand–Prince method is more suitable when the higher-order solution is used to
  continue the integration, a practice known as local extrapolation
  (Shampine 1986; Hairer, Nørsett & Wanner 2008, pp. 178–179).

  Its Butcher table is:

  .. math::

      \begin{array}{l|llllll}
          0 &  \\
          1 / 5 & 1 / 5 & & & \\
          3 / 10 & 3 / 40 & 9 / 40 & & & \\
          4 / 5 & 44 / 45 & -56 / 15 & 32 / 9 & & \\
          8 / 9 & 19372 / 6561 & -25360 / 2187 & 64448 / 6561 & -212 / 729 & \\
          1 & 9017 / 3168 & -355 / 33 & 46732 / 5247 & 49 / 176 & -5103 / 18656 & \\
          1 & 35 / 384 & 0 & 500 / 1113 & 125 / 192 & -2187 / 6784 & 11 / 84 & \\
          \hline & 35 / 384 & 0 & 500 / 1113 & 125 / 192 & -2187 / 6784 & 11 / 84 & 0 \\
          & 5179 / 57600 & 0 & 7571 / 16695 & 393 / 640 & -92097 / 339200 & 187 / 2100 & 1 / 40
      \end{array}

  References
  ----------

  .. [1] https://en.wikipedia.org/wiki/Dormand%E2%80%93Prince_method
  .. [2] Dormand, J. R.; Prince, P. J. (1980), "A family of embedded Runge-Kutta formulae",
         Journal of Computational and Applied Mathematics, 6 (1): 19–26,
         doi:10.1016/0771-050X(80)90013-3.
  """

  A = [(),
       (0.2,),
       (0.075, 0.225),
       ('44/45', '-56/15', '32/9'),
       ('19372/6561', '-25360/2187', '64448/6561', '-212/729'),
       ('9017/3168', '-355/33', '46732/5247', '49/176', '-5103/18656'),
       ('35/384', 0, '500/1113', '125/192', '-2187/6784', '11/84')]
  B1 = ['35/384', 0, '500/1113', '125/192', '-2187/6784', '11/84', 0]
  B2 = ['5179/57600', 0, '7571/16695', '393/640', '-92097/339200', '187/2100', 0.025]
  C = [0, 0.2, 0.3, 0.8, '8/9', 1, 1]


register_ode_integrator('rkdp', DormandPrince)


class CashKarp(AdaptiveRKIntegrator):
  r"""The Cash–Karp method  for ODEs.

  The Cash–Karp method was proposed by Professor Jeff R. Cash from Imperial College London
  and Alan H. Karp from IBM Scientific Center. it uses six function evaluations to calculate
  fourth- and fifth-order accurate solutions. The difference between these solutions is then
  taken to be the error of the (fourth order) solution. This error estimate is very convenient
  for adaptive stepsize integration algorithms.

  It has the characteristics of:

      - method stage = 6
      - method order = 4
      - Butcher Tables:

  .. math::

      \begin{array}{l|lllll}
          0 & & & & & & \\
          1 / 5 & 1 / 5 & & & & & \\
          3 / 10 & 3 / 40 & 9 / 40 & & & \\
          3 / 5 & 3 / 10 & -9 / 10 & 6 / 5 & & \\
          1 & -11 / 54 & 5 / 2 & -70 / 27 & 35 / 27 & & \\
          7 / 8 & 1631 / 55296 & 175 / 512 & 575 / 13824 & 44275 / 110592 & 253 / 4096 & \\
          \hline & 37 / 378 & 0 & 250 / 621 & 125 / 594 & 0 & 512 / 1771 \\
          & 2825 / 27648 & 0 & 18575 / 48384 & 13525 / 55296 & 277 / 14336 & 1 / 4
      \end{array}

  References
  ----------

  .. [1] https://en.wikipedia.org/wiki/Cash%E2%80%93Karp_method
  .. [2] J. R. Cash, A. H. Karp. "A variable order Runge-Kutta method for initial value
         problems with rapidly varying right-hand sides", ACM Transactions on Mathematical
         Software 16: 201-222, 1990. doi:10.1145/79505.79507
  """

  A = [(),
       (0.2,),
       (0.075, 0.225),
       (0.3, -0.9, 1.2),
       ('-11/54', 2.5, '-70/27', '35/27'),
       ('1631/55296', '175/512', '575/13824', '44275/110592', '253/4096')]
  B1 = ['37/378', 0, '250/621', '125/594', 0, '512/1771']
  B2 = ['2825/27648', 0, '18575/48384', '13525/55296', '277/14336', 0.25]
  C = [0, 0.2, 0.3, 0.6, 1, 0.875]


register_ode_integrator('ck', CashKarp)


class BogackiShampine(AdaptiveRKIntegrator):
  r"""The Bogacki–Shampine method for ODEs.

  The Bogacki–Shampine method was proposed by Przemysław Bogacki and Lawrence F.
  Shampine in 1989 (Bogacki & Shampine 1989). The Bogacki–Shampine method is a
  Runge–Kutta method of order three with four stages with the First Same As Last
  (FSAL) property, so that it uses approximately three function evaluations per
  step. It has an embedded second-order method which can be used to implement adaptive step size.

  It has the characteristics of:

      - method stage = 4
      - method order = 3
      - Butcher Tables:

  .. math::

      \begin{array}{l|lll}
          0 & & & \\
          1 / 2 & 1 / 2 & & \\
          3 / 4 & 0 & 3 / 4 & \\
          1 & 2 / 9 & 1 / 3 & 4 / 9 \\
          \hline & 2 / 9 & 1 / 3 & 4 / 90 \\
          & 7 / 24 & 1 / 4 & 1 / 3 & 1 / 8
      \end{array}

  References
  ----------

  .. [1] https://en.wikipedia.org/wiki/Bogacki%E2%80%93Shampine_method
  .. [2] Bogacki, Przemysław; Shampine, Lawrence F. (1989), "A 3(2) pair of Runge–Kutta
         formulas", Applied Mathematics Letters, 2 (4): 321–325, doi:10.1016/0893-9659(89)90079-7
  """

  A = [(),
       (0.5,),
       (0., 0.75),
       ('2/9', '1/3', '4/0'), ]
  B1 = ['2/9', '1/3', '4/9', 0]
  B2 = ['7/24', 0.25, '1/3', 0.125]
  C = [0, 0.5, 0.75, 1]


register_ode_integrator('bs', BogackiShampine)


class HeunEuler(AdaptiveRKIntegrator):
  r"""The Heun–Euler method for ODEs.

  The simplest adaptive Runge–Kutta method involves combining Heun's method,
  which is order 2, with the Euler method, which is order 1.

  It has the characteristics of:

      - method stage = 2
      - method order = 1
      - Butcher Tables:

  .. math::

      \begin{array}{c|cc}
          0&\\
          1& 	1 \\
      \hline
      &	1/2& 	1/2\\
          &	1 &	0
      \end{array}

  """

  A = [(), (1,)]
  B1 = [0.5, 0.5]
  B2 = [1, 0]
  C = [0, 1]


register_ode_integrator('heun_euler', HeunEuler)


class DOP853(AdaptiveRKIntegrator):
  # def DOP853(f=None, tol=None, adaptive=None, dt=None, show_code=None, each_var_is_scalar=None):
  r"""The DOP853 method for ODEs.

  DOP853 is an explicit Runge-Kutta method of order 8(5,3) due to Dormand & Prince
  (with stepsize control and dense output).

  References
  ----------

  .. [1] E. Hairer, S.P. Norsett and G. Wanner, "Solving ordinary Differential Equations
         I. Nonstiff Problems", 2nd edition. Springer Series in Computational Mathematics,
         Springer-Verlag (1993).
  .. [2] http://www.unige.ch/~hairer/software.html
  """
  pass


class BoSh3(AdaptiveRKIntegrator):
  """
  Bogacki--Shampine's 3/2 method.

  3rd order explicit Runge--Kutta method. Has an embedded 2nd order method for
  adaptive step sizing.

  """
  A = [(),
       (0.5,),
       (0.0, 0.75),
       ('2/9', '1/3', '4/9')]
  B1 = ['2/9', '1/3', '4/9', 0.0]
  B2 = ['-5/72', 1 / 12, '1/9', '-1/8']
  C = [0., 0.5, 0.75, 1.0]


register_ode_integrator('BoSh3', BoSh3)
