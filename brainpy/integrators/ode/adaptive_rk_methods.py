# -*- coding: utf-8 -*-

from brainpy import backend
from brainpy.integrators import constants
from .wrapper import adaptive_rk_wrapper

__all__ = [
  'rkf45',
  'rkf12',
  'rkdp',
  'ck',
  'bs',
  'heun_euler'
]


def _base(A, B1, B2, C, f=None, tol=None, adaptive=None,
          dt=None, show_code=None, var_type=None):
  """

  Parameters
  ----------
  A : list
  B1 : list
  B2 : list
  C : list
  f : callable
  tol : float
  adaptive : bool
  im_return : list
      Intermediate value return.
  dt : float
  show_code : bool
  var_type : str

  Returns
  -------

  """
  adaptive = False if (adaptive is None) else adaptive
  dt = backend.get_dt() if (dt is None) else dt
  tol = 0.1 if tol is None else tol
  show_code = False if tol is None else show_code
  var_type = constants.SCALAR_VAR if var_type is None else var_type

  if f is None:
    return lambda f: adaptive_rk_wrapper(f,
                                         dt=dt,
                                         A=A,
                                         B1=B1,
                                         B2=B2,
                                         C=C,
                                         tol=tol,
                                         adaptive=adaptive,
                                         show_code=show_code,
                                         var_type=var_type)
  else:
    return adaptive_rk_wrapper(f,
                               dt=dt,
                               A=A,
                               B1=B1,
                               B2=B2,
                               C=C,
                               tol=tol,
                               adaptive=adaptive,
                               show_code=show_code,
                               var_type=var_type)


def rkf45(f=None, tol=None, adaptive=None, dt=None, show_code=None, var_type=None):
  """The Runge–Kutta–Fehlberg method for ordinary differential equations.

  The method presented in Fehlberg's 1969 paper has been dubbed the
  RKF45 method, and is a method of order :math:`O(h^4)` with an error
  estimator of order :math:`O(h^5)`. The novelty of Fehlberg's method is
  that it is an embedded method from the Runge–Kutta family, meaning that
  identical function evaluations are used in conjunction with each other
  to create methods of varying order and similar error constants.

  It has the characteristics of:

      - method stage = 6
      - method order = 5
      - Butcher Tables:

  .. math::

      \\begin{array}{l|lllll}
          0 & & & & & & \\\\
          1 / 4 & 1 / 4 & & & & \\\\
          3 / 8 & 3 / 32 & 9 / 32 & & \\\\
          12 / 13 & 1932 / 2197 & -7200 / 2197 & 7296 / 2197 & \\\\
          1 & 439 / 216 & -8 & 3680 / 513 & -845 / 4104 & & \\\\
          1 / 2 & -8 / 27 & 2 & -3544 / 2565 & 1859 / 4104 & -11 / 40 & \\\\
          \\hline & 16 / 135 & 0 & 6656 / 12825 & 28561 / 56430 & -9 / 50 & 2 / 55 \\\\
          & 25 / 216 & 0 & 1408 / 2565 & 2197 / 4104 & -1 / 5 & 0
      \\end{array}

  References
  ----------

  [1] https://en.wikipedia.org/wiki/Runge%E2%80%93Kutta%E2%80%93Fehlberg_method
  [2] Erwin Fehlberg (1969). Low-order classical Runge-Kutta formulas with step
      size control and their application to some heat transfer problems . NASA
      Technical Report 315.
      https://ntrs.nasa.gov/api/citations/19690021375/downloads/19690021375.pdf

  """

  A = [(), (0.25,), (0.09375, 0.28125),
       ('1932/2197', '-7200/2197', '7296/2197'),
       ('439/216', -8, '3680/513', '-845/4104'),
       ('-8/27', 2, '-3544/2565', '1859/4104', -0.275)]
  B1 = ['16/135', 0, '6656/12825', '28561/56430', -0.18, '2/55']
  B2 = ['25/216', 0, '1408/2565', '2197/4104', -0.2, 0]
  C = [0, 0.25, 0.375, '12/13', 1, '1/3']

  return _base(A=A,
               B1=B1,
               B2=B2,
               C=C,
               f=f,
               dt=dt,
               tol=tol,
               adaptive=adaptive,
               show_code=show_code,
               var_type=var_type)


def rkf12(f=None, tol=None, adaptive=None, dt=None, show_code=None, var_type=None):
  """The Fehlberg RK1(2) method for ordinary differential equations.

  The Fehlberg method has two methods of orders 1 and 2.

  It has the characteristics of:

      - method stage = 2
      - method order = 1
      - Butcher Tables:

  .. math::

      \\begin{array}{l|ll}
          0 & & \\\\
          1 / 2 & 1 / 2 & \\\\
          1 & 1 / 256 & 255 / 256 & \\\\
          \\hline & 1 / 512 & 255 / 256 & 1 / 512 \\\\
          & 1 / 256 & 255 / 256 & 0
      \\end{array}

  References
  ----------

  .. [1] Fehlberg, E. (1969-07-01). "Low-order classical Runge-Kutta
          formulas with stepsize control and their application to some heat
          transfer problems"

  """

  A = [(), (0.5,), ('1/256', '255/256')]
  B1 = ['1/512', '255/256', '1/512']
  B2 = ['1/256', '255/256', 0]
  C = [0, 0.5, 1]

  return _base(A=A,
               B1=B1,
               B2=B2,
               C=C,
               f=f,
               dt=dt,
               tol=tol,
               adaptive=adaptive,
               show_code=show_code,
               var_type=var_type)


def rkdp(f=None, tol=None, adaptive=None, dt=None, show_code=None, var_type=None):
  """The Dormand–Prince method for ordinary differential equations.

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

  It has the characteristics of:

      - method stage = 7
      - method order = 5
      - Butcher Tables:

  .. math::

      \\begin{array}{l|llllll}
          0 &  \\\\
          1 / 5 & 1 / 5 & & & \\\\
          3 / 10 & 3 / 40 & 9 / 40 & & & \\\\
          4 / 5 & 44 / 45 & -56 / 15 & 32 / 9 & & \\\\
          8 / 9 & 19372 / 6561 & -25360 / 2187 & 64448 / 6561 & -212 / 729 & \\\\
          1 & 9017 / 3168 & -355 / 33 & 46732 / 5247 & 49 / 176 & -5103 / 18656 & \\\\
          1 & 35 / 384 & 0 & 500 / 1113 & 125 / 192 & -2187 / 6784 & 11 / 84 & \\\\
          \\hline & 35 / 384 & 0 & 500 / 1113 & 125 / 192 & -2187 / 6784 & 11 / 84 & 0 \\\\
          & 5179 / 57600 & 0 & 7571 / 16695 & 393 / 640 & -92097 / 339200 & 187 / 2100 & 1 / 40
      \\end{array}

  References
  ----------

  [1] https://en.wikipedia.org/wiki/Dormand%E2%80%93Prince_method
  [2] Dormand, J. R.; Prince, P. J. (1980), "A family of embedded Runge-Kutta formulae",
      Journal of Computational and Applied Mathematics, 6 (1): 19–26,
      doi:10.1016/0771-050X(80)90013-3.
  """

  A = [(), (0.2,), (0.075, 0.225),
       ('44/45', '-56/15', '32/9'),
       ('19372/6561', '-25360/2187', '64448/6561', '-212/729'),
       ('9017/3168', '-355/33', '46732/5247', '49/176', '-5103/18656'),
       ('35/384', 0, '500/1113', '125/192', '-2187/6784', '11/84')]
  B1 = ['35/384', 0, '500/1113', '125/192', '-2187/6784', '11/84', 0]
  B2 = ['5179/57600', 0, '7571/16695', '393/640', '-92097/339200', '187/2100', 0.025]
  C = [0, 0.2, 0.3, 0.8, '8/9', 1, 1]

  return _base(A=A,
               B1=B1,
               B2=B2,
               C=C,
               f=f,
               dt=dt,
               tol=tol,
               adaptive=adaptive,
               show_code=show_code,
               var_type=var_type)


def ck(f=None, tol=None, adaptive=None, dt=None, show_code=None, var_type=None):
  """The Cash–Karp method  for ordinary differential equations.

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

      \\begin{array}{l|lllll}
          0 & & & & & & \\\\
          1 / 5 & 1 / 5 & & & & & \\\\
          3 / 10 & 3 / 40 & 9 / 40 & & & \\\\
          3 / 5 & 3 / 10 & -9 / 10 & 6 / 5 & & \\\\
          1 & -11 / 54 & 5 / 2 & -70 / 27 & 35 / 27 & & \\\\
          7 / 8 & 1631 / 55296 & 175 / 512 & 575 / 13824 & 44275 / 110592 & 253 / 4096 & \\\\
          \\hline & 37 / 378 & 0 & 250 / 621 & 125 / 594 & 0 & 512 / 1771 \\\\
          & 2825 / 27648 & 0 & 18575 / 48384 & 13525 / 55296 & 277 / 14336 & 1 / 4
      \\end{array}

  References
  ----------

  [1] https://en.wikipedia.org/wiki/Cash%E2%80%93Karp_method
  [2] J. R. Cash, A. H. Karp. "A variable order Runge-Kutta method for initial value
      problems with rapidly varying right-hand sides", ACM Transactions on Mathematical
      Software 16: 201-222, 1990. doi:10.1145/79505.79507
  """

  A = [(), (0.2,), (0.075, 0.225), (0.3, -0.9, 1.2),
       ('-11/54', 2.5, '-70/27', '35/27'),
       ('1631/55296', '175/512', '575/13824', '44275/110592', '253/4096')]
  B1 = ['37/378', 0, '250/621', '125/594', 0, '512/1771']
  B2 = ['2825/27648', 0, '18575/48384', '13525/55296', '277/14336', 0.25]
  C = [0, 0.2, 0.3, 0.6, 1, 0.875]

  return _base(A=A,
               B1=B1,
               B2=B2,
               C=C,
               f=f,
               dt=dt,
               tol=tol,
               adaptive=adaptive,
               show_code=show_code,
               var_type=var_type)


def bs(f=None, tol=None, adaptive=None, dt=None, show_code=None, var_type=None):
  """The Bogacki–Shampine method for ordinary differential equations.

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

      \\begin{array}{l|lll}
          0 & & & \\\\
          1 / 2 & 1 / 2 & & \\\\
          3 / 4 & 0 & 3 / 4 & \\\\
          1 & 2 / 9 & 1 / 3 & 4 / 9 \\\\
          \\hline & 2 / 9 & 1 / 3 & 4 / 90 \\\\
          & 7 / 24 & 1 / 4 & 1 / 3 & 1 / 8
      \\end{array}

  References
  ----------

  [1] https://en.wikipedia.org/wiki/Bogacki%E2%80%93Shampine_method
  [2] Bogacki, Przemysław; Shampine, Lawrence F. (1989), "A 3(2) pair of Runge–Kutta
      formulas", Applied Mathematics Letters, 2 (4): 321–325, doi:10.1016/0893-9659(89)90079-7
  """

  A = [(), (0.5,), (0., 0.75), ('2/9', '1/3', '4/0'), ]
  B1 = ['2/9', '1/3', '4/9', 0]
  B2 = ['7/24', 0.25, '1/3', 0.125]
  C = [0, 0.5, 0.75, 1]

  return _base(A=A,
               B1=B1,
               B2=B2,
               C=C,
               f=f,
               dt=dt,
               tol=tol,
               adaptive=adaptive,
               show_code=show_code,
               var_type=var_type)


def heun_euler(f=None, tol=None, adaptive=None, dt=None, show_code=None, var_type=None):
  """The Heun–Euler method for ordinary differential equations.

  The simplest adaptive Runge–Kutta method involves combining Heun's method,
  which is order 2, with the Euler method, which is order 1.

  It has the characteristics of:

      - method stage = 2
      - method order = 1
      - Butcher Tables:

  .. math::

      \\begin{array}{c|cc}
          0&\\\\
          1& 	1 \\\\
      \\hline
      &	1/2& 	1/2\\\\
          &	1 &	0
      \\end{array}

  """

  A = [(), (1,)]
  B1 = [0.5, 0.5]
  B2 = [1, 0]
  C = [0, 1]

  return _base(A=A,
               B1=B1,
               B2=B2,
               C=C,
               f=f,
               dt=dt,
               tol=tol,
               adaptive=adaptive,
               show_code=show_code,
               var_type=var_type)


def DOP853(f=None, tol=None, adaptive=None, dt=None, show_code=None, each_var_is_scalar=None):
  """The DOP853 method for ordinary differential equations.

  DOP853 is an explicit Runge-Kutta method of order 8(5,3) due to Dormand & Prince
  (with stepsize control and dense output).

  References
  ----------

  [1] E. Hairer, S.P. Norsett and G. Wanner, "Solving ordinary Differential Equations
      I. Nonstiff Problems", 2nd edition. Springer Series in Computational Mathematics,
      Springer-Verlag (1993).
  [2] http://www.unige.ch/~hairer/software.html
  """
  pass
