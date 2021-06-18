# -*- coding: utf-8 -*-

from brainpy import backend
from brainpy.integrators import constants
from .wrapper import general_rk_wrapper
from .wrapper import rk2_wrapper

__all__ = [
  'euler',
  'midpoint',
  'heun2',
  'ralston2',
  'rk2',
  'rk3',
  'heun3',
  'ralston3',
  'ssprk3',
  'rk4',
  'ralston4',
  'rk4_38rule',
]


def _base(A, B, C, f, show_code, dt, var_type):
  dt = backend.get_dt() if dt is None else dt
  show_code = False if show_code is None else show_code
  var_type = constants.SCALAR_VAR if var_type is None else var_type

  if f is None:
    return lambda f: general_rk_wrapper(f=f, show_code=show_code, dt=dt, A=A, B=B, C=C,
                                        var_type=var_type)
  else:
    return general_rk_wrapper(f=f, show_code=show_code, dt=dt, A=A, B=B, C=C,
                              var_type=var_type)


def euler(f=None, show_code=None, dt=None, var_type=None):
  """The Euler method is first order. The lack of stability
      and accuracy limits its popularity mainly to use as a
      simple introductory example of a numeric solution method.
  """
  A = [(), ]
  B = [1]
  C = [0]
  return _base(A=A,
               B=B,
               C=C,
               f=f,
               show_code=show_code,
               dt=dt,
               var_type=var_type)


def midpoint(f=None, show_code=None, dt=None, var_type=None):
  """midpoint method for ordinary differential equations.

  The (explicit) midpoint method is a second-order method
  with two stages.

  It has the characteristics of:

      - method stage = 2
      - method order = 2
      - Butcher Tables:

  .. backend::

      \\begin{array}{c|cc}
          0 & 0 & 0 \\\\
          1 / 2 & 1 / 2 & 0 \\\\
          \\hline & 0 & 1
      \\end{array}

  """
  A = [(), (0.5,)]
  B = [0, 1]
  C = [0, 0.5]
  return _base(A=A,
               B=B,
               C=C,
               f=f,
               show_code=show_code,
               dt=dt,
               var_type=var_type)


def heun2(f=None, show_code=None, dt=None, var_type=None):
  """Heun's method for ordinary differential equations.

  Heun's method is a second-order method with two stages.
  It is also known as the explicit trapezoid rule, improved
  Euler's method, or modified Euler's method.

  It has the characteristics of:

      - method stage = 2
      - method order = 2
      - Butcher Tables:

  .. backend::

      \\begin{array}{c|cc}
          0.0 & 0.0 & 0.0 \\\\
          1.0 & 1.0 & 0.0 \\\\
          \\hline & 0.5 & 0.5
      \\end{array}

  """
  A = [(), (1,)]
  B = [0.5, 0.5]
  C = [0, 1]
  return _base(A=A,
               B=B,
               C=C,
               f=f,
               show_code=show_code,
               dt=dt,
               var_type=var_type)


def ralston2(f=None, show_code=None, dt=None, var_type=None):
  """Ralston's method for ordinary differential equations.

  Ralston's method is a second-order method with two stages and
  a minimum local error bound.

  It has the characteristics of:

      - method stage = 2
      - method order = 2
      - Butcher Tables:

  .. backend::

      \\begin{array}{c|cc}
          0 & 0 & 0 \\\\
          2 / 3 & 2 / 3 & 0 \\\\
          \\hline & 1 / 4 & 3 / 4
      \\end{array}
  """
  A = [(), ('2/3',)]
  B = [0.25, 0.75]
  C = [0, '2/3']
  return _base(A=A,
               B=B,
               C=C,
               f=f,
               show_code=show_code,
               dt=dt,
               var_type=var_type)


def rk2(f=None, show_code=None, dt=None, beta=None, var_type=None):
  """Runge–Kutta methods for ordinary differential equations.

  Generic second-order method.

  It has the characteristics of:

      - method stage = 2
      - method order = 2
      - Butcher Tables:

  .. backend::

      \\begin{array}{c|cc}
          0 & 0 & 0 \\\\
          \\beta & \\beta & 0 \\\\
          \\hline & 1 - {1 \\over 2 * \\beta} & {1 \over 2 * \\beta}
      \\end{array}
  """
  beta = 2 / 3 if beta is None else beta
  dt = backend.get_dt() if dt is None else dt
  show_code = False if show_code is None else show_code
  var_type = constants.POPU_VAR if var_type is None else var_type

  if f is None:
    return lambda f: rk2_wrapper(f, show_code=show_code, dt=dt, beta=beta,
                                 var_type=var_type)
  else:
    return rk2_wrapper(f, show_code=show_code, dt=dt, beta=beta,
                       var_type=var_type)


def rk3(f=None, show_code=None, dt=None, var_type=None):
  """Classical third-order Runge-Kutta method for ordinary differential equations.

  It has the characteristics of:

      - method stage = 3
      - method order = 3
      - Butcher Tables:

  .. backend::

      \\begin{array}{c|ccc}
          0 & 0 & 0 & 0 \\\\
          1 / 2 & 1 / 2 & 0 & 0 \\\\
          1 & -1 & 2 & 0 \\\\
          \\hline & 1 / 6 & 2 / 3 & 1 / 6
      \\end{array}

  """
  A = [(), (0.5,), (-1, 2)]
  B = ['1/6', '2/3', '1/6']
  C = [0, 0.5, 1]
  return _base(A=A,
               B=B,
               C=C,
               f=f,
               show_code=show_code,
               dt=dt,
               var_type=var_type)


def heun3(f=None, show_code=None, dt=None, var_type=None):
  """Heun's third-order method for ordinary differential equations.

  It has the characteristics of:

      - method stage = 3
      - method order = 3
      - Butcher Tables:

  .. backend::

      \\begin{array}{c|ccc}
          0 & 0 & 0 & 0 \\\\
          1 / 3 & 1 / 3 & 0 & 0 \\\\
          2 / 3 & 0 & 2 / 3 & 0 \\\\
          \\hline & 1 / 4 & 0 & 3 / 4
      \\end{array}

  """
  A = [(), ('1/3',), (0, '2/3')]
  B = [0.25, 0, 0.75]
  C = [0, '1/3', '2/3']
  return _base(A=A,
               B=B,
               C=C,
               f=f,
               show_code=show_code,
               dt=dt,
               var_type=var_type)


def ralston3(f=None, show_code=None, dt=None, var_type=None):
  """Ralston's third-order method for ordinary differential equations.

  It has the characteristics of:

      - method stage = 3
      - method order = 3
      - Butcher Tables:

  .. backend::

      \\begin{array}{c|ccc}
          0 & 0 & 0 & 0 \\\\
          1 / 2 & 1 / 2 & 0 & 0 \\\\
          3 / 4 & 0 & 3 / 4 & 0 \\\\
          \\hline & 2 / 9 & 1 / 3 & 4 / 9
      \\end{array}

  References
  ----------

  .. [1] Ralston, Anthony (1962). "Runge-Kutta Methods with Minimum Error Bounds".
      Math. Comput. 16 (80): 431–437. doi:10.1090/S0025-5718-1962-0150954-0

  """
  A = [(), (0.5,), (0, 0.75)]
  B = ['2/9', '1/3', '4/9']
  C = [0, 0.5, 0.75]
  return _base(A=A,
               B=B,
               C=C,
               f=f,
               show_code=show_code,
               dt=dt,
               var_type=var_type)


def ssprk3(f=None, show_code=None, dt=None, var_type=None):
  """Third-order Strong Stability Preserving Runge-Kutta (SSPRK3).

  It has the characteristics of:

      - method stage = 3
      - method order = 3
      - Butcher Tables:

  .. backend::

      \\begin{array}{c|ccc}
          0 & 0 & 0 & 0 \\\\
          1 & 1 & 0 & 0 \\\\
          1 / 2 & 1 / 4 & 1 / 4 & 0 \\\\
          \\hline & 1 / 6 & 1 / 6 & 2 / 3
      \\end{array}

  """
  A = [(), (1,), (0.25, 0.25)]
  B = ['1/6', '1/6', '2/3']
  C = [0, 1, 0.5]
  return _base(A=A,
               B=B,
               C=C,
               f=f,
               show_code=show_code,
               dt=dt,
               var_type=var_type)


def rk4(f=None, show_code=None, dt=None, var_type=None):
  """Classical fourth-order Runge-Kutta method for ordinary differential equations.

  It has the characteristics of:

      - method stage = 4
      - method order = 4
      - Butcher Tables:

  .. backend::

      \\begin{array}{c|cccc}
          0 & 0 & 0 & 0 & 0 \\\\
          1 / 2 & 1 / 2 & 0 & 0 & 0 \\\\
          1 / 2 & 0 & 1 / 2 & 0 & 0 \\\\
          1 & 0 & 0 & 1 & 0 \\\\
          \\hline & 1 / 6 & 1 / 3 & 1 / 3 & 1 / 6
      \\end{array}

  """

  A = [(), (0.5,), (0., 0.5), (0., 0., 1)]
  B = ['1/6', '1/3', '1/3', '1/6']
  C = [0, 0.5, 0.5, 1]
  return _base(A=A,
               B=B,
               C=C,
               f=f,
               show_code=show_code,
               dt=dt,
               var_type=var_type)


def ralston4(f=None, show_code=None, dt=None, var_type=None):
  """Ralston's fourth-order method for ordinary differential equations.

  It has the characteristics of:

      - method stage = 4
      - method order = 4
      - Butcher Tables:

  .. backend::

      \\begin{array}{c|cccc}
          0 & 0 & 0 & 0 & 0 \\\\
          .4 & .4 & 0 & 0 & 0 \\\\
          .45573725 & .29697761 & .15875964 & 0 & 0 \\\\
          1 & .21810040 & -3.05096516 & 3.83286476 & 0 \\\\
          \\hline & .17476028 & -.55148066 & 1.20553560 & .17118478
      \\end{array}

  References
  ----------

  [1] Ralston, Anthony (1962). "Runge-Kutta Methods with Minimum Error Bounds".
      Math. Comput. 16 (80): 431–437. doi:10.1090/S0025-5718-1962-0150954-0

  """
  A = [(), (.4,), (.29697761, .15875964), (.21810040, -3.05096516, 3.83286476)]
  B = [.17476028, -.55148066, 1.20553560, .17118478]
  C = [0, .4, .45573725, 1]
  return _base(A=A,
               B=B,
               C=C,
               f=f,
               show_code=show_code,
               dt=dt,
               var_type=var_type)


def rk4_38rule(f=None, show_code=None, dt=None, var_type=None):
  """3/8-rule fourth-order method for ordinary differential equations.

  This method doesn't have as much notoriety as the "classical" method,
  but is just as classical because it was proposed in the same paper
  (Kutta, 1901).

  It has the characteristics of:

      - method stage = 4
      - method order = 4
      - Butcher Tables:

  .. backend::

      \\begin{array}{c|cccc}
          0 & 0 & 0 & 0 & 0 \\\\
          1 / 3 & 1 / 3 & 0 & 0 & 0 \\\\
          2 / 3 & -1 / 3 & 1 & 0 & 0 \\\\
          1 & 1 & -1 & 1 & 0 \\\\
          \\hline & 1 / 8 & 3 / 8 & 3 / 8 & 1 / 8
      \\end{array}

  """
  A = [(), ('1/3',), ('-1/3', '1'), (1, -1, 1)]
  B = ['1/8', '3/8', '3/8', '1/8']
  C = [0, '1/3', '2/3', 1]
  return _base(A=A,
               B=B,
               C=C,
               f=f,
               show_code=show_code,
               dt=dt,
               var_type=var_type)
