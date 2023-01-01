# -*- coding: utf-8 -*-

r"""This module provides explicit Runge-Kutta methods for ODEs.

Given an initial value problem specified as:

.. math::
    \frac{dy}{dt}=f(t,y),\quad y(t_{0})=y_{0}.

Let the step-size :math:`h > 0`.

Then, the general schema of explicit Runge–Kutta methods is [1]_:

.. math::
    y_{n+1}=y_{n}+h\sum _{i=1}^{s}b_{i}k_{i},

where

.. math::
    \begin{aligned}
    k_{1}&=f(t_{n},y_{n}),\\
    k_{2}&=f(t_{n}+c_{2}h,y_{n}+h(a_{21}k_{1})),\\
    k_{3}&=f(t_{n}+c_{3}h,y_{n}+h(a_{31}k_{1}+a_{32}k_{2})),\\
    &\\ \vdots \\
    k_{s}&=f(t_{n}+c_{s}h,y_{n}+h(a_{s1}k_{1}+a_{s2}k_{2}+\cdots +a_{s,s-1}k_{s-1})).
    \end{aligned}

To specify a particular method, one needs to provide the integer :math:`s` (the number
of stages), and the coefficients :math:`a_{ij}` (for :math:`1 \le j < i \le s`),
:math:`b_i` (for :math:`i = 1, 2, \cdots, s`) and :math:`c_i` (for :math:`i = 2, 3, \cdots, s`).

The matrix :math:`[a_{ij}]` is called the *Runge–Kutta matrix*, while the :math:`b_i`
and :math:`c_i` are known as the *weights* and the *nodes*. These data are usually
arranged in a mnemonic device, known as a **Butcher tableau** (named after John C. Butcher):

.. math::
    \begin{array}{c|llll}
    0 & & & & & \\
    c_{2} & a_{21} & & & & \\
    c_{3} & a_{31} & a_{32} & & & \\
    \vdots & \vdots & & \ddots & \\
    c_{s} & a_{s 1} & a_{s 2} & \cdots & a_{s, s-1} \\
    \hline & b_{1} & b_{2} & \cdots & b_{s-1} & b_{s}
    \end{array}

A Taylor series expansion shows that the Runge–Kutta method is consistent if and only if

.. math::
    \sum _{i=1}^{s}b_{i}=1.

Another popular condition for determining coefficients is:

.. math::
    \sum_{j=1}^{i-1}a_{ij}=c_{i}{\text{ for }}i=2,\ldots ,s.


More details please see references [2]_ [3]_ [4]_.

.. [1] Press, W. H., B. P. Flannery, S. A. Teukolsky, and W. T.
       Vetterling. "Section 17.1 Runge-Kutta Method." Numerical Recipes:
       The Art of Scientific Computing (2007).
.. [2] https://en.wikipedia.org/wiki/Runge%E2%80%93Kutta_methods
.. [3] Butcher, John Charles. Numerical methods for ordinary differential equations.
       John Wiley & Sons, 2016.
.. [4] Iserles, A., 2009. A first course in the numerical analysis of differential
       equations (No. 44). Cambridge university press.

"""

from brainpy._src.integrators import constants as C, utils
from brainpy._src.integrators.ode import common
from brainpy._src.integrators.ode.base import ODEIntegrator
from .generic import register_ode_integrator

__all__ = [
  'ExplicitRKIntegrator',
  'Euler',
  'MidPoint',
  'Heun2',
  'Ralston2',
  'RK2',
  'RK3',
  'Heun3',
  'Ralston3',
  'SSPRK3',
  'RK4',
  'Ralston4',
  'RK4Rule38',
]


class ExplicitRKIntegrator(ODEIntegrator):
  r"""Explicit Runge–Kutta methods for ordinary differential equation.

  For the system,

  .. math::

      \frac{d y}{d t}=f(t, y)


  Explicit Runge-Kutta methods take the form

  .. math::

      k_{i}=f\left(t_{n}+c_{i}h,y_{n}+h\sum _{j=1}^{s}a_{ij}k_{j}\right) \\
      y_{n+1}=y_{n}+h \sum_{i=1}^{s} b_{i} k_{i}

  Each method listed on this page is defined by its Butcher tableau,
  which puts the coefficients of the method in a table as follows:

  .. math::

      \begin{array}{c|cccc}
          c_{1} & a_{11} & a_{12} & \ldots & a_{1 s} \\
          c_{2} & a_{21} & a_{22} & \ldots & a_{2 s} \\
          \vdots & \vdots & \vdots & \ddots & \vdots \\
          c_{s} & a_{s 1} & a_{s 2} & \ldots & a_{s s} \\
          \hline & b_{1} & b_{2} & \ldots & b_{s}
      \end{array}

  Parameters
  ----------
  f : callable
      The derivative function.
  show_code : bool
      Whether show the formatted code.
  dt : float
      The numerical precision.
  """
  A = []  # The A matrix in the Butcher tableau.
  B = []  # The B vector in the Butcher tableau.
  C = []  # The C vector in the Butcher tableau.

  def __init__(self,
               f,
               var_type=None,
               dt=None,
               name=None,
               show_code=False,
               state_delays=None,
               neutral_delays=None):
    super(ExplicitRKIntegrator, self).__init__(f=f,
                                               var_type=var_type,
                                               dt=dt,
                                               name=name,
                                               show_code=show_code,
                                               state_delays=state_delays,
                                               neutral_delays=neutral_delays)

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
    utils.check_kws(self.arg_names, keywords)
    self.build()

  def build(self):
    # step stage
    common.step(self.variables, C.DT,
                self.A, self.C, self.code_lines, self.parameters)
    # variable update
    return_args = common.update(self.variables, C.DT, self.B, self.code_lines)
    # returns
    self.code_lines.append(f'  return {", ".join(return_args)}')
    # compile
    self.integral = utils.compile_code(
      code_scope={k: v for k, v in self.code_scope.items()},
      code_lines=self.code_lines,
      show_code=self.show_code,
      func_name=self.func_name)


class Euler(ExplicitRKIntegrator):
  r"""The Euler method for ODEs.

  Also named as `Forward Euler method`, or `Explicit Euler` method.

  Given an ODE system,

    .. math::

        y'(t)=f(t,y(t)),\qquad y(t_{0})=y_{0},

  by using Euler method [1]_, we should choose a value :math:`h` for the
  size of every step and set :math:`t_{n}=t_{0}+nh`. Now, one step
  of the Euler method from :math:`t_{n}` to :math:`t_{n+1}=t_{n}+h` is:

  .. math::

      y_{n+1}=y_{n}+hf(t_{n},y_{n}).

  Note that the method increments a solution through an interval :math:`h`
  while using derivative information from only the beginning of the interval.
  As a result, the step's error is :math:`O(h^2)`.

  **Geometric interpretation**

  Illustration of the Euler method. The unknown curve is in blue,
  and its polygonal approximation is in red [2]_:

  .. image:: ../../../../_static/ode_Euler_method.svg
     :align: center

  **Derivation**

  There are several ways to get Euler method [2]_.

  The first is to consider the Taylor expansion of the function :math:`y`
  around :math:`t_{0}`:

  .. math::

      y(t_{0}+h)=y(t_{0})+hy'(t_{0})+{\frac {1}{2}}h^{2}y''(t_{0})+O(h^{3}).

  where :math:`y'(t_0)=f(t_0,y)`. We ignore the quadratic and higher-order
  terms, then we get Euler method. The Taylor expansion is used below to
  analyze the error committed by the Euler method, and it can be extended
  to produce Runge–Kutta methods.

  The second way is to replace the derivative with the forward finite
  difference formula:

  .. math::

      y'(t_{0})\approx {\frac {y(t_{0}+h)-y(t_{0})}{h}}.

  The third method is integrate the differential equation from :math:`t_{0}`
  to :math:`t_{0}+h` and apply the fundamental theorem of calculus to get:

  .. math::

      y(t_{0}+h)-y(t_{0})=\int _{t_{0}}^{t_{0}+h}f(t,y(t))\,\mathrm {d} t \approx hf(t_{0},y(t_{0})).

  **Note**

  Euler method is a first order numerical procedure for solving
  ODEs with a given initial value. The lack of stability
  and accuracy limits its popularity mainly to use as a
  simple introductory example of a numeric solution method.

  References
  ----------
  .. [1] W. H.; Flannery, B. P.; Teukolsky, S. A.; and Vetterling,
         W. T. Numerical Recipes in FORTRAN: The Art of Scientific
         Computing, 2nd ed. Cambridge, England: Cambridge University
         Press, p. 710, 1992.
  .. [2] https://en.wikipedia.org/wiki/Euler_method
  """
  A = [(), ]
  B = [1]
  C = [0]


register_ode_integrator('euler', Euler)


class MidPoint(ExplicitRKIntegrator):
  r"""Explicit midpoint method for ODEs.

  Also known as the `modified Euler method` [1]_.

  The midpoint method is a one-step method for numerically solving
  the differential equation given by:

  .. math::

       y'(t) = f(t, y(t)), \quad y(t_0) = y_0 .

  The formula of the explicit midpoint method is:

  .. math::

       y_{n+1} = y_n + hf\left(t_n+\frac{h}{2},y_n+\frac{h}{2}f(t_n, y_n)\right).

  Therefore, the Butcher tableau of the midpoint method is:

  .. math::

      \begin{array}{c|cc}
          0 & 0 & 0 \\
          1 / 2 & 1 / 2 & 0 \\
          \hline & 0 & 1
      \end{array}


  **Derivation**

  Compared to the slope formula of Euler method :math:`y'(t) \approx \frac{y(t+h) - y(t)}{h}`,
  the midpoint method use

  .. math::

       y'\left(t+\frac{h}{2}\right) \approx \frac{y(t+h) - y(t)}{h},

  The reason why we use this, please see the following geometric interpretation.
  Then, we get

  .. math::

       y(t+h) \approx y(t) + hf\left(t+\frac{h}{2},y\left(t+\frac{h}{2}\right)\right).

  However, we do not know :math:`y(t+h/2)`. The solution is then to use a Taylor
  series expansion exactly as the Euler method to solve:

  .. math::

      y\left(t + \frac{h}{2}\right) \approx y(t) + \frac{h}{2}y'(t)=y(t) + \frac{h}{2}f(t, y(t)),

  Finally, we can get the final step function:

  .. math::

      y(t + h) \approx y(t) + hf\left(t + \frac{h}{2}, y(t) + \frac{h}{2}f(t, y(t))\right).

  **Geometric interpretation**

  In the basic Euler's method, the tangent of the curve at :math:`(t_{n},y_{n})` is computed
  using :math:`f(t_{n},y_{n})`. The next value :math:`y_{n+1}` is found where the tangent
  intersects the vertical line :math:`t=t_{n+1}`. However, if the second derivative is only
  positive between :math:`t_{n}` and :math:`t_{n+1}`, or only negative, the curve will
  increasingly veer away from the tangent, leading to larger errors as :math:`h` increases.

  Compared with the Euler method, midpoint method use the tangent at the midpoint (upper, green
  line segment in the following figure [2]_), which would most likely give a more accurate
  approximation of the curve in that interval.

  .. image:: ../../../../_static/ode_Midpoint_method_illustration.png
     :align: center

  Although this midpoint tangent could not be accurately calculated, we can estimate midpoint
  value of :math:`y(t)` by using the original Euler's method. Finally, the improved tangent
  is used to calculate the value of :math:`y_{n+1}` from :math:`y_{n}`. This last step is
  represented by the red chord in the diagram.

  **Note**

  Note that the red chord is not exactly parallel to the green segment (the true tangent),
  due to the error in estimating the value of :math:`y(t)` at the midpoint.

  References
  ----------

  .. [1] Süli, Endre, and David F. Mayers. An Introduction to Numerical Analysis. no. 1, 2003.
  .. [2] https://en.wikipedia.org/wiki/Midpoint_method
  """
  A = [(), (0.5,)]
  B = [0, 1]
  C = [0, 0.5]


register_ode_integrator('midpoint', MidPoint)


class Heun2(ExplicitRKIntegrator):
  r"""Heun's method for ODEs.

  This method is named after Karl Heun [1]_. It is also known as
  the `explicit trapezoid rule`, `improved Euler's method`, or `modified Euler's method`.

  Given ODEs with a given initial value,

  .. math::
      y'(t) = f(t,y(t)), \qquad y(t_0)=y_0,

  the two-stage Heun's method is formulated as:

  .. math::
      \tilde{y}_{n+1} = y_n + h f(t_n,y_n)

  .. math::
      y_{n+1} = y_n + \frac{h}{2}[f(t_n, y_n) + f(t_{n+1},\tilde{y}_{n+1})],

  where :math:`h` is the step size and :math:`t_{n+1}=t_n+h`.

  Therefore, the Butcher tableau of the two-stage Heun's method is:

  .. math::
      \begin{array}{c|cc}
          0.0 & 0.0 & 0.0 \\
          1.0 & 1.0 & 0.0 \\
          \hline & 0.5 & 0.5
      \end{array}


  **Geometric interpretation**

  In the :py:func:`brainpy.integrators.ode.midpoint`, we have already known Euler
  method has big estimation error because it uses the
  line tangent to the function at the beginning of the interval :math:`t_n` as an
  estimate of the slope of the function over the interval :math:`(t_n, t_{n+1})`.

  In order to address this problem, Heun's Method considers the tangent lines to
  the solution curve at both ends of the interval (:math:`t_n` and :math:`t_{n+1}`),
  one (:math:`f(t_n, y_n)`) which *underestimates*, and one
  (:math:`f(t_{n+1},\tilde{y}_{n+1})`, approximated using Euler's Method) which
  *overestimates* the ideal vertical coordinates. The ideal point lies approximately
  halfway between the erroneous overestimation and underestimation, the average of the two slopes.

  .. image:: ../../../../_static/ode_Heun2_Method_Diagram.jpg
     :align: center

  .. math::
      \begin{aligned}
      {\text{Slope}}_{\text{left}}=&f(t_{n},y_{n}) \\
      {\text{Slope}}_{\text{right}}=&f(t_{n}+h,y_{n}+hf(t_{n},y_{n})) \\
      {\text{Slope}}_{\text{ideal}}=&{\frac {1}{2}}({\text{Slope}}_{\text{left}}+{\text{Slope}}_{\text{right}})
      \end{aligned}

  References
  ----------

  .. [1] Süli, Endre, and David F. Mayers. An Introduction to Numerical Analysis. no. 1, 2003.
  """
  A = [(), (1,)]
  B = [0.5, 0.5]
  C = [0, 1]


register_ode_integrator('heun2', Heun2)


class Ralston2(ExplicitRKIntegrator):
  r"""Ralston's method for ODEs.

  Ralston's method is a second-order method with two stages and
  a minimum local error bound.

  Given ODEs with a given initial value,

  .. math::
      y'(t) = f(t,y(t)), \qquad y(t_0)=y_0,

  the Ralston's second order method is given by

  .. math::
      y_{n+1}=y_{n}+\frac{h}{4} f\left(t_{n}, y_{n}\right)+
      \frac{3 h}{4} f\left(t_{n}+\frac{2 h}{3}, y_{n}+\frac{2 h}{3} f\left(t_{n}, y_{n}\right)\right)

  Therefore, the corresponding Butcher tableau is:

  .. math::
      \begin{array}{c|cc}
          0 & 0 & 0 \\
          2 / 3 & 2 / 3 & 0 \\
          \hline & 1 / 4 & 3 / 4
      \end{array}
  """
  A = [(), ('2/3',)]
  B = [0.25, 0.75]
  C = [0, '2/3']


register_ode_integrator('ralston2', Ralston2)


class RK2(ExplicitRKIntegrator):
  r"""Generic second order Runge-Kutta method for ODEs.

  **Derivation**

  In the :py:func:`brainpy.integrators.ode.midpoint`,
  :py:func:`brainpy.integrators.ode.heun2`, and :py:func:`brainpy.integrators.ode.ralston2`,
  we have already known first-order Euler method :py:func:`brainpy.integrators.ode.euler`
  has big estimation error.

  Here, we seek to derive a generic second order Runge-Kutta method [1]_ for the
  given ODE system with a given initial value,

  .. math::
      y'(t) = f(t,y(t)), \qquad y(t_0)=y_0,

  we want to get a generic solution:

  .. math::
      \begin{align} y_{n+1} &= y_{n} + h \left ( a_1 K_1 + a_2 K_2 \right ) \tag{1}
      \end{align}

  where :math:`a_1` and :math:`a_2` are some weights to be determined,
  and :math:`K_1` and :math:`K_2` are derivatives on the form:

  .. math::
      \begin{align}
      K_1 & = f(t_n,y_n) \qquad \text{and} \qquad K_2 = f(t_n + p_1 h,y_n + p_2 K_1 h ) \tag{2}
      \end{align}

  By substitution of (2) in (1) we get:

  .. math::
      \begin{align}
      y_{n+1} &= y_{n} + a_1 h f(t_n,y_n) + a_2 h f(t_n + p_1 h,y_n + p_2 K_1 h) \tag{3}
      \end{align}

  Now, we may find a Taylor-expansion of :math:`f(t_n + p_1 h, y_n + p_2 K_1 h )`

  .. math::
      \begin{align}
      f(t_n + p_1 h, y_n + p_2 K_1 h ) &= f + p_1 h f_t + p_2 K_1 h f_y  + \text{h.o.t.} \nonumber \\
        & = f + p_1 h f_t + p_2 h f f_y  + \text{h.o.t.}   \tag{4}
      \end{align}

  where :math:`f_t \equiv \frac{\partial f}{\partial t}` and
  :math:`f_y \equiv \frac{\partial f}{\partial y}`.

  By substitution of (4) in (3) we eliminate the implicit dependency of :math:`y_{n+1}`

  .. math::
      \begin{align}
      y_{n+1} &= y_{n} +  a_1 h f(t_n,y_n) + a_2 h \left (f + p_1 h f_t + p_2 h f f_y \right ) \nonumber \\
              &= y_{n} + (a_1 + a_2) h f + \left (a_2 p_1 f_t + a_2 p_2 f f_y \right) h^2     \tag{5}
      \end{align}

  In the next, we try to get the second order Taylor expansion of the solution:

  .. math::
      \begin{align}
       y(t_n+h) = y_n + h y' + \frac{h^2}{2} y''  + O(h^3) \tag{6}
      \end{align}

  where the second order derivative is given by

  .. math::
      \begin{align}
       y'' = \frac{d^2 y}{dt^2} = \frac{df}{dt} = \frac{\partial{f}}{\partial{t}}
       \frac{dt}{dt} + \frac{\partial{f}}{\partial{y}}  \frac{dy}{dt} = f_t + f f_y  \tag{7}
      \end{align}

  Substitution of (7) into (6) yields:

  .. math::
      \begin{align}
       y(t_n+h) = y_n + h f + \frac{h^2}{2}  \left (f_t + f f_y \right )  + O(h^3) \tag{8}
      \end{align}

  Finally, in order to approximate (8) by using (5), we get the generic second order
  Runge-Kutta method, where

  .. math::
      \begin{aligned}
      a_1 + a_2 = 1  \\
      a_2 p_1 = \frac{1}{2} \\
      a_2 p_2 = \frac{1}{2}.
      \end{aligned}

  Furthermore, let :math:`p_1=\beta`, we get

  .. math::
      \begin{aligned}
      p_1 = & \beta \\
      p_2 = & \beta \\
      a_2 = &\frac{1}{2\beta} \\
      a_1 = &1 - \frac{1}{2\beta} .
      \end{aligned}

  Therefore, the corresponding Butcher tableau is:

  .. math::

      \begin{array}{c|cc}
          0 & 0 & 0 \\
          \beta & \beta & 0 \\
          \hline & 1 - {1 \over 2 * \beta} & {1 \over 2 * \beta}
      \end{array}

  References
  ----------

  .. [1] Chapra, Steven C., and Raymond P. Canale. Numerical methods
         for engineers. Vol. 1221. New York: Mcgraw-hill, 2011.

  """

  def __init__(self,
               f,
               beta=2 / 3,
               var_type=None,
               dt=None,
               name=None,
               show_code=False,
               state_delays=None,
               neutral_delays=None):
    self.A = [(), (beta,)]
    self.B = [1 - 1 / (2 * beta), 1 / (2 * beta)]
    self.C = [0, beta]
    super(RK2, self).__init__(f=f,
                              var_type=var_type,
                              dt=dt,
                              name=name,
                              show_code=show_code,
                              state_delays=state_delays,
                              neutral_delays=neutral_delays)


register_ode_integrator('rk2', RK2)


class RK3(ExplicitRKIntegrator):
  r"""Classical third-order Runge-Kutta method for ODEs.

  For the given initial value problem :math:`y'(x) = f(t,y);\, y(t_0) = y_0`,
  the third order Runge-Kutta method is given by:

  .. math::
      y_{n+1} = y_n + 1/6 ( k_1 + 4 k_2 + k_3),

  where

  .. math::
      k_1 = h f(t_n, y_n), \\
      k_2 = h f(t_n + h / 2, y_n + k_1 / 2), \\
      k_3 = h f(t_n + h, y_n - k_1 + 2 k_2 ),

  where :math:`t_n = t_0 + n h.`

  Error term :math:`O(h^4)`,  correct up to the third order term in Taylor series expansion.

  The Taylor series expansion is :math:`y(t+h)=y(t)+\frac{k}{6}+\frac{2 k_{2}}{3}+\frac{k_{3}}{6}+O\left(h^{4}\right)`.

  The corresponding Butcher tableau is:
  
  .. math::
      \begin{array}{c|ccc}
          0 & 0 & 0 & 0 \\
          1 / 2 & 1 / 2 & 0 & 0 \\
          1 & -1 & 2 & 0 \\
          \hline & 1 / 6 & 2 / 3 & 1 / 6
      \end{array}

  """
  A = [(), (0.5,), (-1, 2)]
  B = ['1/6', '2/3', '1/6']
  C = [0, 0.5, 1]


register_ode_integrator('rk3', RK3)


class Heun3(ExplicitRKIntegrator):
  r"""Heun's third-order method for ODEs.

  It has the characteristics of:

      - method stage = 3
      - method order = 3
      - Butcher Tables:

  .. math::

      \begin{array}{c|ccc}
          0 & 0 & 0 & 0 \\
          1 / 3 & 1 / 3 & 0 & 0 \\
          2 / 3 & 0 & 2 / 3 & 0 \\
          \hline & 1 / 4 & 0 & 3 / 4
      \end{array}

  """
  A = [(), ('1/3',), (0, '2/3')]
  B = [0.25, 0, 0.75]
  C = [0, '1/3', '2/3']


register_ode_integrator('heun3', Heun3)


class Ralston3(ExplicitRKIntegrator):
  r"""Ralston's third-order method for ODEs.

  It has the characteristics of:

      - method stage = 3
      - method order = 3
      - Butcher Tables:

  .. math::
      \begin{array}{c|ccc}
          0 & 0 & 0 & 0 \\
          1 / 2 & 1 / 2 & 0 & 0 \\
          3 / 4 & 0 & 3 / 4 & 0 \\
          \hline & 2 / 9 & 1 / 3 & 4 / 9
      \end{array}

  References
  ----------

  .. [1] Ralston, Anthony (1962). "Runge-Kutta Methods with Minimum Error Bounds".
         Math. Comput. 16 (80): 431–437. doi:10.1090/S0025-5718-1962-0150954-0

  """
  A = [(), (0.5,), (0, 0.75)]
  B = ['2/9', '1/3', '4/9']
  C = [0, 0.5, 0.75]


register_ode_integrator('ralston3', Ralston3)


class SSPRK3(ExplicitRKIntegrator):
  r"""Third-order Strong Stability Preserving Runge-Kutta (SSPRK3).

  It has the characteristics of:

      - method stage = 3
      - method order = 3
      - Butcher Tables:

  .. math::
      \begin{array}{c|ccc}
          0 & 0 & 0 & 0 \\
          1 & 1 & 0 & 0 \\
          1 / 2 & 1 / 4 & 1 / 4 & 0 \\
          \hline & 1 / 6 & 1 / 6 & 2 / 3
      \end{array}

  """
  A = [(), (1,), (0.25, 0.25)]
  B = ['1/6', '1/6', '2/3']
  C = [0, 1, 0.5]


register_ode_integrator('ssprk3', SSPRK3)


class RK4(ExplicitRKIntegrator):
  r"""Classical fourth-order Runge-Kutta method for ODEs.

  For the given initial value problem of

  .. math::
      {\frac {dy}{dt}}=f(t,y),\quad y(t_{0})=y_{0}.

  The fourth-order RK method is formulated as:

  .. math::
      \begin{aligned}
      y_{n+1}&=y_{n}+{\frac {1}{6}}h\left(k_{1}+2k_{2}+2k_{3}+k_{4}\right),\\
      t_{n+1}&=t_{n}+h\\
      \end{aligned}

  for :math:`n = 0, 1, 2, 3, \cdot`, using

  .. math::
      \begin{aligned}
      k_{1}&=\ f(t_{n},y_{n}),\\
      k_{2}&=\ f\left(t_{n}+{\frac {h}{2}},y_{n}+h{\frac {k_{1}}{2}}\right),\\
      k_{3}&=\ f\left(t_{n}+{\frac {h}{2}},y_{n}+h{\frac {k_{2}}{2}}\right),\\
      k_{4}&=\ f\left(t_{n}+h,y_{n}+hk_{3}\right).
      \end{aligned}

  Here :math:`y_{n+1}` is the RK4 approximation of :math:`y(t_{n+1})`, and the next
  value (:math:`y_{n+1}`) is determined by the present value (:math:`y_{n}`) plus
  the weighted average of four increments, where each increment is the product
  of the size of the interval, :math:`h`, and an estimated slope specified by function
  :math:`f` on the right-hand side of the differential equation.

  - :math:`k_{1}` is the slope at the beginning of the interval, using :math:`y` (Euler's method);
  - :math:`k_{2}` is the slope at the midpoint of the interval, using :math:`y` and :math:`k_{1}`;
  - :math:`k_{3}` is again the slope at the midpoint, but now using :math:`y` and :math:`k_{2}`;
  - :math:`k_{4}` is the slope at the end of the interval, using :math:`y` and :math:`k_{3}`.

  The RK4 method is a fourth-order method, meaning that the local truncation error is on the order
  of (:math:`O(h^{5}`), while the total accumulated error is on the order of (:math:`O(h^{4}`).

  The corresponding Butcher tableau is:

  .. math::
      \begin{array}{c|cccc}
          0 & 0 & 0 & 0 & 0 \\
          1 / 2 & 1 / 2 & 0 & 0 & 0 \\
          1 / 2 & 0 & 1 / 2 & 0 & 0 \\
          1 & 0 & 0 & 1 & 0 \\
          \hline & 1 / 6 & 1 / 3 & 1 / 3 & 1 / 6
      \end{array}

  References
  ----------
  .. [1] Lambert, J. D. and Lambert, D. Ch. 5 in Numerical Methods for Ordinary
         Differential Systems: The Initial Value Problem. New York: Wiley, 1991.
  .. [2] Press, W. H.; Flannery, B. P.; Teukolsky, S. A.; and Vetterling, W. T.
         "Runge-Kutta Method" and "Adaptive Step Size Control for Runge-Kutta."
         §16.1 and 16.2 in Numerical Recipes in FORTRAN: The Art of Scientific
         Computing, 2nd ed. Cambridge, England: Cambridge University Press,
         pp. 704-716, 1992.
  """

  A = [(), (0.5,), (0., 0.5), (0., 0., 1)]
  B = ['1/6', '1/3', '1/3', '1/6']
  C = [0, 0.5, 0.5, 1]


register_ode_integrator('rk4', RK4)


class Ralston4(ExplicitRKIntegrator):
  r"""Ralston's fourth-order method for ODEs.

  It has the characteristics of:

      - method stage = 4
      - method order = 4
      - Butcher Tables:

  .. math::

      \begin{array}{c|cccc}
          0 & 0 & 0 & 0 & 0 \\
          .4 & .4 & 0 & 0 & 0 \\
          .45573725 & .29697761 & .15875964 & 0 & 0 \\
          1 & .21810040 & -3.05096516 & 3.83286476 & 0 \\
          \hline & .17476028 & -.55148066 & 1.20553560 & .17118478
      \end{array}

  References
  ----------

  .. [1] Ralston, Anthony (1962). "Runge-Kutta Methods with Minimum Error Bounds".
         Math. Comput. 16 (80): 431–437. doi:10.1090/S0025-5718-1962-0150954-0

  """
  A = [(), (.4,), (.29697761, .15875964), (.21810040, -3.05096516, 3.83286476)]
  B = [.17476028, -.55148066, 1.20553560, .17118478]
  C = [0, .4, .45573725, 1]


register_ode_integrator('ralston4', Ralston4)


class RK4Rule38(ExplicitRKIntegrator):
  r"""3/8-rule fourth-order method for ODEs.

  A slight variation of "the" Runge–Kutta method is also due
  to Kutta in 1901 [1]_ and is called the 3/8-rule. The primary
  advantage this method has is that almost all of the error
  coefficients are smaller than in the popular method, but it
  requires slightly more FLOPs (floating-point operations) per
  time step.


  It has the characteristics of:

      - method stage = 4
      - method order = 4
      - Butcher Tables:

  .. math::

      \begin{array}{c|cccc}
          0 & 0 & 0 & 0 & 0 \\
          1 / 3 & 1 / 3 & 0 & 0 & 0 \\
          2 / 3 & -1 / 3 & 1 & 0 & 0 \\
          1 & 1 & -1 & 1 & 0 \\
          \hline & 1 / 8 & 3 / 8 & 3 / 8 & 1 / 8
      \end{array}


  References
  ----------

  .. [1] Hairer, Ernst; Nørsett, Syvert Paul; Wanner, Gerhard (1993),
         Solving ordinary differential equations I: Nonstiff problems,
         Berlin, New York: Springer-Verlag, ISBN 978-3-540-56670-0.

  """
  A = [(), ('1/3',), ('-1/3', '1'), (1, -1, 1)]
  B = [0.125, 0.375, 0.375, 0.125]
  C = [0, '1/3', '2/3', 1]


register_ode_integrator('rk4_38rule', RK4Rule38)
