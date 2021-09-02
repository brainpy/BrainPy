Error Analysis of Numerical Methods
-----------------------------------

In order to identify the essential properties of numerical methods, we define basic notions [1]_.

For the given ODE system

.. math::
    \frac{dy}{dt}=f(t,y),\quad y(t_{0})=y_{0},

we define :math:`y(t_n)` as the solution of IVP evaluated at
:math:`t=t_n`, and :math:`y_n` is a numerical approximation of :math:`y(t_n)` at the same
location by a **generic explicit numerical scheme** (no matter explicit, implicit or
multi-step scheme):

.. math::
    \begin{align}
    y_{n+1} = y_n + h \phi(t_n,y_n,h),  \tag{2}
    \end{align}

where :math:`h` is the discretization step for :math:`t`, i.e., :math:`h=t_{n+1}-t_n`,
and :math:`\phi(t_n,y_n,h)` is the increment function. We say that the defined numerical
scheme is consistent if :math:`\lim_{h\to0} \phi(t,y,h) = \phi(t,y,0) = f(t,y)`.

Then, the **approximation error** is defined as

.. math::
     e_n = y(t_n) - y_n.

The **absolute error** is defined as

.. math::
    |e_n| = |y(t_n) - y_n|.

The **relative error** is defined as

.. math::
    r_n =\frac{|y(t_n) - y_n|}{|y(t_n)|}.

The **exact differential operator** is defined as

.. math::
    \begin{align}
    L_e(y) = y' - f(t,y) = 0
    \end{align}

The **approximate differential operator** is defined as

.. math::
    \begin{align}
    L_a(y_n) = y(t_{n+1}) - [y_n + \phi(t_n,y_n,h)].
    \end{align}

Finally, the **local truncation error (LTE)** is defined as

.. math::
    \begin{align}
    \tau_n = \frac{1}{h} L_a(y(x_n)).
    \end{align}

In practice, the evaluation of the exact solution for different :math:`t` around :math:`t_n`
(required by :math:`L_a`) is performed using a Taylor series expansion.

Finally, we can state that a scheme is :math:`p`-th order accurate by examining its
LTE and observing its leading term

.. math::
    \begin{align}
    \tau_n = C h^p + H.O.T.,
    \end{align}

where :math:`C` is a constant, independent of :math:`h`, and :math:`H.O.T.` are the
higher order terms of the LTE.

**Example: LTE for Euler's scheme**

Consider the IVP defined by :math:`y' = \lambda y`, with initial condition :math:`y(0)=1`.

The approximation operator for Euler's scheme is

.. math::
    \begin{align}
    L^{euler}_a = y(t_{n+1}) - [y_n + h \lambda y_n],
    \end{align}

then the LTE can be computed by

.. math::
    \begin{align}
    \tau_n = & \frac{1}{h}\left\{ L_a(y(t_n))\right\} = \frac{1}{h}\left\{ y(t_{n+1}) - [y(t_n) + h \lambda y(t_n)]\right\}, \\
    = & \frac{1}{h}\left\{ y(t_n) + h y'(t_n) + \frac{h^2}{2} y''(t_n) + \ldots + \frac{1}{p!} h^p y^{(p)}(t_n) - y(t_n) - h \lambda y(t_n) \right\} \\
    = & \frac{1}{2} h y''(t_n) + \ldots + \frac{1}{p!} h^{p-1} y^{(p)}(t_n) \\
    \approx & \frac{1}{2} h y''(t_n),
    \end{align}

where we assume :math:`y_n = y(t_n)`.



.. [1] https://folk.ntnu.no/leifh/teaching/tkt4140/._main022.html
