Numerical integrators
=====================

``NumpyBrain`` provides several explicit and implicit methods for 
the numerical integration of Ordinary Differential Equations (ODEs)
and Stochastic Differential Equations (SDEs). Here, we will present
the mathematical details of the implemented methods, and provide the 
examples on how to use them.

.. contents::
    :local:
    :depth: 2


Numerical integration of ordinary differential equations
--------------------------------------------------------


Forward Euler method
********************

The simplest way for solving ordinary differential equations is "the
Euler method" by Press et al. (1992) [1]_ :

.. math::

    y_{n+1} = y_n + f(y_n, t_n) \Delta t

This formula advances a solution from :math:`y_n` to :math:`y_{n+1}=y_n+h`.
Note that the method increments a solution through an interval :math:`h`
while using derivative information from only the beginning of the interval.
As a result, the step's error is :math:`O(h^2)`.


RK2: Second-order Runge-Kutta method
************************************

``NumpyBrain`` also provides the second-order explicit RK method.
It is given in parametric form by [2]_

.. math::

    k_1	&=	f(y_n, t_n)  \\
    k_2	&=	f(y_n + \beta \Delta t k_1, t_n + \beta \Delta t) \\
    y_{n+1} &= y_n + \Delta t [(1-\frac{1}{2\beta})k_1+\frac{1}{2\beta}k_2]


Classical choices for :math:`\beta` are ``1/2`` (explicit midpoint method),
``2/3`` (Ralston’s method), and ``1`` (Heun’s method, also known
as the ``explicit trapezoid rule``).


RK3: Third-order Runge-Kutta method
***********************************

Runge-Kutta’s third-order method is given by [3]_ [4]_ [5]_

.. math::

    k_1 &= f(y_n, t_n) \\
    k_2 &= f(y_n + \frac{\Delta t}{2}k_1, tn+\frac{\Delta t}{2}) \\
    k_3 &= f(y_n -\Delta t k_1 + 2\Delta t k_2, t_n + \Delta t) \\
    y_{n+1} &= y_{n} + \frac{\Delta t}{6}(k_1 + 4k_2+k_3)


RK4: Fourth-order Runge-kutta method
************************************

The classical fourth-order Runge–Kutta (RK4) [3]_ [4]_ [5]_, which is
the most popular of the RK methods, is

.. math::

    k_1 &= f(y_n, t_n) \\
    k_2 &= f(y_n + \frac{\Delta t}{2}k_1, t_n + \frac{\Delta t}{2}) \\
    k_3 &= f(y_n + \frac{\Delta t}{2}k_2, t_n + \frac{\Delta t}{2}) \\
    k_4 &= f(y_n + \Delta t k_3, t_n + \Delta t) \\
    y_{n+1} &= y_n + \frac{\Delta t}{6}(k_1 + 2*k_2 + 2* k_3 + k_4)



RK4 alternative ("3/8" rule)
****************************

There is an alternative for RK4 method. It is a less often used fourth-order
explicit RK method, and was also proposed by Kutta [6]_:

.. math::

    k_1 &= f(y_n, t_n) \\
    k_2 &= f(y_n + \frac{\Delta t}{3}k_1, t_n + \frac{\Delta t}{3}) \\
    k_3 &= f(y_n - \frac{\Delta t}{3}k_1 + \Delta t k_2, t_n + \frac{2 \Delta t}{3}) \\
    k_4 &= f(y_n + \Delta t k_1 - \Delta t k_2 + \Delta t k_3, t_n + \Delta t) \\
    y_{n+1} &= y_n + \frac{\Delta t}{8}(k_1 + 3*k_2 + 3* k_3 + k_4)


Explicit Euler method
*********************

The explicit midpoint method [7]_ is given by the formula

.. math::

    k1 &= f(y_n, t_n) \\
    k2 &= f(y_n + \frac{\Delta t}{2}k1, t_n + \frac{\Delta t}{2}) \\
    y_{n+1} &= y_n + \Delta t k_2

Or, in one line

.. math::

    y_{n+1} = y_n + \Delta t f(y_n + \frac{\Delta t}{2}f(y_n, t_n),
    t_n + \frac{\Delta t}{2})

The explicit midpoint method is also known as the modified Euler method.


Backward Euler method
*********************

The backward Euler method (or implicit Euler method) [8]_ [9]_
provide a different way for the
approximation of ordinary differential equations comparing with
the (standard) Euler method. The backward Euler method has error of
order ``1`` in time, it computes the approximations using

.. math::

    y_{n+1}=y_{n}+hf(t_{n+1},y_{n+1}).

This differs from the (forward) Euler method in that the latter
uses :math:`f(t_{n},y_{n})` in place of :math:`f(t_{n+1},y_{n+1})`.

**Solution**

The backward Euler method is an implicit method: the new approximation
:math:`y_{n+1}` appears on both sides of the equation, and thus the method
needs to solve an algebraic equation for the unknown :math:`y_{n+1}`.
For non-stiff problems, this can be done with fixed-point iteration:

.. math::

    y_{n+1}^{(0)} & =y_{n} \\
    y_{n+1}^{(i+1)} & =y_{n}+hf(t_{n+1},y_{n+1}^{(i)}).

If this sequence converges (within a given tolerance), then the method
takes its limit as the new approximation :math:`y_{n+1}`.

Alternatively, we can use the Newton–Raphson method to solve the
algebraic equation.

**Algorithmic summary of Backward Euler method**

For each timestep :math:`n`, do the following:

1, Initialize:

.. math::

    y_{n+1}^{(0)} &\leftarrow y_{n} \\
    i &\leftarrow 0

2, Update:

.. math::

    k &\leftarrow f(y_{n+1}^{(i)}, t_{n+1}) \\
    i &\leftarrow i + 1 \\
    y_{n+1}^{(i)} &= y_n + \Delta t k

3, If :math:`u^{(i)}_{n+1}` is “close” to :math:`u^{(i-1)}_{n+1}`,
the method has converged and the solution is complete. Jump to step 6.

4, If :math:`i = i_{max}`, the method did not converge. No solution
obtained; raise an exception.

5, Next iteration. Continue from step 2.

6, Set the solution obtained as

.. math::

    y_{n+1} \leftarrow y_{n+1}^{(i)}



Trapezoidal rule
****************

In numerical analysis and scientific computing,
the trapezoidal rule [10]_ is a numerical method to solve ordinary
differential equations derived from the trapezoidal rule for
computing integrals. The trapezoidal rule is an implicit
second-order method, which can be considered as both a
Runge–Kutta method and a linear multistep method.

The trapezoidal rule is given by the formula

.. math::

    y_{n+1}=y_{n}+\frac{1}{2}\Delta t {\Big (}f(t_{n},y_{n})+f(t_{n+1},y_{n+1}){\Big )}.

This is an implicit method: the value :math:`y_{n+1}` appears on both
sides of the equation, and to actually calculate it, we have to solve
an equation which will usually be nonlinear. One possible method for
solving this equation is Newton's method. We can use the Euler method
to get a fairly good estimate for the solution, which can be used as
the initial guess of Newton's method.



Implicit midpoint rule
**********************

The implicit midpoint method [11]_ is given by

.. math::

    y_{n+1} = y_n + \Delta t f(\frac{1}{2}(y_n + y_{n+1}),
    t_n + \frac{\Delta t}{2})

The implicit method is the most simple collocation method, and, applied to
Hamiltonian dynamics, a symplectic integrator.


Numerical integration of stochastic differential equations
----------------------------------------------------------

Before we diving into the mathematical basis of implemented algorithms,
let's distinguish the differences between two kinds of integrals of SDEs.

Itô and Stratonovich SDEs
*************************

One-dimensional stochastic differentiable equation (SDE) is given by [17]_ [18]_

.. math::

    \frac{dX_t}{dt} = f(X_t, t) dt + g(X_t, t)dW_t

where :math:`X_t = X(t)` is the realization of a stochastic processor
random variable. :math:`f(X_t,t)` is called the *drift* coefficient, that
is the deterministic part of the SDE characterizing the local trend.
:math:`g(X_t,t)` denotes the *diffusion* coefficient, that is the stochastic
part which influences the average size of the fluctuations of :math:`X`.
The fluctuations themselves originate from the stochastic process
:math:`W_t` called Wiener process.

Interpreted as an integral, we get

.. math::

    X_t = X_{t0} + \int_{t0}^{t}f(X_s, s) ds + \int_{t0}^{t}g(X_s, s) dW_s

where the first integral is an ordinary Riemann integral. As the sample paths
of a Wiener process are not differentiable, the Japanese mathematician K. Itô
defined in 1940s a new type of integral called **Itô stochastic integral**.

In 1960s, the Russian physicist R. L. Stratonovich proposed another kind of
stochastic integral called **Stratonovich stochastic integral**
and used the symbol “:math:`\circ`” to distinct it from the former
Itô integral [12]_ [17]_.

.. math::

    X_t = X_{t_0} + \int_{t0}^{t}f(X_s, s)ds + \int_{t0}^{t}g(X_s, s) \circ dW_s


The difference between two kinds of SDE is that Itô SDE evaluate the stochastic
integral at the left-point of the intervals [19]_, i.e.,

.. math::

    \int_{t0}^{t} g(X_s, s) dW_s = lim_{h\to 0} \sum_{k=0}^{m-1}
    g(X_{t_k}, t_k)[W(t_{k+1}) - W({t_k})]

while, Stratonovich SDE evaluate the integral at the mid-point of each intervals, i.e.,

.. math::

    \int_{t0}^{t} g(X_s, s) dW_s = lim_{h\to 0} \sum_{k=0}^{m-1}
    g(X_{(t_{k+1} - t_k)/2}, (t_{k+1} - t_k)/2)[W(t_{k+1}) - W({t_k})]

As a results, the Itô and Stratonovich representations do not converge towards
the same solution. Both integrals have their advantages and disadvantages and
which one should be used is more a modelling than mathematical issue. In financial
mathematics, the Itô interpretation is usually used since Itô calculus
only takes into account information about the past. The Stratonovich
interpretation is the most frequently used within the physical sciences [17]_.
More discussion please see ref [21]_.

In NumpyBrain, we implement the most widely-used numerical integration method for
SDEs, where :ref:`Euler-Maruyama method` and :ref:`Euler-Heun method` are explicit
order 0.5 strong Taylor scheme integrator, and :ref:`Milstein method` and
:ref:`Derivative-free Milstein method` are
integrators awith explicit order 1.0 strong Taylor scheme.

.. _Euler-Maruyama method:

Euler-Maruyama method
*********************

The simplest stochastic numerical approximation is the Euler-Maruyama
method that requires the problem to be described using the Itô scheme.

This approximation is a continuous time stochastic process that
satisfy the iterative scheme [20]_.

.. math::

    Y_{n+1} = Y_n + f(Y_n)h_n + g(Y_n)\Delta W_n

where :math:`n=0,1, \cdots , N-1`, :math:`Y_0=x_0`, :math:`Y_n = Y(t_n)`,
:math:`h_n = t_{n+1} - t_n` is the step size,
:math:`\Delta W_n = [W(t_{n+1}) - W(t_n)] \sim N(0, h_n)=\sqrt{h}N(0, 1)`
with :math:`W(t_0) = 0`.

For simplicity, we rewrite the above equation into

.. math::

    Y_{n+1} = Y_n + f_n h + g_n \Delta W_n

As the order of convergence for the Euler-Maruyama method is low (strong order of
convergence 0.5, weak order of convergence 1), the numerical results are inaccurate
unless a small step size is used. By adding one more term from the stochastic
Taylor expansion, one obtains a 1.0 strong order of convergence scheme known
as *Milstein scheme* [20]_.

.. _Euler-Heun method:

Euler-Heun method
*****************

If a problem is described using the Stratonovich scheme, then the Euler-Heun
method cab be used [14]_ [17]_.

.. math::
    Y_{n+1} &= Y_n + f_n h + {1 \over 2}[g_n + g(\overline{Y}_n)] \Delta W_n

    \overline{Y}_n &= Y_n + g_n \Delta W_n


Or, it is written as [22]_

.. math::

    Y_1 &= y_n + f(y_n)h + g_n \Delta W_n
    
    y_{n+1} &= y_n + {1 \over 2}[f(y_n) + f(Y_1)]h + {1 \over 2} [g(y_n) + g(Y_1)] \Delta W_n

.. _Milstein method:

Milstein method
***************

The Milstein scheme is slightly different whether it is the Itô or
Stratonovich representation that is used [14]_ [17]_ [18]_. It can be
proved that Milstein scheme converges strongly with order 1
(and weakly with order 1) to the solution of the SDE.

In Itô scheme, the Milstein method is described as

.. math::

    Y_{n+1} = Y_n + f_n h + g_n \Delta W_n + {1 \over 2}g_n g_n' [(\Delta W_n)^2 - h] \\

where :math:`g_n' = {dg(Y_n) \over dY_n}` is the first derivative of :math:`g_n`.

In Stratonovich schema, it is written as

.. math::

    Y_{n+1} = Y_n + f_n h + g_n \Delta W_n + {1 \over 2} g_n g_n' (\Delta W_n)^2

Note that when *additive noise* is used, i.e. when :math:`g_n` is constant and not
anymore a function of :math:`Y_n`, then both Itô and Stratonovich interpretations
are equivalent (:math:`g_n'= 0`).

.. _Derivative-free Milstein method:

Derivative-free Milstein method
*******************************

The drawback of the previous method is that it requires the analytic derivative
of :math:`g(Y_n)`, analytic expression that can become quickly highly complex.
The following implementation approximates this derivative thanks to a
Runge-Kutta approach [17]_.

In Itô scheme, it is expressed as

.. math::

    Y_{n+1} = Y_n + f_n h + g_n \Delta W_n + {1 \over 2\sqrt{h}}
    [g(\overline{Y_n}) - g_n] [(\Delta W_n)^2-h]

where :math:`\overline{Y_n} = Y_n + f_n h + g_n \sqrt{h}`.


In Stratonovich scheme, it is expressed as

.. math::

    Y_{n+1} = Y_n + f_n h + g_n\Delta W_n +  {1 \over 2\sqrt{h}}
    [g(\overline{Y_n}) - g_n] (\Delta W_n)^2



Define your own integration algorithms
--------------------------------------

Here, you can customize your own ODE methods.



**References**

.. [1] W. H.; Flannery, B. P.; Teukolsky, S. A.; and Vetterling,
        W. T. Numerical Recipes in FORTRAN: The Art of Scientific
        Computing, 2nd ed. Cambridge, England: Cambridge University
        Press, p. 710, 1992.
.. [2] https://lpsa.swarthmore.edu/NumInt/NumIntSecond.html
.. [3] http://mathworld.wolfram.com/Runge-KuttaMethod.html
.. [4] https://en.wikipedia.org/wiki/Runge%E2%80%93Kutta_methods
.. [5] https://zh.wikipedia.org/wiki/龙格－库塔法
.. [6] https://en.wikipedia.org/wiki/List_of_Runge%E2%80%93Kutta_methods
.. [7] https://en.wikipedia.org/wiki/Midpoint_method
.. [8] Forward and Backward Euler Methods,
       http://web.mit.edu/10.001/Web/Course_Notes/Differential_Equations_Notes/node3.html
.. [9] Butcher, John C. (2003), Numerical Methods for Ordinary
       Differential Equations, New York: John Wiley & Sons, ISBN 978-0-471-96758-3.
.. [10] Trapezoidal rule (differential equations),
        https://en.wikipedia.org/wiki/Trapezoidal_rule_(differential_equations)
.. [11] https://en.wikipedia.org/wiki/Midpoint_method

.. [12] K.E.S. Abe, W. Shaw, and M. Giles, Pricing Exotic Options using Local,
        Implied and Stochastic Volatility obtained from Market Data, (2004).
.. [14] H. Gilsing and T. Shardlow, SDELab: A package for solving stochastic
        differential equations in MATLAB, Journal of Computational and Applied
        Mathematics 205 (2007), no. 2, 1002{1018.
.. [17] P.E. Kloeden, E. Platen, and H. Schurz, Numerical solution of SDE
        through computer experiments, Springer, 1994.
.. [18] H. Lamba, Stepsize control for the Milstein scheme using rst-exit-times.
.. [19] L. Panzar and E.C. Cipu, Using of stochastic Ito and Stratonovich
        integrals derived security pricing, (2005).
.. [20] U. Picchini, Sde toolbox: Simulation and estimation of stochastic
        differential equations with matlab.
.. [21] N. G. Van Kampen, Stochastic processes in physics and chemistry,
        North-Holland, 2007.
.. [22] Burrage, Kevin, P. M. Burrage, and Tianhai Tian. "Numerical methods
        for strong solutions of stochastic differential equations:
        an overview." Proceedings of the Royal Society of London. Series
        A: Mathematical, Physical and Engineering Sciences 460.2041 (2004):
        373-402.

