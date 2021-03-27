Numerical Solvers for ODEs
==========================

``BrainPy`` provides several methods for
the numerical integration of Ordinary Differential Equations (ODEs).



Explicit Runge-Kutta methods
----------------------------

.. list-table::
   :header-rows: 1

   * - Methods
     - Keywords
   * - Euler
     - euler
   * - Midpoint
     - midpoint
   * - Heun's second-order method
     - heun2
   * - Ralston's second-order method
     - ralston2
   * - RK2
     - rk2
   * - RK3
     - rk3
   * - RK4
     - rk4
   * - Heun's third-order method
     - heun3
   * - Ralston's third-order method
     - ralston3
   * - Third-order Strong Stability Preserving Runge-Kutta
     - ssprk3
   * - Ralston's fourth-order method
     - ralston4
   * - Runge-Kutta 3/8-rule fourth-order method
     - rk4_38rule


Adaptive Runge-Kutta methods
----------------------------


.. list-table::
   :header-rows: 1

   * - Methods
     - keywords
   * - Runge–Kutta–Fehlberg 4(5)
     - rkf45
   * - Runge–Kutta–Fehlberg 1(2)
     - rkf12
   * - Dormand–Prince method
     - rkdp
   * - Cash–Karp method
     - ck
   * - Bogacki–Shampine method
     - bs
   * - Heun–Euler method
     - heun_euler


Other methods
-------------

.. list-table::
   :header-rows: 1

   * - Methods
     - keywords
   * - Exponential Euler
     - exponential_euler
