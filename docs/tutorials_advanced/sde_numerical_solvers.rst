Numerical Solvers for SDEs
==========================

``BrainPy`` provides several methods for the numerical integration
of Stochastic Differential Equations (SDEs).


Methods for scalar Wiener process
---------------------------------

.. list-table::
   :header-rows: 1

   * - Methods
     - Keywords
     - Supported SDE type
   * - Strong SRK scheme SRI1W1
     - srk1w1_scalar
     - ITO_SDE
   * - Strong SRK scheme SRI2W1
     - srk2w1_scalar
     - ITO_SDE
   * - Strong SRK scheme KlPl
     - KlPl_scalar
     - ITO_SDE
   * - Euler method
     - euler
     - ITO_SDE, STRA_SDE
   * - Heun method
     - heun
     - STRA_SDE
   * - Derivative-free Milstein
     - milstein
     - ITO_SDE, STRA_SDE
   * - Exponential Euler
     - exponential_euler
     - ITO_SDE

Methods for vector Wiener process
---------------------------------

.. list-table::
   :header-rows: 1

   * - Methods
     - Keywords
     - Supported SDE type
   * - Euler method
     - euler
     - ITO_SDE, STRA_SDE
   * - Heun method
     - heun
     - STRA_SDE
   * - Derivative-free Milstein
     - milstein
     - ITO_SDE, STRA_SDE
   * - Exponential Euler
     - exponential_euler
     - ITO_SDE
