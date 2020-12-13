Numerical integrators
=====================

``BrainPy`` provides several methods for the numerical integration
of Ordinary Differential Equations (ODEs)
and Stochastic Differential Equations (SDEs).


+-----------------------+--------------------+------------+-------------+
|         Method        |  keyword for use   |     ODE    |      SDE    |
+=======================+====================+============+=============+
| `Euler`_              |  euler             |     Y      |       Y     |
+-----------------------+--------------------+------------+-------------+
| `ExponentialEuler`_   | exponential        |     Y      |       Y     |
+-----------------------+--------------------+------------+-------------+
| `MidPoint`_           |  midpoint          |     Y      |       N     |
+-----------------------+--------------------+------------+-------------+
| `Heun`_               |  heun              |     Y      |       Y     |
+-----------------------+--------------------+------------+-------------+
| `RK2`_                |  rk2               |     Y      | coming soon |
+-----------------------+--------------------+------------+-------------+
| `RK3`_                |  rk3               |     Y      | coming soon |
+-----------------------+--------------------+------------+-------------+
| `RK4`_                |  rk4               |     Y      | coming soon |
+-----------------------+--------------------+------------+-------------+
| `RK4Alternative`_     | rk4_alternative    |     Y      |       N     |
+-----------------------+--------------------+------------+-------------+
| `MilsteinIto`_        | milstein_ito       |     Y      |       Y     |
+-----------------------+--------------------+------------+-------------+
| `MilsteinStra`_       | milstein_stra      |     Y      |       Y     |
+-----------------------+--------------------+------------+-------------+


.. _Euler: ../apis/_autosummary/brainpy.integration.Euler.rst
.. _MidPoint: ../apis/_autosummary/brainpy.integration.MidPoint.rst
.. _Heun: ../apis/_autosummary/brainpy.integration.Heun.rst
.. _RK2: ../apis/_autosummary/brainpy.integration.RK2.rst
.. _RK3: ../apis/_autosummary/brainpy.integration.RK3.rst
.. _RK4: ../apis/_autosummary/brainpy.integration.RK4.rst
.. _RK4Alternative: ../apis/_autosummary/brainpy.integration.RK4Alternative.rst
.. _MilsteinIto: ../apis/_autosummary/brainpy.integration.MilsteinIto.rst
.. _MilsteinStra: ../apis/_autosummary/brainpy.integration.MilsteinStra.rst
.. _ExponentialEuler: ../apis/_autosummary/brainpy.integration.ExponentialEuler.rst
