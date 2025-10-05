``brainpy.integrators`` module
==============================

.. currentmodule:: brainpy.integrators 
.. automodule:: brainpy.integrators 

.. contents::
   :local:
   :depth: 2

ODE integrators
---------------

.. currentmodule:: brainpy.integrators.ode 
.. automodule:: brainpy.integrators.ode 

Base ODE Integrator
~~~~~~~~~~~~~~~~~~~

.. autosummary::
   :toctree: generated/
   :nosignatures:
   :template: classtemplate.rst

   ODEIntegrator


Generic ODE Functions
~~~~~~~~~~~~~~~~~~~~~

.. autosummary::
   :toctree: generated/
   :nosignatures:
   :template: classtemplate.rst

   set_default_odeint
   get_default_odeint
   register_ode_integrator
   get_supported_methods


Explicit Runge-Kutta ODE Integrators
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autosummary::
   :toctree: generated/
   :nosignatures:
   :template: classtemplate.rst

   ExplicitRKIntegrator
   Euler
   MidPoint
   Heun2
   Ralston2
   RK2
   RK3
   Heun3
   Ralston3
   SSPRK3
   RK4
   Ralston4
   RK4Rule38


Adaptive Runge-Kutta ODE Integrators
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autosummary::
   :toctree: generated/
   :nosignatures:
   :template: classtemplate.rst

   AdaptiveRKIntegrator
   RKF12
   RKF45
   DormandPrince
   CashKarp
   BogackiShampine
   HeunEuler


Exponential ODE Integrators
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autosummary::
   :toctree: generated/
   :nosignatures:
   :template: classtemplate.rst

   ExponentialEuler


SDE integrators
---------------

.. currentmodule:: brainpy.integrators.sde 
.. automodule:: brainpy.integrators.sde 

Base SDE Integrator
~~~~~~~~~~~~~~~~~~~

.. autosummary::
   :toctree: generated/
   :nosignatures:
   :template: classtemplate.rst

   SDEIntegrator


Generic SDE Functions
~~~~~~~~~~~~~~~~~~~~~

.. autosummary::
   :toctree: generated/
   :nosignatures:
   :template: classtemplate.rst

   set_default_sdeint
   get_default_sdeint
   register_sde_integrator
   get_supported_methods


Normal SDE Integrators
~~~~~~~~~~~~~~~~~~~~~~

.. autosummary::
   :toctree: generated/
   :nosignatures:
   :template: classtemplate.rst

   Euler
   Heun
   Milstein
   MilsteinGradFree
   ExponentialEuler


SRK methods for scalar Wiener process
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autosummary::
   :toctree: generated/
   :nosignatures:
   :template: classtemplate.rst

   SRK1W1
   SRK2W1
   KlPl


FDE integrators
---------------

.. currentmodule:: brainpy.integrators.fde 
.. automodule:: brainpy.integrators.fde 

Base FDE Integrator
~~~~~~~~~~~~~~~~~~~

.. autosummary::
   :toctree: generated/
   :nosignatures:
   :template: classtemplate.rst

   FDEIntegrator


Generic FDE Functions
~~~~~~~~~~~~~~~~~~~~~

.. autosummary::
   :toctree: generated/
   :nosignatures:
   :template: classtemplate.rst

   set_default_fdeint
   get_default_fdeint
   register_fde_integrator
   get_supported_methods


Methods for Caputo Fractional Derivative
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autosummary::
   :toctree: generated/
   :nosignatures:
   :template: classtemplate.rst

   CaputoEuler
   CaputoL1Schema


Methods for Riemann-Liouville Fractional Derivative
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autosummary::
   :toctree: generated/
   :nosignatures:
   :template: classtemplate.rst

   GLShortMemory


