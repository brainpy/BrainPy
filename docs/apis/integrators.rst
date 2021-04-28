``brainpy.integrators`` module
==============================

.. currentmodule:: brainpy.integrators
.. automodule:: brainpy.integrators

.. toctree::
    :maxdepth: 3

    integrators/ODE
    integrators/SDE


General functions
-----------------

.. autosummary::
    :toctree: _autosummary

    odeint
    sdeint
    set_default_odeint
    get_default_odeint
    set_default_sdeint
    get_default_sdeint



.. rubric:: :doc:`integrators/ODE`

.. autosummary::
    :nosignatures:

    ode.euler
    ode.midpoint
    ode.heun2
    ode.ralston2
    ode.rk2
    ode.rk3
    ode.heun3
    ode.ralston3
    ode.ssprk3
    ode.rk4
    ode.ralston4
    ode.rk4_38rule

    ode.rkf45
    ode.rkf12
    ode.rkdp
    ode.ck
    ode.bs
    ode.heun_euler

    ode.exponential_euler



.. rubric:: :doc:`integrators/SDE`

.. autosummary::
    :nosignatures:

    sde.euler
    sde.heun
    sde.milstein
    sde.exponential_euler
    sde.srk1w1_scalar
    sde.srk2w1_scalar
    sde.KlPl_scalar
