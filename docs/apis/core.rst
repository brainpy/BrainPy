npbrain.core package
====================

.. currentmodule:: npbrain.core
.. automodule:: npbrain.core


.. contents::
    :local:
    :depth: 2


Numerical integration methods
-----------------------------

The most commonly used function is `integrate`:

.. autosummary::
    :toctree: _autosummary

    integrate

Methods for ordinary differential equations.

.. autosummary::
    :toctree: _autosummary

    ode_euler
    ode_rk2
    midpoint
    ode_heun
    ode_rk3
    ode_rk4
    ode_rk4_alternative
    ode_backward_euler
    trapezoidal_rule
    ode_exponential_euler

Methods for stochastic differential equations.

.. autosummary::
    :toctree: _autosummary

    sde_euler
    Milstein_dfree_Ito
    sde_heun
    Milstein_dfree_Stra

    sde_exponential_euler


Neurons
-------

.. autosummary::
    :toctree: _autosummary

    judge_spike
    initial_neu_state
    format_geometry
    format_refractory
    generate_fake_neuron

.. autoclass:: Neurons
    :members:


Synapses
--------

.. autosummary::
    :toctree: _autosummary

    format_delay
    initial_syn_state

.. autoclass:: Synapses
    :members:


Monitors
--------

.. autoclass:: Monitor
   :members:

.. autoclass:: SpikeMonitor
   :members:

.. autoclass:: StateMonitor
   :members:

.. autosummary::
    :toctree: _autosummary

    raster_plot
    firing_rate

Network
-------

.. autoclass:: Network
   :members: add, run, run_time


