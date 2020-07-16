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

    forward_Euler
    rk2
    midpoint
    rk3
    rk4
    rk4_alternative
    backward_Euler
    trapezoidal_rule

Methods for stochastic differential equations.

.. autosummary::
    :toctree: _autosummary

    Euler_method
    Milstein_dfree_Ito
    Heun_method
    Milstein_dfree_Stra


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

    record_conductance
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


