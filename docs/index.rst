BrainPy documentation
=====================

BrainPy is a highly flexible and extensible framework targeting on the
high-performance brain modeling. Among its key ingredients, BrainPy supports:

1. **General numerical solvers** for ODEs, SDEs, DDEs, FDEs and others.
2. **Dynamics simulation tools** for various brain objects, like
   neurons, synapses, networks, soma, dendrites, channels, and even more.
3. **Dynamics analysis tools** for differential equations, including
   phase plane analysis, bifurcation analysis, continuation analysis and
   sensitive analysis.
4. **Seamless integration with deep learning models**, but has the high speed
   acceleration because of JIT compilation.
5. And more ......


.. note::

    Comprehensive examples of BrainPy please see BrainModels: https://github.com/PKU-NIP-Lab/BrainModels.


.. toctree::
   :maxdepth: 1
   :caption: Quickstart

   quickstart/installation
   quickstart/numerical_solvers
   quickstart/dynamics_simulation
   quickstart/dynamics_analysis
   quickstart/deep_neural_network
   quickstart/how_brainpy_works


.. toctree::
   :maxdepth: 1
   :caption: Tutorials for math module


.. toctree::
   :maxdepth: 1
   :caption: Integrator Tutorials

   tutorial_intg/ode_numerical_solvers
   tutorial_intg/sde_numerical_solvers

.. toctree::
   :maxdepth: 1
   :caption: Simulation Tutorials

   tutorial_simulation/efficient_synaptic_computation
   tutorial_simulation/synaptic_connectivity
   tutorial_simulation/monitor
   tutorial_simulation/repeat_mode


.. toctree::
   :maxdepth: 1
   :caption: Analysis Tutorials


.. toctree::
   :maxdepth: 1
   :caption: DNN Tutorials


.. toctree::
   :maxdepth: 1
   :caption: API documentation

   apis/base
   apis/math
   apis/integrators
   apis/simulation
   apis/analysis
   apis/dnn
   apis/inputs
   apis/measure
   apis/visualization
   apis/changelog


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
