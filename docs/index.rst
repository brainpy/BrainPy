BrainPy documentation
=====================

`BrainPy`_ is a highly flexible and extensible framework targeting on the
high-performance brain modeling. Among its key ingredients, BrainPy supports:

1. **JIT compilation** for class objects.
2. **Numerical solvers** for ODEs, SDEs, DDEs, FDEs and others.
3. **Dynamics simulation tools** for various brain objects, like
   neurons, synapses, networks, soma, dendrites, channels, and even more.
4. **Dynamics analysis tools** for differential equations, including
   phase plane analysis, bifurcation analysis, continuation analysis and
   sensitive analysis.
5. **Seamless integration with deep learning models**, but has the high speed
   acceleration because of JIT compilation.
6. And more ......


.. _BrainPy: https://github.com/PKU-NIP-Lab/BrainPy

.. note::

    Comprehensive examples of BrainPy please see BrainModels: https://github.com/PKU-NIP-Lab/BrainModels.


.. toctree::
   :maxdepth: 1
   :caption: Quickstart

   quickstart/installation
   quickstart/jit_compilation
   quickstart/numerical_solvers
   quickstart/dynamics_simulation
   quickstart/deep_neural_network


.. toctree::
   :maxdepth: 1
   :caption: JIT tutorials




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
   tutorial_simulation/monitor_and_inputs
   tutorial_simulation/inputs


.. toctree::
   :maxdepth: 1
   :caption: Analysis Tutorials

   tutorial_analysis/sym_analysis


.. toctree::
   :maxdepth: 1
   :caption: DNN Tutorials


.. toctree::
   :maxdepth: 1
   :caption: API documentation

   apis/base.rst
   apis/math.rst
   apis/integrators.rst
   apis/simulation.rst
   apis/analysis.rst
   apis/visualization.rst
   apis/tools.rst
   apis/changelog.rst


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
