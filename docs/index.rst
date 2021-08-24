BrainPy documentation
=====================

BrainPy is a highly flexible and extensible framework targeting on the
high-performance brain modeling. Among its key ingredients, BrainPy supports:

1. **General numerical solvers** for ODEs, SDEs, DDEs, FDEs and others.
2. **Dynamics simulation tools** for various brain objects, like
   neurons, synapses, networks, soma, dendrites, channels, and even molecular.
3. **Dynamics analysis tools** for differential equations, including
   phase plane analysis, bifurcation analysis, continuation analysis and
   sensitive analysis.
4. **Seamless integration with deep learning models**, but has the high speed
   acceleration because of JIT compilation.


Intuitive tutorials of BrainPy please see our
`handbook <https://pku-nip-lab.github.io/BrainPyHandbook/>`_,
and comprehensive examples of BrainPy please see
`BrainModels <https://brainmodels.readthedocs.io/>`_.


.. toctree::
   :maxdepth: 1
   :caption: Quickstart

   quickstart/installation
   quickstart/numerical_solvers
   quickstart/sim0_neurodynamics_simulation
   quickstart/sim1_efficient_synaptic_computation
   quickstart/sim2_synaptic_connectivity
   quickstart/sim3_running_order_scheduling
   quickstart/sim4_monitor
   quickstart/sim5_unified_operations
   quickstart/sim6_repeat_mode
   quickstart/neurodynamics_analysis
   quickstart/how_brainpy_works

.. toctree::
   :maxdepth: 1
   :caption: Backend on Tensor/Numba

   tutorials/numba_cpu_backend


.. toctree::
   :maxdepth: 1
   :caption: Backend on Numba-CUDA

   tutorials/tips_on_numba_backend

.. toctree::
   :maxdepth: 1
   :caption: Backend on JAX



.. toctree::
   :maxdepth: 1
   :caption: API documentation

   apis/analysis
   apis/integrators
   apis/connectivity
   apis/inputs_module
   apis/changelog


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
