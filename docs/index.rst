BrainPy documentation
=====================


Brain modeling heavily relies on calculus. Focused on differential equations,
`BrainPy <https://github.com/PKU-NIP-Lab/BrainPy>`_
provides an integrative simulation and analysis framework for neurodynamics in
computational neuroscience and brain-inspired computation. It provides three
core functions:

- **General numerical solvers** for ODEs and SDEs (future will support DDEs and FDEs).
- **Neurodynamics simulation tools** for various brain objects, such like neurons, synapses
  and networks (future will support soma and dendrites).
- **Neurodynamics analysis tools** for differential equations, including phase plane
  analysis and bifurcation analysis (future will support continuation analysis and
  sensitive analysis).

Intuitive tutorials of BrainPy please see our
`handbook <https://pku-nip-lab.github.io/BrainPyHandbook/>`_,
and comprehensive examples of BrainPy please see
`BrainModels <https://brainmodels.readthedocs.io/en/latest/>`_.


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
