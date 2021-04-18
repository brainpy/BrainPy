BrainPy documentation
=====================


Brain modeling heavily relies on calculus. Focused on differential equations,
`BrainPy <https://github.com/PKU-NIP-Lab/BrainPy>`_
provides an integrative simulation and analysis framework for neurodynamics in
computational neuroscience and brain-inspired computation. It provides three
core functions:

- *General numerical solvers* for ODEs and SDEs (future will support DDEs and FDEs).
- *Neurodynamics simulation tools* for various brain objects, such like neurons, synapses
  and networks (future will support soma and dendrites).
- *Neurodynamics analysis tools* for differential equations, including phase plane
  analysis and bifurcation analysis (future will support continuation analysis and
  sensitive analysis).

Comprehensive examples of BrainPy please see
`BrainModels <https://brainmodels.readthedocs.io/en/latest/>`_.


.. Hint::

    “Do you know calculus?”

    “You had better learn it. It’s the language God talks.”

    -- Richard Feynman


.. toctree::
   :maxdepth: 2
   :caption: Quickstart

   quickstart/installation
   quickstart/numerical_solvers
   quickstart/neurodynamics_simulation
   quickstart/neurodynamics_analysis

.. toctree::
   :maxdepth: 2
   :caption: Tutorials

   tutorials/synaptic_connectivity
   tutorials/efficient_synaptic_computation


.. toctree::
   :maxdepth: 2
   :caption: Advanced Tutorials

   tutorials_advanced/ode_numerical_solvers
   tutorials_advanced/sde_numerical_solvers
   tutorials_advanced/tips_on_numba_backend
   tutorials_advanced/how_it_works


.. toctree::
   :maxdepth: 1
   :caption: API documentation

   apis/inputs_module
   apis/changelog


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
