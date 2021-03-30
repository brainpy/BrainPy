BrainPy documentation
=====================

``BrainPy`` is an integrative framework for computational neuroscience
and brain-inspired computation. It provides three core functions for
neurodyanmics modeling:

- *General numerical solvers* for ODEs and SDEs (future will support DDEs and FDEs).
- *Neurodynamics simulation tools* for various brain objects, such like neurons, synapses
  and networks (future will support soma and dendrites).
- *Neurodynamics analysis tools* for differential equations, including phase plane
  analysis and bifurcation analysis (future will support continuation analysis and
  sensitive analysis).


Comprehensive examples of BrainPy please see
`BrainModels <https://brainmodels.readthedocs.io/en/latest/>`_.


.. note::

    BrainPy is a project under development.
    More features are coming soon. Contributions are welcome.
    https://github.com/PKU-NIP-Lab/BrainPy


.. toctree::
   :maxdepth: 1
   :caption: Quickstart

   quickstart/installation
   quickstart/numerical_solvers
   quickstart/neurodynamics_simulation
   quickstart/neurodynamics_analysis

.. toctree::
   :maxdepth: 2
   :caption: Tutorials

..   tutorials/tutorial_for_computational_neuroscience
..   tutorials/tutorial_for_brain_inspired_computation

.. toctree::
   :maxdepth: 1
   :caption: Advanced Tutorials

   tutorials_advanced/ode_numerical_solvers
   tutorials_advanced/sde_numerical_solvers
   tutorials_advanced/usage_of_inputs_module
   tutorials_advanced/tips_on_jit
   tutorials_advanced/how_it_works


.. toctree::
   :maxdepth: 1
   :caption: API documentation

   apis/analysis
   apis/backend
   apis/connectivity
   apis/integrators
   apis/simulation
   apis/tools
   apis/visualization
   apis/errors
   apis/inputs
   apis/measure
   apis/running
   apis/changelog


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
