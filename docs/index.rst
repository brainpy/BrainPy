BrainPy documentation
=====================

`BrainPy`_ is a highly flexible and extensible framework targeting on the
high-performance brain modeling. Among its key ingredients, BrainPy supports:

1. **JIT compilation** for class objects.
2. **Numerical solvers** for ODEs, SDEs and others.
3. **Dynamics simulation tools** for various brain objects, like
   neurons, synapses, networks, soma, dendrites, channels, and even more.
4. **Dynamics analysis tools** for differential equations, including
   phase plane analysis, bifurcation analysis, linearization analysis,
   and fixed/slow point finding.
5. **Seamless integration with deep learning models**.
6. And more ......


.. _BrainPy: https://github.com/PKU-NIP-Lab/BrainPy

.. note::

    Comprehensive examples of BrainPy please see:

    - BrainModels: https://brainmodels.readthedocs.io/
    - BrainPyExamples: https://brainpy-examples.readthedocs.io/



.. toctree::
   :maxdepth: 1
   :caption: Quickstart

   quickstart/installation
   quickstart/jit_compilation
   quickstart/dynamics_intro


.. toctree::
   :maxdepth: 1
   :caption: Math Foundation

   tutorial_math/overview
   tutorial_math/tensors
   tutorial_math/variables
   tutorial_math/base
   tutorial_math/compilation
   tutorial_math/differentiation
   tutorial_math/control_flows
   tutorial_math/optimizers


.. toctree::
   :maxdepth: 2
   :caption: Dynamics Ecosystem

   tutorial_intg/index
   tutorial_building/index
   tutorial_simulation/index
   tutorial_training/index
   tutorial_analysis/index


.. toctree::
   :maxdepth: 1
   :caption: API Documentation

   apis/auto/base.rst
   apis/math.rst
   apis/integrators.rst
   apis/building.rst
   apis/simulation.rst
   apis/training.rst
   apis/analysis.rst
   apis/auto/visualization.rst
   apis/auto/tools.rst
   apis/auto/changelog.rst


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
