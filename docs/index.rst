BrainPy documentation
=====================

`BrainPy`_ is a highly flexible and extensible framework targeting on the
high-performance brain modeling. Among its key ingredients, BrainPy supports:

1. **JIT compilation** for functions and class objects.
2. **Numerical solvers** for ODEs, SDEs and others.
3. **Dynamics simulation tools** for various brain objects, like
   neurons, synapses, networks, soma, dendrites, channels, and even more.
4. **Dynamics analysis tools** for differential equations, including
   phase plane analysis, bifurcation analysis, and
   linearization analysis.
5. **Seamless integration with deep learning models**, but has the high speed
   acceleration because of JIT compilation.
6. And more ......


.. _BrainPy: https://github.com/PKU-NIP-Lab/BrainPy

.. note::

    Comprehensive examples of BrainPy please see:

    - BrainModels: https://github.com/PKU-NIP-Lab/BrainModels
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

   tutorial_math/tensors
   tutorial_math/variables
   tutorial_math/base
   tutorial_math/compilation
   tutorial_math/differentiation
   tutorial_math/control_flows
   tutorial_math/optimizers


.. toctree::
   :maxdepth: 2
   :caption: Programming System

   tutorial_intg/index
   tutorial_simulation/index
   tutorial_analysis/index
   tutorial_training/index


.. toctree::
   :maxdepth: 1
   :caption: API Documentation

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
