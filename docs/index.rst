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

    The source code of BrainPy is open-sourced in GitHub:

    - BrainPy: https://github.com/PKU-NIP-Lab/BrainPy


.. toctree::
   :maxdepth: 1
   :caption: Quickstart

   quickstart/installation
   quickstart/simulation
   quickstart/training
   quickstart/analysis


.. toctree::
   :maxdepth: 2
   :caption: BDP Tutorials

   tutorial_basics/math_basics
   tutorial_simulation/index
   tutorial_training/index
   tutorial_analysis/index


.. toctree::
   :maxdepth: 2
   :caption: Toolboxes

   tutorial_intg/index
   tutorial_toolbox/synaptic_connections
   tutorial_toolbox/synaptic_weights
   tutorial_toolbox/optimizers
   tutorial_toolbox/loss
   tutorial_toolbox/other


.. toctree::
   :maxdepth: 1
   :caption: Advanced Tutorials

   tutorial_math/overview
   tutorial_math/tensors
   tutorial_math/variables
   tutorial_math/base
   tutorial_math/compilation
   tutorial_math/differentiation
   tutorial_math/control_flows


.. toctree::
   :maxdepth: 1
   :caption: API Documentation

   apis/auto/base.rst
   apis/math.rst
   apis/dyn.rst
   apis/nn.rst
   apis/analysis.rst
   apis/integrators.rst
   apis/auto/inputs.rst
   apis/auto/running.rst
   apis/auto/connect.rst
   apis/auto/initialize.rst
   apis/auto/losses.rst
   apis/auto/optimizers.rst
   apis/auto/measure.rst
   apis/auto/datasets.rst
   apis/auto/tools.rst
   apis/auto/changelog-brainpy.rst
   apis/auto/changelog-brainpylib.rst


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
