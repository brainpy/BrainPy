BrainPy documentation
=====================

`BrainPy`_ is a highly flexible and extensible framework targeting on the
high-performance Brain Dynamics Programming (BDP). Among its key ingredients, BrainPy supports:

- **JIT compilation** and **automatic differentiation** for class objects.
- **Numerical methods** for ordinary differential equations (ODEs),
  stochastic differential equations (SDEs),
  delay differential equations (DDEs),
  fractional differential equations (FDEs), etc.
- **Dynamics simulation** tools for various brain objects, like
  neurons, synapses, networks, soma, dendrites, channels, and even more.
- **Dynamics training** tools with various machine learning algorithms,
  like FORCE learning, ridge regression, back-propagation, etc.
- **Dynamics analysis** tools for differential equations, including
  phase plane analysis, bifurcation analysis, linearization analysis,
  and fixed/slow point finding.
- And more others ......


.. _BrainPy: https://github.com/PKU-NIP-Lab/BrainPy


Comprehensive examples of BrainPy please see:

- BrainPyExamples: https://brainpy-examples.readthedocs.io/

The code of BrainPy is open-sourced at GitHub:

- BrainPy: https://github.com/PKU-NIP-Lab/BrainPy



.. toctree::
   :maxdepth: 1
   :caption: Quickstart

   quickstart/installation
   quickstart/simulation
   quickstart/rate_model
   quickstart/training
   quickstart/analysis


.. toctree::
   :maxdepth: 2
   :caption: BDP Tutorials

   tutorial_basics/index
   tutorial_simulation/index
   tutorial_training/index
   tutorial_analysis/index


.. toctree::
   :maxdepth: 1
   :caption: Toolboxes

   tutorial_toolbox/ode_numerical_solvers
   tutorial_toolbox/sde_numerical_solvers
   tutorial_toolbox/fde_numerical_solvers
   tutorial_toolbox/dde_numerical_solvers
   tutorial_toolbox/joint_equations
   tutorial_toolbox/synaptic_connections
   tutorial_toolbox/synaptic_weights
   tutorial_toolbox/optimizers
   tutorial_toolbox/runners
   tutorial_toolbox/inputs
   tutorial_toolbox/monitors
   tutorial_toolbox/saving_and_loading


.. toctree::
   :maxdepth: 1
   :caption: Advanced Tutorials

   tutorial_math/variables
   tutorial_math/base
   tutorial_math/compilation
   tutorial_math/differentiation
   tutorial_math/control_flows
   tutorial_math/low-level_operator_customization
   tutorial_math/interoperation


.. toctree::
   :maxdepth: 1
   :caption: API Documentation

   apis/auto/base.rst
   apis/math.rst
   apis/dyn.rst
   apis/train.rst
   apis/analysis.rst
   apis/integrators.rst
   apis/datasets.rst
   apis/algorithms.rst
   apis/auto/inputs.rst
   apis/auto/connect.rst
   apis/auto/initialize.rst
   apis/auto/losses.rst
   apis/auto/optimizers.rst
   apis/auto/measure.rst
   apis/auto/running.rst
   apis/tools.rst
   apis/auto/changelog-brainpy.rst
   apis/auto/changelog-brainpylib.rst


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
