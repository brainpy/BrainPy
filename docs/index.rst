BrainPy documentation
=====================

`BrainPy`_ is a highly flexible and extensible framework targeting on the
high-performance Brain Dynamics Programming (BDP). Among its key ingredients, BrainPy supports:

- **JIT compilation** and **automatic differentiation** for class objects.
- **Numerical methods** for ordinary differential equations (ODEs),
  stochastic differential equations (SDEs),
  delay differential equations (DDEs),
  fractional differential equations (FDEs), etc.
- **Dynamics building** with the modular and composable programming interface.
- **Dynamics simulation** for various brain objects with parallel supports.
- **Dynamics training** with various machine learning algorithms,
  like FORCE learning, ridge regression, back-propagation, etc.
- **Dynamics analysis** for low- and high-dimensional systems, including
  phase plane analysis, bifurcation analysis, linearization analysis,
  and fixed/slow point finding.
- And more others ......


.. _BrainPy: https://github.com/brainpy/BrainPy



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

   tutorial_math/index
   tutorial_building/index
   tutorial_simulation/index
   tutorial_training/index
   tutorial_analysis/index


.. toctree::
   :maxdepth: 1
   :caption: Advanced Tutorials

   tutorial_advanced/variables.ipynb
   tutorial_advanced/base_and_collector.ipynb
   tutorial_advanced/compilation.ipynb
   tutorial_advanced/differentiation.ipynb
   tutorial_advanced/adavanced_lowdim_analysis.ipynb
   tutorial_advanced/operator_customization.ipynb
   tutorial_advanced/interoperation.ipynb



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
   tutorial_toolbox/saving_and_loading
   tutorial_toolbox/inputs


.. toctree::
   :maxdepth: 1
   :caption: Frequently Asked Questions

   tutorial_FAQs/citing_and_publication
   tutorial_FAQs/uniqueness_of-brainpy-math


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
   apis/auto/changelog.rst


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
