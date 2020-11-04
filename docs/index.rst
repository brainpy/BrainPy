BrainPy documentation
========================

``BrainPy`` is a lightweight framework based on the latest Just-In-Time (JIT)
compilers. The goal of ``BrainPy`` is to provide
a highly flexible and efficient neural simulation framework for Python users.
It endows the users with the fully data/logic flow control.
The core of the framework is a micro-kernel, and it's easy to understand (see
*the document coming soon*).
Based on the kernel, the extension of the new models or the customization of the
data/logic flows are very simple for users. Ample examples (such as LIF neuron,
HH neuron, or AMPA synapse, GABA synapse and GapJunction) are also provided.
Besides the consideration of **flexibility**, for accelerating the running
**speed** of NumPy codes, `Numba` is used. For most of the times,
models running on `Numba` backend is very fast.


.. note::

    BrainPy is a project under development.
    More features are coming soon. Contributions are welcome.
    https://github.com/PKU-NIP-Lab/BrainPy


.. toctree::
   :maxdepth: 1
   :caption: Introduction

   intro/installation
   intro/motivations

.. toctree::
   :maxdepth: 1
   :caption: User guides

   guides/numerical_integrators
   guides/usage_of_connect_module
   guides/usage_of_inputs_module

.. toctree::
   :maxdepth: 1
   :caption: API documentation


   apis/profile
   apis/numpy
   apis/core
   apis/integration
   apis/connectivity
   apis/visualization
   apis/measure
   apis/running
   apis/inputs
   apis/errors
   apis/tools
   apis/changelog


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
