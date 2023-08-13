BrainPy documentation
=====================

`BrainPy`_ is a highly flexible and extensible framework targeting on the
general-purpose Brain Dynamics Programming (BDP). Among its key ingredients, BrainPy supports:

.. _BrainPy: https://github.com/brainpy/BrainPy


Features
^^^^^^^^^

.. grid::

   .. grid-item::
      :columns: 12 12 12 6

      .. card:: OO Transformations
         :class-card: sd-border-0
         :shadow: none
         :class-title: sd-fs-5

         .. div:: sd-font-normal

            BrainPy supports object-oriented transformations, including
            JIT compilation, Autograd.

   .. grid-item::
      :columns: 12 12 12 6

      .. card:: Numerical Integrators
         :class-card: sd-border-0
         :shadow: none
         :class-title: sd-fs-5

         .. div:: sd-font-normal

            Numerical methods for ODEs, SDEs, DDEs, FDEs, etc.

   .. grid-item::
      :columns: 12 12 12 6

      .. card:: Model Building
         :class-card: sd-border-0
         :shadow: none
         :class-title: sd-fs-5

         .. div:: sd-font-normal

            BrainPy provides a modular and composable programming interface for building dynamics.

   .. grid-item::
      :columns: 12 12 12 6

      .. card:: Model Simulation
         :class-card: sd-border-0
         :shadow: none
         :class-title: sd-fs-5

         .. div:: sd-font-normal

            BrainPy supports dynamics simulation for various brain objects with parallel supports.


   .. grid-item::
      :columns: 12 12 12 6

      .. card:: Model Training
         :class-card: sd-border-0
         :shadow: none
         :class-title: sd-fs-5

         .. div:: sd-font-normal

            BrainPy supports dynamics training with various machine learning algorithms, like FORCE learning, ridge regression, back-propagation, etc.

   .. grid-item::
      :columns: 12 12 12 6

      .. card:: Model Analysis
         :class-card: sd-border-0
         :shadow: none
         :class-title: sd-fs-5

         .. div:: sd-font-normal

            BrainPy supports dynamics analysis for low- and high-dimensional systems, including phase plane analysis, bifurcation analysis, linearization analysis, and fixed/slow point finding.

----

Installation
^^^^^^^^^^^^
.. tab-set::

    .. tab-item:: CPU

       .. code-block:: bash

          pip install brainpy brainpylib

    .. tab-item:: GPU (CUDA)

       .. code-block:: bash

          pip install brainpy brainpylib

For more information about supported accelerators and platforms, and for other installation details, please see installation section.

----

Learn more
^^^^^^^^^^

.. grid::

   .. grid-item::
      :columns: 6 6 6 4

      .. card:: :material-regular:`rocket_launch;2em` Installation
         :class-card: sd-text-black sd-bg-light
         :link: quickstart/installation.html

   .. grid-item::
      :columns: 6 6 6 4

      .. card:: :material-regular:`library_books;2em` Core Concepts
         :class-card: sd-text-black sd-bg-light
         :link: core_concepts.html

   .. grid-item::
      :columns: 6 6 6 4

      .. card:: :material-regular:`science;2em` Brain Dynamics Tutorials
         :class-card: sd-text-black sd-bg-light
         :link: brain_dynamics_tutorials.html

   .. grid-item::
      :columns: 6 6 6 4

      .. card:: :material-regular:`science;2em` Advanced Tutorials
         :class-card: sd-text-black sd-bg-light
         :link: advanced_tutorials.html

   .. grid-item::
      :columns: 6 6 6 4

      .. card:: :material-regular:`science;2em` Toolboxes
         :class-card: sd-text-black sd-bg-light
         :link: toolboxes.html

   .. grid-item::
      :columns: 6 6 6 4

      .. card:: :material-regular:`science;2em` Frequently Asked Questions
         :class-card: sd-text-black sd-bg-light
         :link: FAQ.html

   .. grid-item::
      :columns: 6 6 6 4

      .. card:: :material-regular:`science;2em` API documentation
         :class-card: sd-text-black sd-bg-light
         :link: api.html

.. note::
   BrainPy is still an experimental research project.
   APIs may be changed over time. Please always keeps
   in mind what BrainPy version you are using.


.. toctree::
   :hidden:
   :maxdepth: 1
   :caption: Quickstart

   quickstart/installation
   quickstart/simulation
   quickstart/training
   quickstart/analysis



.. toctree::
   :hidden:
   :maxdepth: 2
   :caption: Tutorials

   core_concepts.rst
   brain_dynamics_tutorials.rst
   advanced_tutorials.rst
   toolboxes.rst
   FAQ.rst
   api.rst



Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
