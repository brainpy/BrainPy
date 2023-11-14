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

            BrainPy provides various numerical integration methods for ODEs, SDEs, DDEs, FDEs, etc.

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

            BrainPy supports dynamics analysis for low- and high-dimensional systems, including phase plane, bifurcation, linearization, and fixed/slow point analysis.

----

Installation
^^^^^^^^^^^^
.. tab-set::

    .. tab-item:: CPU

       .. code-block:: bash

          pip install -U "jax[cpu]"

          pip install -U brainpy brainpylib  # windows, linux, macos

    .. tab-item:: GPU (CUDA-11x)

       .. code-block:: bash

          pip install -U "jax[cuda11_local]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

          pip install -U brainpy brainpylib-cu11x  # only on linux

    .. tab-item:: GPU (CUDA-12x)

       .. code-block:: bash

          pip install --upgrade "jax[cuda12_local]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

          pip install -U brainpy brainpylib-cu12x  # only on linux

For more information about supported accelerators and platforms, and for other installation details, please see `installation <quickstart/installation.html>`_ section.

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

      .. card:: :material-regular:`science;2em` BDP Tutorials
         :class-card: sd-text-black sd-bg-light
         :link: tutorials.html

   .. grid-item::
      :columns: 6 6 6 4

      .. card:: :material-regular:`token;2em` Advanced Tutorials
         :class-card: sd-text-black sd-bg-light
         :link: advanced_tutorials.html

   .. grid-item::
      :columns: 6 6 6 4

      .. card:: :material-regular:`settings;2em` BDP Toolboxes
         :class-card: sd-text-black sd-bg-light
         :link: toolboxes.html

   .. grid-item::
      :columns: 6 6 6 4

      .. card:: :material-regular:`rocket_launch;2em` FAQ
         :class-card: sd-text-black sd-bg-light
         :link: FAQ.html

   .. grid-item::
      :columns: 6 6 6 4

      .. card:: :material-regular:`data_exploration;2em` API documentation
         :class-card: sd-text-black sd-bg-light
         :link: api.html

   .. grid-item::
      :columns: 6 6 6 4

      .. card:: :material-regular:`settings;2em` Examples
         :class-card: sd-text-black sd-bg-light
         :link: https://brainpy-examples.readthedocs.io/en/latest/index.html


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
   tutorials.rst
   toolboxes.rst
   advanced_tutorials.rst
   FAQ.rst
   api.rst

