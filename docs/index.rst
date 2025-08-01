BrainPy documentation
=====================

`BrainPy`_ is a highly flexible and extensible framework targeting on the
general-purpose Brain Dynamics Programming (BDP).

.. _BrainPy: https://github.com/brainpy/BrainPy


----

Installation
^^^^^^^^^^^^
.. tab-set::

    .. tab-item:: CPU

       .. code-block:: bash

          pip install -U brainpy[cpu]

    .. tab-item:: GPU (CUDA 12)

       .. code-block:: bash

          pip install -U brainpy[cuda12]

    .. tab-item:: TPU

       .. code-block:: bash

          pip install -U brainpy[tpu]


    .. tab-item:: Ecosystem

       .. code-block:: bash

          pip install -U BrainX


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

   .. grid-item::
      :columns: 6 6 6 4

      .. card:: :material-regular:`settings;2em` Ecosystem
         :class-card: sd-text-black sd-bg-light
         :link: https://brainmodeling.readthedocs.io


.. note::
   BrainPy is still an experimental research project.
   APIs may be changed over time. Please always keeps
   in mind what BrainPy version you are using.


.. note::
   BrainPy now is rewritten based on [brainstate](https://github.com/chaobrain/brainstate).



----



.. toctree::
   :hidden:
   :maxdepth: 1
   :caption: Quickstart

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

