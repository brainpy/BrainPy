``brainpy.state`` documentation
=====================================

`brainpy.state` provides a new ``State``-based programming paradigm for building and simulating spiking neural networks.


Compared to ``brainpy.dyn``, ``brainpy.state`` provides:

  - A more intuitive and flexible way to define and manage the state of neural network components (neurons, synapses, etc.).

  - Improved performance and scalability for large-scale simulations.

  - Seamless integration with `BrainX <https://brainmodeling.readthedocs.io>`_ ecosystem.


.. note::

   ``brainpy.state`` is written based on `brainstate <https://github.com/chaobrain/brainstate>`_.
   This documentation is for the latest version 3.x.



Installation
^^^^^^^^^^^^

.. tab-set::

    .. tab-item:: CPU

       .. code-block:: bash

          pip install -U brainpy[cpu]

    .. tab-item:: GPU

       .. code-block:: bash

          pip install -U brainpy[cuda12]

          pip install -U brainpy[cuda13]

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

      .. card:: :material-regular:`rocket_launch;2em` 5-Minute Tutorial
         :class-card: sd-text-black sd-bg-light
         :link: quickstart/5min-tutorial.html

   .. grid-item::
      :columns: 6 6 6 4

      .. card:: :material-regular:`library_books;2em` Core Concepts
         :class-card: sd-text-black sd-bg-light
         :link: quickstart/concepts-overview.html

   .. grid-item::
      :columns: 6 6 6 4

      .. card:: :material-regular:`school;2em` Tutorials
         :class-card: sd-text-black sd-bg-light
         :link: tutorials/index.html

   .. grid-item::
      :columns: 6 6 6 4

      .. card:: :material-regular:`explore;2em` Examples Gallery
         :class-card: sd-text-black sd-bg-light
         :link: examples/gallery.html

   .. grid-item::
      :columns: 6 6 6 4

      .. card:: :material-regular:`data_exploration;2em` API Documentation
         :class-card: sd-text-black sd-bg-light
         :link: apis.html

   .. grid-item::
      :columns: 6 6 6 4

      .. card:: :material-regular:`swap_horiz;2em` Migration from 2.x
         :class-card: sd-text-black sd-bg-light
         :link: migration/migration-guide.html

   .. grid-item::
      :columns: 6 6 6 4

      .. card:: :material-regular:`settings;2em` Ecosystem
         :class-card: sd-text-black sd-bg-light
         :link: https://brainmodeling.readthedocs.io

   .. grid-item::
      :columns: 6 6 6 4

      .. card:: :material-regular:`history;2em` Changelog
         :class-card: sd-text-black sd-bg-light
         :link: changelog.html

   .. grid-item::
      :columns: 6 6 6 4

      .. card:: :material-regular:`data_exploration;2em` Classical APIs
         :class-card: sd-text-black sd-bg-light
         :link: https://brainpy.readthedocs.io/


----

See also the ecosystem
^^^^^^^^^^^^^^^^^^^^^^

``brainpy`` is one part of our `brain simulation ecosystem <https://brainmodeling.readthedocs.io/>`_.




.. toctree::
   :hidden:
   :maxdepth: 1
   :caption: Quickstart

   quickstart/installation.rst
   quickstart/5min-tutorial.ipynb
   quickstart/concepts-overview.rst


.. toctree::
   :hidden:
   :maxdepth: 2
   :caption: Core Concepts

   core-concepts/architecture.rst
   core-concepts/neurons.rst
   core-concepts/synapses.rst
   core-concepts/projections.rst
   core-concepts/state-management.rst


.. toctree::
   :hidden:
   :maxdepth: 2
   :caption: Tutorials

   tutorials/index.rst


.. toctree::
   :hidden:
   :maxdepth: 2
   :caption: How-to Guides

   how-to-guides/index.rst


.. toctree::
   :hidden:
   :maxdepth: 2
   :caption: Examples

   examples/gallery.rst


.. toctree::
   :hidden:
   :maxdepth: 2
   :caption: Migration

   migration/migration-guide.rst


.. toctree::
   :hidden:
   :maxdepth: 2
   :caption: API Reference

   api/index.rst
   changelog.md

