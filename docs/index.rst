``brainpy`` documentation
=========================

`brainpy <https://github.com/brainpy/BrainPy>`_


----

Features
^^^^^^^^^

.. grid::



   .. grid-item::
      :columns: 12 12 12 6

      .. card:: Program Compilation
         :class-card: sd-border-0
         :shadow: none
         :class-title: sd-fs-6

         .. div:: sd-font-normal

            ``BrainState`` supports `program compilation <./apis/compile.html>`__ (such as just-in-time compilation) with its `state-based <./apis/brainstate.html>`__ IR construction.



   .. grid-item::
      :columns: 12 12 12 6

      .. card:: Program Augmentation
         :class-card: sd-border-0
         :shadow: none
         :class-title: sd-fs-6

         .. div:: sd-font-normal

            ``BrainState`` supports program `functionality augmentation <./apis/augment.html>`__ (such batching) with its `graph-based <./apis/graph.html>`__ Python objects.




----


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

----


See also the ecosystem
^^^^^^^^^^^^^^^^^^^^^^


``brainpy`` is one part of our `brain simulation ecosystem <https://brainmodeling.readthedocs.io/>`_.


.. toctree::
   :hidden:
   :maxdepth: 1
   :caption: Quickstart

   quickstart/concepts-en.ipynb
   quickstart/concepts-zh.ipynb





.. toctree::
   :hidden:
   :maxdepth: 2
   :caption: API Reference

   changelog.md
   apis/brainstate.rst
   apis/graph.rst
   apis/transform.rst
   apis/nn.rst
   apis/random.rst
   apis/util.rst
   apis/surrogate.rst
   apis/typing.rst
   apis/mixin.rst
   apis/environ.rst

