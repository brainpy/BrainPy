``brainpy`` documentation
=========================

`brainpy <https://github.com/brainpy/BrainPy>`_ provides a powerful and flexible framework
for building, simulating, and training spiking neural networks.



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
   apis.rst

