API Documentation
=================

.. toctree::
   :maxdepth: 1


   apis/brainpy.rst
   apis/math.rst
   apis/dnn.rst
   apis/dyn.rst
   apis/integrators.rst
   apis/analysis.rst
   apis/connect.rst
   apis/encoding.rst
   apis/initialize.rst
   apis/inputs.rst
   apis/losses.rst
   apis/measure.rst
   apis/optim.rst
   apis/running.rst
   apis/mixin.rst
   apis/state.rst

.. admonition:: ``brainpy`` and ``brainpy.state``

   ``brainpy.state`` is the state-based modeling layer of BrainPy, built on
   ``brainstate`` for compatibility with the BrainX ecosystem: its models carry
   physical units through ``brainunit`` and can be trained online with
   ``braintrace``. It is the recommended starting point for new ``State``-based,
   differentiable models, while the classic ``DynamicalSystem``-based API is
   unchanged and fully supported.

   Looking for where a ``brainpy`` symbol lives in the wider ecosystem? The
   `BrainPy migration notes <https://brainx.chaobrain.com/summ/brainpy-migration-notes.html>`_
   map the classic modules onto their BrainX counterparts, and
   `BrainPy to BrainX <https://brainx.chaobrain.com/summ/brainpy-to-brainx.html>`_
   explains how the packages relate.

The following APIs will no longer be maintained in the future, but you can still use them normally.

.. toctree::
   :maxdepth: 1

   apis/deprecated/channels.rst
   apis/deprecated/neurons.rst
   apis/deprecated/rates.rst
   apis/deprecated/synapses.rst
   apis/deprecated/synouts.rst
   apis/deprecated/synplast.rst
   apis/deprecated/layers.rst
