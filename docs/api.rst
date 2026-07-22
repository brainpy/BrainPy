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
   ``brainpy.state`` module <https://brainx.chaobrain.com/brainpy-state/apis/index.html>

.. admonition:: ``brainpy`` and ``brainpy.state``

   ``brainpy.state`` is the state-based modeling layer of BrainPy — developed and
   released as the standalone ``brainpy_state`` package and surfaced here through the
   ``brainpy.state`` namespace (bundled with ``brainpy >= 2.7.6``; no separate install
   needed). It is the recommended starting point for new ``State``-based, differentiable
   spiking-network models, while the classic ``DynamicalSystem``-based API is unchanged
   and fully supported. See the `brainpy.state relationship page
   <https://brainx.chaobrain.com/brainpy-state/project/relationship.html>`_.

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
