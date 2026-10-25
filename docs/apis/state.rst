``brainpy.state`` module
========================

The ``brainpy.state`` module provides state-based neuron, synapse, synaptic
output, projection, short-term plasticity, readout, and input models built on
``brainstate``. Every model keeps its variables in explicit ``State`` objects,
so it composes with the ``brainstate`` transformations (``jit``, ``grad``,
``vmap``, ``for_loop``) for both simulation and gradient-based training.

``brainpy.state`` is the state-based modeling layer of BrainPy, built on
``brainstate`` for compatibility with the BrainX ecosystem: its models carry
physical units through ``brainunit`` and can be trained online with
``braintrace``. It is the recommended starting point for new ``State``-based,
differentiable models, while the classic ``DynamicalSystem``-based API is
unchanged and fully supported.

.. toctree::
   :maxdepth: 1

   brainpy.state.base.rst
   brainpy.state.neurons.rst
   brainpy.state.synapses.rst
   brainpy.state.synouts.rst
   brainpy.state.projections.rst
   brainpy.state.plasticity.rst
   brainpy.state.readouts.rst
   brainpy.state.inputs.rst
