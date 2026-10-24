How-to guides
=============

Task-oriented recipes for the BrainPy-style layer. Each guide is self-contained
and solves one concrete problem — pick the one that matches what you need right
now. The guides are organized into two tracks that mirror the dual audience of
``brainpy.state``.

.. grid:: 1 1 2 2
   :gutter: 3

   .. grid-item-card:: Simulation track
      :class-header: sd-bg-light

      For computational-neuroscience work: building and running biophysical
      networks, choosing models, shaping synaptic interactions, and reproducing
      published results.
      ^^^
      - :doc:`sim-choose-neuron`
      - :doc:`sim-coba-cuba-synapses`
      - :doc:`sim-short-term-plasticity`
      - :doc:`sim-delays`
      - :doc:`sim-reproduce-a-paper`

   .. grid-item-card:: Training track
      :class-header: sd-bg-light

      For brain-inspired computing / SNN-ML work: making spiking networks
      differentiable, attaching readouts, and training through long rollouts
      without running out of memory.
      ^^^
      - :doc:`train-surrogate-gradients`
      - :doc:`train-readouts`
      - :doc:`train-long-rollouts-checkpoint`

The two tracks share the same models and the same transform primitives — the
split is by *task*, not by a different library. A network you build for
simulation can be trained, and a trained network can be simulated. The bridge
between the two is explained in :doc:`/state/concepts/alignpre-alignpost` and
:doc:`/state/concepts/differentiability`.

.. rubric:: Simulation track

.. toctree::
   :maxdepth: 1

   sim-choose-neuron
   sim-coba-cuba-synapses
   sim-short-term-plasticity
   sim-delays
   sim-reproduce-a-paper

.. rubric:: Training track

.. toctree::
   :maxdepth: 1

   train-surrogate-gradients
   train-readouts
   train-long-rollouts-checkpoint
