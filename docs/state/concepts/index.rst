=============
Core Concepts
=============

One differentiable substrate, two worlds
========================================

``brainpy.state`` exists to make a single argument: **brain simulation and
brain-inspired computing are the same computation, expressed once.** The neurons,
synapses, and projections you assemble to *simulate* a biophysical network are the
exact objects you *train* with gradients and scale with linear-memory online
learning. There is no second framework, no re-implementation, no translation layer
between the two worlds — one substrate carries both.

That substrate rests on a few deliberate design choices, and this spine explains
each one in the order it builds on the last:

- **State.** Functional JAX has no mutable variables, yet a spiking network is
  nothing but evolving state (membrane potentials, conductances, traces). Explicit
  ``State`` objects reconcile the two — and, crucially, make every variable visible
  to autodiff and compilation.
- **Physical units.** Voltages are in millivolts, time constants in milliseconds,
  conductances in nanosiemens. Carrying units through construction turns a whole
  class of modeling errors into exceptions raised *before* a simulation runs.
- **Model anatomy.** Neurons and synapses share one ``Dynamics`` contract:
  initialize state, receive input, ``update()`` one step, expose ``get_spike()``.
  Understanding that contract is what lets projections wire populations together.
- **AlignPre / AlignPost** — the **keystone**. The same alignment of synaptic state
  to a *neuron* dimension that makes simulation memory-efficient is what makes
  gradient-based and online learning memory-efficient. This is the hinge of the
  whole framework, and the one chapter to read if you read only one.
- **Differentiability** and **online learning** — the two halves of the bridge.
  Surrogate gradients make spikes trainable through backpropagation-through-time;
  neuron-aligned state then collapses real-time recurrent learning from cubic to
  *linear* memory, putting whole-brain-scale online training within reach.

.. admonition:: The keystone, in one sentence
   :class: note

   A naive simulator stores one synaptic variable per *synapse* — O(p · N_pre ·
   N_post). **AlignPre** and **AlignPost** instead store one variable per *neuron*
   — O(N_pre) or O(N_post) — *without approximation*. That same reduction, reused
   on the backward pass, is what makes linear-memory online learning possible. Read
   :doc:`/state/concepts/alignpre-alignpost`.

Recommended reading order
=========================

The pages are written to be read in sequence; each assumes the one before it.

.. grid:: 1 1 2 2
   :gutter: 3

   .. grid-item-card:: 1 · The state paradigm
      :link: /state/concepts/state-paradigm
      :link-type: doc

      Why functional JAX needs explicit ``State``; ``init_all_states``,
      ``environ.context``, and the ``transform`` primitives that drive a model
      without a bare Python loop.

   .. grid-item-card:: 2 · Physical units
      :link: /state/concepts/physical-units
      :link-type: doc

      ``brainunit`` quantities, unit-safe construction, unit-aware initializers,
      and the pitfalls they catch at construction time.

   .. grid-item-card:: 3 · Model anatomy
      :link: /state/concepts/model-anatomy
      :link-type: doc

      The ``Dynamics`` → ``Neuron`` / ``Synapse`` hierarchy, state variables,
      ``update()`` / ``get_spike()``, and how a neuron receives current — the
      contract projections rely on.

   .. grid-item-card:: 4 · AlignPre / AlignPost ★
      :link: /state/concepts/alignpre-alignpost
      :link-type: doc

      **The keystone.** Three memory regimes, the four projection components, an
      exact (not approximate) reduction, and a decision table for choosing between
      the two alignments.

   .. grid-item-card:: 5 · Differentiability
      :link: /state/concepts/differentiability
      :link-type: doc

      The *simulator → trainable* half of the bridge: surrogate gradients and
      backpropagation-through-time via ``transform``.

   .. grid-item-card:: 6 · Online learning
      :link: /state/concepts/online-learning
      :link-type: doc

      The *trainable → scalable* half: linear-memory RTRL enabled by
      neuron-aligned state, and the ``braintrace`` engine.

Where to go next
================

Once the concepts click, put them to work. The :doc:`/state/index`
tutorials and how-to guides apply this vocabulary to concrete modeling and
training tasks, and the :doc:`/apis/state` is the authoritative model lookup.

.. toctree::
   :hidden:
   :maxdepth: 1

   state-paradigm
   physical-units
   model-anatomy
   alignpre-alignpost
   differentiability
   online-learning
