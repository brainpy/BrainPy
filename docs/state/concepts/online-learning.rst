===============
Online learning
===============

*Who it's for: training, at scale.* This is the **trainable → online / scalable**
half of the bridge. The previous page,
:doc:`/state/concepts/differentiability`, showed that ``brainpy.state`` models are
differentiable and trainable with backpropagation-through-time (BPTT). This page
explains what happens when you need to learn *online* — updating as the network
runs, without storing the whole trajectory — and why the
:doc:`/state/concepts/alignpre-alignpost` keystone is exactly what makes that tractable.

Unlike the rest of this spine, this page is **prose, not a runnable notebook**: the
online-learning engine lives in a separate ecosystem package, `braintrace`, so the
snippets below are illustrative and are *not* executed here. Where you want to run
code, follow the pointer to that package at the end.

From offline BPTT to online RTRL
================================

BPTT unrolls the whole simulation, then propagates gradients *backward* through the
unrolled graph. That is fine offline, but it has two costs that matter at scale: you
must store activations for every time step (memory grows with rollout length), and
no weight update is available until the rollout *and* its backward pass finish — so
it is not a model of learning that happens *while* the network runs.

**Real-time recurrent learning (RTRL)** is the forward-mode alternative. Instead of
looking back, it carries the sensitivity of the hidden state to the parameters
*forward* in time, updating that sensitivity every step alongside the dynamics. This
makes learning online and memory-constant in rollout length — but in its naive form
it is brutally expensive. For a network with :math:`H` hidden units, RTRL maintains
the Jacobian of every hidden state with respect to every parameter. The parameter
count itself scales with :math:`H^2`, and propagating its influence brings the
update cost to :math:`\mathcal{O}(H^3)` per step in compute, with an
:math:`\mathcal{O}(H^2)` hidden-state Jacobian to store. For anything beyond a toy
network, that is prohibitive — which is why RTRL has long been considered
impractical for large recurrent models.

Linear-memory RTRL
==================

The breakthrough is that for **spiking** networks built on neuron-aligned synaptic
state, RTRL's cubic/quadratic cost **collapses to linear** in the number of neurons.
This is the result of Wang et al. (2026, *Nature Communications*) — see
*Related work* below.

Two properties combine to make it work, and both are already familiar from the
keystone:

#. **Synaptic hidden state is aligned to a neuron dimension, not a weight
   dimension.** AlignPre and AlignPost store one synaptic variable per *neuron*
   (:math:`\mathcal{O}(N_\text{pre})` or :math:`\mathcal{O}(N_\text{post})`), not one
   per synapse. The sensitivities RTRL must propagate are indexed by that same small
   neuron dimension, so the eligibility trace and the parameter/hidden Jacobians
   shrink accordingly.
#. **Spiking dynamics are sparse and event-driven.** Only neurons that spike
   contribute, so the per-step propagation touches a small, structured slice of the
   state rather than a dense :math:`H \times H` (or larger) object.

The same **superposition** that makes AlignPost's *forward* pass exact — many
presynaptic spikes summing into one merged postsynaptic conductance via
:math:`g \leftarrow g + 1` — is what is reused on the *backward* (sensitivity)
direction. That reuse is the crux: the eligibility trace and the parameter Jacobian
(otherwise :math:`\mathcal{O}(H^3)`) and the hidden-state Jacobian (otherwise
:math:`\mathcal{O}(H^2)`) all reduce to **:math:`\mathcal{O}(H)`** memory. This is a
direct callback to :doc:`/state/concepts/alignpre-alignpost` §9: the alignment that makes
*simulation* memory-efficient is the alignment that makes *learning*
memory-efficient.

.. admonition:: The two-line summary
   :class: note

   Neuron-aligned synaptic state + event-driven sparsity turn RTRL from
   :math:`\mathcal{O}(H^3)` compute / :math:`\mathcal{O}(H^2)` memory into a
   **linear-memory** online learning rule. The keystone alignment is the enabling
   ingredient.

Whole-brain scaling
===================

Linear memory is not merely a constant-factor win — it changes which models can be
*trained online at all*. With per-neuron rather than per-synapse sensitivity, online
learning becomes feasible at the scale of entire connectomes. The headline
demonstration is a **whole-brain, *Drosophila*-scale spiking network** trained with
this method — a regime that dense RTRL could not approach and that BPTT cannot reach
online. This is the payoff the bridging thesis promised: a brain *simulation* at
connectome scale that is simultaneously a *trainable* brain-inspired model, on one
substrate. See the paper linked under *Related work* for the full results.

The engine: ``braintrace``
==========================

The linear-memory algorithm ships as a dedicated package in the BrainX ecosystem,
separate from ``brainpy.state`` so that the modeling layer stays focused on building
and simulating networks while the learning engine evolves independently.

.. admonition:: A note on names (read this to avoid confusion)
   :class: tip

   The *simulator* is **BrainPy**, and its modern state-based API is
   **brainpy.state** (this documentation). The *online-learning engine* was
   introduced under the name **BrainScale** in the preprint, and is published as
   **BrainTrace** — the installable package is **``braintrace``**. If you read
   "BrainScale" in the preprint and "BrainTrace"/``braintrace`` in the released
   software, they are the same engine.

In practice you build the network exactly as in this spine — neurons, synapses, and
AlignPre/AlignPost projections — and hand it to ``braintrace`` to attach a
linear-memory online learning rule. The following is **illustrative only** (not
executed here; install and consult ``braintrace`` for the real, current API):

.. code-block:: python

   # ILLUSTRATIVE — see the braintrace package for the actual API.
   import brainpy
   import braintrace            # the BrainScale -> BrainTrace online-learning engine

   net = MySpikingNetwork()     # built from brainpy.state neurons + AlignPre/AlignPost
   learner = braintrace.online_learner(net)   # attach linear-memory RTRL

   for x, y in stream:          # learn online, step by step, in constant memory
       learner.step(x, y)

Because the learning rule rides on the *same* neuron-aligned state your simulation
already uses, moving from "I can simulate this network" to "I can train it online at
scale" does not require re-expressing the model.

Related work
============

The linear-memory online learning method described on this page is published as:

- Wang, C., et al. (2026). *Model-agnostic linear-memory online learning in spiking
  neural networks.* Nature Communications.
  `doi:10.1038/s41467-026-68453-w <https://doi.org/10.1038/s41467-026-68453-w>`__.

The earlier preprint introduces the same alignment-and-dimension framing under the
name *BrainScale* (bioRxiv 2024.09.24.614728). The keystone projection design it
builds on is from Wang et al. (2024, ICLR) — see
:doc:`/state/concepts/alignpre-alignpost`.

.. note::

   This 2026 paper is cited here as **related work** because it motivates the
   online-learning capability. The "cite this software" entry for ``brainpy.state``
   itself remains the eLife BrainPy (2023) and ICLR 2024 references — see
   `Citing BrainPy <https://github.com/brainpy/BrainPy#citing>`_.

See also
========

- :doc:`/state/concepts/alignpre-alignpost` — the keystone; §9 makes the simulation ↔
  learning connection this page completes.
- :doc:`/state/concepts/differentiability` — the offline BPTT half of the bridge.
- the `BrainX ecosystem <https://brainx.chaobrain.com/ecosystem/>`_ — where ``braintrace`` sits among ``brainstate``,
  ``brainunit``, ``brainevent``, and ``braintools``.
