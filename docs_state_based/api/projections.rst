Projections
===========

Connect neural populations with the Comm-Syn-Out architecture.

.. currentmodule:: brainpy

Projection Classes
------------------

AlignPostProj
~~~~~~~~~~~~~

.. class:: AlignPostProj(comm, syn, out, post, **kwargs)

   Standard projection aligning synaptic states with postsynaptic neurons.

   **Parameters:**

   - ``comm`` - Communication layer (connectivity)
   - ``syn`` - Synapse dynamics
   - ``out`` - Output computation
   - ``post`` - Postsynaptic neuron population

   **Example:**

   .. code-block:: python

      proj = bp.AlignPostProj(
          comm=brainstate.nn.EventFixedProb(100, 50, prob=0.1, weight=0.5*u.mS),
          syn=bp.Expon.desc(50, tau=5*u.ms),
          out=bp.COBA.desc(E=0*u.mV),
          post=post_neurons
      )

      # Usage
      pre_spikes = pre_neurons.get_spike()
      proj(pre_spikes)

AlignPreProj
~~~~~~~~~~~~

.. class:: AlignPreProj(comm, syn, out, post, **kwargs)

   Projection aligning synaptic states with presynaptic neurons.

   Used for certain learning rules that require presynaptic alignment.

Communication Layers
--------------------

From ``brainstate.nn``:

EventFixedProb
~~~~~~~~~~~~~~

.. code-block:: python

   comm = brainstate.nn.EventFixedProb(
       pre_size,
       post_size,
       prob=0.1,      # Connection probability
       weight=0.5*u.mS  # Synaptic weight
   )

Sparse connectivity with fixed connection probability.

EventAll2All
~~~~~~~~~~~~

.. code-block:: python

   comm = brainstate.nn.EventAll2All(
       pre_size,
       post_size,
       weight=0.5*u.mS
   )

All-to-all connectivity (event-driven).

EventOne2One
~~~~~~~~~~~~

.. code-block:: python

   comm = brainstate.nn.EventOne2One(
       size,
       weight=0.5*u.mS
   )

One-to-one connections (same size populations).

Linear
~~~~~~

.. code-block:: python

   comm = brainstate.nn.Linear(
       in_size,
       out_size,
       w_init=brainstate.init.KaimingNormal()
   )

Dense linear transformation (for small networks).

Complete Examples
-----------------

**E → E Excitatory:**

.. code-block:: python

   E2E = bp.AlignPostProj(
       comm=brainstate.nn.EventFixedProb(n_exc, n_exc, prob=0.02, weight=0.6*u.mS),
       syn=bp.AMPA.desc(n_exc, tau=2*u.ms),
       out=bp.COBA.desc(E=0*u.mV),
       post=E_neurons
   )

**I → E Inhibitory:**

.. code-block:: python

   I2E = bp.AlignPostProj(
       comm=brainstate.nn.EventFixedProb(n_inh, n_exc, prob=0.02, weight=6.7*u.mS),
       syn=bp.GABAa.desc(n_exc, tau=6*u.ms),
       out=bp.COBA.desc(E=-80*u.mV),
       post=E_neurons
   )

**Multi-timescale (AMPA + NMDA):**

.. code-block:: python

   # Fast AMPA
   ampa_proj = bp.AlignPostProj(
       comm=brainstate.nn.EventFixedProb(n_pre, n_post, prob=0.1, weight=0.3*u.mS),
       syn=bp.AMPA.desc(n_post, tau=2*u.ms),
       out=bp.COBA.desc(E=0*u.mV),
       post=post_neurons
   )

   # Slow NMDA
   nmda_proj = bp.AlignPostProj(
       comm=brainstate.nn.EventFixedProb(n_pre, n_post, prob=0.1, weight=0.3*u.mS),
       syn=bp.NMDA.desc(n_post, tau_decay=100*u.ms),
       out=bp.MgBlock.desc(E=0*u.mV),
       post=post_neurons
   )

See Also
--------

- :doc:`../core-concepts/projections` - Complete projection guide
- :doc:`../tutorials/basic/03-network-connections` - Network tutorial
- :doc:`neurons` - Neuron models
- :doc:`synapses` - Synapse models
