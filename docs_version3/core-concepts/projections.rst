Projections: Connecting Neural Populations
==========================================

Projections are BrainPy's mechanism for connecting neural populations. They implement the **Communication-Synapse-Output (Comm-Syn-Out)** architecture, which separates connectivity, synaptic dynamics, and output computation into modular components.

This guide provides a comprehensive understanding of projections in BrainPy 3.0.

.. contents:: Table of Contents
   :local:
   :depth: 2

Overview
--------

What are Projections?
~~~~~~~~~~~~~~~~~~~~~

A **projection** connects a presynaptic population to a postsynaptic population through:

1. **Communication (Comm)**: How spikes propagate through connections
2. **Synapse (Syn)**: Temporal filtering and synaptic dynamics
3. **Output (Out)**: How synaptic currents affect postsynaptic neurons

**Key benefits:**

- Modular design (swap components independently)
- Biologically realistic (separate connectivity and dynamics)
- Efficient (optimized sparse operations)
- Flexible (combine components in different ways)

The Comm-Syn-Out Architecture
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: text

   Presynaptic       Communication        Synapse          Output        Postsynaptic
   Population    ──►  (Connectivity)  ──►  (Dynamics)  ──►  (Current) ──►  Population

   Spikes        ──►  Weight matrix   ──►  g(t)        ──►  I_syn     ──►  Neurons
                      Sparse/Dense         Expon/Alpha     CUBA/COBA

**Flow:**

1. Presynaptic spikes arrive
2. Communication: Spikes propagate through connectivity matrix
3. Synapse: Temporal dynamics filter the signal
4. Output: Convert to current/conductance
5. Postsynaptic neurons receive input

Types of Projections
~~~~~~~~~~~~~~~~~~~~~

BrainPy provides two main projection types:

**AlignPostProj**
   - Align synaptic states with postsynaptic neurons
   - Most common for standard neural networks
   - Efficient memory layout

**AlignPreProj**
   - Align synaptic states with presynaptic neurons
   - Useful for certain learning rules
   - Different memory organization

For most use cases, use ``AlignPostProj``.

Communication Layer
-------------------

The Communication layer defines **how spikes propagate** through connections.

Dense Connectivity
~~~~~~~~~~~~~~~~~~

All neurons potentially connected (though weights may be zero).

**Use case:** Small networks, fully connected layers

.. code-block:: python

   import brainpy as bp
   import brainstate
   import brainunit as u

   # Dense linear transformation
   comm = brainstate.nn.Linear(
       in_size=100,    # Presynaptic neurons
       out_size=50,    # Postsynaptic neurons
       w_init=brainstate.init.KaimingNormal(),
       b_init=None     # No bias for synapses
   )

**Characteristics:**

- Memory: O(n_pre × n_post)
- Computation: Full matrix multiplication
- Best for: Small networks, fully connected architectures

Sparse Connectivity
~~~~~~~~~~~~~~~~~~~

Only a subset of connections exist (biologically realistic).

**Use case:** Large networks, biological connectivity patterns

Event-Based Fixed Probability
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Connect neurons with fixed probability.

.. code-block:: python

   # Sparse random connectivity (2% connection probability)
   comm = brainstate.nn.EventFixedProb(
       pre_size=1000,
       post_size=800,
       prob=0.02,              # 2% connectivity
       weight=0.5 * u.mS       # Synaptic weight
   )

**Characteristics:**

- Memory: O(n_pre × n_post × prob)
- Computation: Only active connections
- Best for: Large-scale networks, biological models

Event-Based All-to-All
^^^^^^^^^^^^^^^^^^^^^^^

All neurons connected (but stored sparsely).

.. code-block:: python

   # All-to-all sparse (event-driven)
   comm = brainstate.nn.EventAll2All(
       pre_size=100,
       post_size=100,
       weight=0.3 * u.mS
   )

Event-Based One-to-One
^^^^^^^^^^^^^^^^^^^^^^^

One-to-one mapping (same size populations).

.. code-block:: python

   # One-to-one connections
   comm = brainstate.nn.EventOne2One(
       size=100,
       weight=1.0 * u.mS
   )

**Use case:** Feedforward pathways, identity mappings

Comparison Table
~~~~~~~~~~~~~~~~

.. list-table:: Communication Layer Options
   :header-rows: 1
   :widths: 20 20 20 20 20

   * - Type
     - Memory
     - Speed
     - Use Case
     - Example
   * - Linear (Dense)
     - High (O(n²))
     - Fast (optimized)
     - Small networks
     - Fully connected
   * - EventFixedProb
     - Low (O(n²p))
     - Very fast
     - Large networks
     - Cortical connectivity
   * - EventAll2All
     - Medium
     - Fast
     - Medium networks
     - Recurrent layers
   * - EventOne2One
     - Minimal (O(n))
     - Fastest
     - Feedforward
     - Sensory pathways

Synapse Layer
-------------

The Synapse layer defines **temporal dynamics** of synaptic transmission.

Exponential Synapse
~~~~~~~~~~~~~~~~~~~

Single exponential decay (most common).

**Dynamics:**

.. math::

   \tau \frac{dg}{dt} = -g + \sum_k \delta(t - t_k)

**Implementation:**

.. code-block:: python

   # Exponential synapse with 5ms time constant
   syn = bp.Expon.desc(
       size=100,           # Postsynaptic population size
       tau=5.0 * u.ms      # Decay time constant
   )

**Characteristics:**

- Single time constant
- Fast computation
- Good for most applications

**When to use:** Default choice for most models

Alpha Synapse
~~~~~~~~~~~~~

Dual exponential with rise and decay.

**Dynamics:**

.. math::

   \tau \frac{dg}{dt} = -g + h

   \tau \frac{dh}{dt} = -h + \sum_k \delta(t - t_k)

**Implementation:**

.. code-block:: python

   # Alpha synapse
   syn = bp.Alpha.desc(
       size=100,
       tau=10.0 * u.ms     # Characteristic time
   )

**Characteristics:**

- Realistic rise time
- Smoother response
- Slightly slower computation

**When to use:** When rise time matters, more biological realism

NMDA Synapse
~~~~~~~~~~~~

Voltage-dependent NMDA receptors.

**Dynamics:**

.. math::

   g_{NMDA} = \frac{g}{1 + \eta [Mg^{2+}] e^{-\gamma V}}

**Implementation:**

.. code-block:: python

   # NMDA receptor
   syn = bp.NMDA.desc(
       size=100,
       tau_decay=100.0 * u.ms,    # Slow decay
       tau_rise=2.0 * u.ms,       # Fast rise
       a=0.5 / u.mM,              # Mg²⁺ sensitivity
       cc_Mg=1.2 * u.mM           # Mg²⁺ concentration
   )

**Characteristics:**

- Voltage-dependent
- Slow kinetics
- Important for plasticity

**When to use:** Long-term potentiation, working memory models

AMPA Synapse
~~~~~~~~~~~~

Fast glutamatergic transmission.

.. code-block:: python

   # AMPA receptor (fast excitation)
   syn = bp.AMPA.desc(
       size=100,
       tau=2.0 * u.ms      # Fast decay (~2ms)
   )

**When to use:** Fast excitatory transmission

GABA Synapse
~~~~~~~~~~~~

Inhibitory transmission.

**GABAa (fast):**

.. code-block:: python

   # GABAa receptor (fast inhibition)
   syn = bp.GABAa.desc(
       size=100,
       tau=6.0 * u.ms      # ~6ms decay
   )

**GABAb (slow):**

.. code-block:: python

   # GABAb receptor (slow inhibition)
   syn = bp.GABAb.desc(
       size=100,
       tau_decay=150.0 * u.ms,    # Very slow
       tau_rise=3.5 * u.ms
   )

**When to use:**
- GABAa: Fast inhibition, cortical networks
- GABAb: Slow inhibition, rhythm generation

Custom Synapses
~~~~~~~~~~~~~~~

Create custom synaptic dynamics by subclassing ``Synapse``.

.. code-block:: python

   class DoubleExpSynapse(bp.Synapse):
       """Custom synapse with two time constants."""

       def __init__(self, size, tau_fast=2*u.ms, tau_slow=10*u.ms, **kwargs):
           super().__init__(size, **kwargs)
           self.tau_fast = tau_fast
           self.tau_slow = tau_slow

           # State variables
           self.g_fast = brainstate.ShortTermState(jnp.zeros(size))
           self.g_slow = brainstate.ShortTermState(jnp.zeros(size))

       def reset_state(self, batch_size=None):
           shape = self.size if batch_size is None else (batch_size, self.size)
           self.g_fast.value = jnp.zeros(shape)
           self.g_slow.value = jnp.zeros(shape)

       def update(self, x):
           dt = brainstate.environ.get_dt()

           # Fast component
           dg_fast = -self.g_fast.value / self.tau_fast.to_decimal(u.ms)
           self.g_fast.value += dg_fast * dt.to_decimal(u.ms) + x * 0.7

           # Slow component
           dg_slow = -self.g_slow.value / self.tau_slow.to_decimal(u.ms)
           self.g_slow.value += dg_slow * dt.to_decimal(u.ms) + x * 0.3

           return self.g_fast.value + self.g_slow.value

Output Layer
------------

The Output layer defines **how synaptic conductance affects neurons**.

CUBA (Current-Based)
~~~~~~~~~~~~~~~~~~~~

Synaptic conductance directly becomes current.

**Model:**

.. math::

   I_{syn} = g_{syn}

**Implementation:**

.. code-block:: python

   # Current-based output
   out = bp.CUBA.desc()

**Characteristics:**

- Simple and fast
- No voltage dependence
- Good for rate-based models

**When to use:**
- Abstract models
- When voltage dependence not important
- Faster computation needed

COBA (Conductance-Based)
~~~~~~~~~~~~~~~~~~~~~~~~~

Synaptic conductance with reversal potential.

**Model:**

.. math::

   I_{syn} = g_{syn} (E_{syn} - V_{post})

**Implementation:**

.. code-block:: python

   # Excitatory conductance-based
   out_exc = bp.COBA.desc(E=0.0 * u.mV)

   # Inhibitory conductance-based
   out_inh = bp.COBA.desc(E=-80.0 * u.mV)

**Characteristics:**

- Voltage-dependent
- Biologically realistic
- Self-limiting (saturates near reversal)

**When to use:**
- Biologically detailed models
- When voltage dependence matters
- Shunting inhibition needed

MgBlock (NMDA)
~~~~~~~~~~~~~~

Voltage-dependent magnesium block for NMDA.

.. code-block:: python

   # NMDA with Mg²⁺ block
   out_nmda = bp.MgBlock.desc(
       E=0.0 * u.mV,
       cc_Mg=1.2 * u.mM,
       alpha=0.062 / u.mV,
       beta=3.57
   )

**When to use:** NMDA receptors, voltage-dependent plasticity

Complete Projection Examples
-----------------------------

Example 1: Simple Feedforward
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import brainpy as bp
   import brainstate
   import brainunit as u

   # Create populations
   pre = bp.LIF(100, V_rest=-65*u.mV, V_th=-50*u.mV, tau=10*u.ms)
   post = bp.LIF(50, V_rest=-65*u.mV, V_th=-50*u.mV, tau=10*u.ms)

   # Create projection: 100 → 50 neurons
   proj = bp.AlignPostProj(
       comm=brainstate.nn.EventFixedProb(
           pre_size=100,
           post_size=50,
           prob=0.1,              # 10% connectivity
           weight=0.5 * u.mS
       ),
       syn=bp.Expon.desc(
           size=50,               # Postsynaptic size
           tau=5.0 * u.ms
       ),
       out=bp.CUBA.desc(),
       post=post                  # Postsynaptic population
   )

   # Initialize
   brainstate.nn.init_all_states([pre, post, proj])

   # Simulate
   def step(inp):
       # Get presynaptic spikes
       pre_spikes = pre.get_spike()

       # Update projection
       proj(pre_spikes)

       # Update neurons
       pre(inp)
       post(0.0 * u.nA)  # Projection provides input

       return pre.get_spike(), post.get_spike()

Example 2: Excitatory-Inhibitory Network
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   class EINetwork(brainstate.nn.Module):
       def __init__(self, n_exc=800, n_inh=200):
           super().__init__()

           # Populations
           self.E = bp.LIF(n_exc, V_rest=-65*u.mV, V_th=-50*u.mV, tau=15*u.ms)
           self.I = bp.LIF(n_inh, V_rest=-65*u.mV, V_th=-50*u.mV, tau=10*u.ms)

           # E → E projection (AMPA, excitatory)
           self.E2E = bp.AlignPostProj(
               comm=brainstate.nn.EventFixedProb(n_exc, n_exc, prob=0.02, weight=0.6*u.mS),
               syn=bp.AMPA.desc(n_exc, tau=2.0*u.ms),
               out=bp.COBA.desc(E=0.0*u.mV),
               post=self.E
           )

           # E → I projection (AMPA, excitatory)
           self.E2I = bp.AlignPostProj(
               comm=brainstate.nn.EventFixedProb(n_exc, n_inh, prob=0.02, weight=0.6*u.mS),
               syn=bp.AMPA.desc(n_inh, tau=2.0*u.ms),
               out=bp.COBA.desc(E=0.0*u.mV),
               post=self.I
           )

           # I → E projection (GABAa, inhibitory)
           self.I2E = bp.AlignPostProj(
               comm=brainstate.nn.EventFixedProb(n_inh, n_exc, prob=0.02, weight=6.7*u.mS),
               syn=bp.GABAa.desc(n_exc, tau=6.0*u.ms),
               out=bp.COBA.desc(E=-80.0*u.mV),
               post=self.E
           )

           # I → I projection (GABAa, inhibitory)
           self.I2I = bp.AlignPostProj(
               comm=brainstate.nn.EventFixedProb(n_inh, n_inh, prob=0.02, weight=6.7*u.mS),
               syn=bp.GABAa.desc(n_inh, tau=6.0*u.ms),
               out=bp.COBA.desc(E=-80.0*u.mV),
               post=self.I
           )

       def update(self, inp_e, inp_i):
           # Get spikes BEFORE updating neurons
           spk_e = self.E.get_spike()
           spk_i = self.I.get_spike()

           # Update all projections
           self.E2E(spk_e)
           self.E2I(spk_e)
           self.I2E(spk_i)
           self.I2I(spk_i)

           # Update neurons (projections provide synaptic input)
           self.E(inp_e)
           self.I(inp_i)

           return spk_e, spk_i

Example 3: Multi-Timescale Synapses
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Combine AMPA (fast) and NMDA (slow) for realistic excitation.

.. code-block:: python

   class DualExcitatory(brainstate.nn.Module):
       """E → E with both AMPA and NMDA."""

       def __init__(self, n_pre=100, n_post=100):
           super().__init__()

           self.post = bp.LIF(n_post, V_rest=-65*u.mV, V_th=-50*u.mV, tau=10*u.ms)

           # Fast AMPA component
           self.ampa_proj = bp.AlignPostProj(
               comm=brainstate.nn.EventFixedProb(n_pre, n_post, prob=0.1, weight=0.3*u.mS),
               syn=bp.AMPA.desc(n_post, tau=2.0*u.ms),
               out=bp.COBA.desc(E=0.0*u.mV),
               post=self.post
           )

           # Slow NMDA component
           self.nmda_proj = bp.AlignPostProj(
               comm=brainstate.nn.EventFixedProb(n_pre, n_post, prob=0.1, weight=0.3*u.mS),
               syn=bp.NMDA.desc(n_post, tau_decay=100.0*u.ms, tau_rise=2.0*u.ms),
               out=bp.MgBlock.desc(E=0.0*u.mV, cc_Mg=1.2*u.mM),
               post=self.post
           )

       def update(self, pre_spikes):
           # Both projections share same presynaptic spikes
           self.ampa_proj(pre_spikes)
           self.nmda_proj(pre_spikes)

           # Post receives combined input
           self.post(0.0 * u.nA)

           return self.post.get_spike()

Advanced Topics
---------------

Delay Projections
~~~~~~~~~~~~~~~~~

Add synaptic delays to projections.

.. code-block:: python

   # Projection with 5ms synaptic delay
   proj_delayed = bp.AlignPostProj(
       comm=brainstate.nn.EventFixedProb(100, 100, prob=0.1, weight=0.5*u.mS),
       syn=bp.Expon.desc(100, tau=5.0*u.ms),
       out=bp.CUBA.desc(),
       post=post_neurons,
       delay=5.0 * u.ms  # Synaptic delay
   )

**Use cases:**
- Biologically realistic transmission delays
- Axonal conduction delays
- Synchronization studies

Heterogeneous Weights
~~~~~~~~~~~~~~~~~~~~~~

Different weights for different connections.

.. code-block:: python

   import jax.numpy as jnp

   # Custom weight matrix
   n_pre, n_post = 100, 50
   weights = jnp.abs(brainstate.random.randn(n_pre, n_post)) * 0.5 * u.mS

   # Sparse with heterogeneous weights
   comm = brainstate.nn.EventJitFPHomoLinear(
       num_in=n_pre,
       num_out=n_post,
       prob=0.1,
       weight=weights  # Heterogeneous
   )

Learning Synapses
~~~~~~~~~~~~~~~~~

Combine with plasticity (see :doc:`../tutorials/advanced/06-synaptic-plasticity`).

.. code-block:: python

   # Projection with learnable weights
   class PlasticProjection(brainstate.nn.Module):
       def __init__(self, n_pre, n_post):
           super().__init__()

           # Initialize weights as parameters
           self.weights = brainstate.ParamState(
               jnp.ones((n_pre, n_post)) * 0.5 * u.mS
           )

           self.proj = bp.AlignPostProj(
               comm=CustomComm(self.weights),  # Use learnable weights
               syn=bp.Expon.desc(n_post, tau=5.0*u.ms),
               out=bp.CUBA.desc(),
               post=post_neurons
           )

       def update_weights(self, dw):
           """Update weights based on learning rule."""
           self.weights.value += dw

Best Practices
--------------

Choosing Communication Type
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Use EventFixedProb when:**
- Large networks (>1000 neurons)
- Sparse connectivity (<10%)
- Biological models

**Use Linear when:**
- Small networks (<1000 neurons)
- Fully connected layers
- Training with gradients

**Use EventOne2One when:**
- Same-size populations
- Feedforward pathways
- Identity mappings

Choosing Synapse Type
~~~~~~~~~~~~~~~~~~~~~~

**Use Expon when:**
- Default choice for most models
- Fast computation needed
- Simple dynamics sufficient

**Use Alpha when:**
- Rise time is important
- More biological realism
- Smoother responses

**Use AMPA/NMDA/GABA when:**
- Specific receptor types matter
- Pharmacological studies
- Detailed biological models

Choosing Output Type
~~~~~~~~~~~~~~~~~~~~~

**Use CUBA when:**
- Abstract models
- Training with gradients
- Speed is critical

**Use COBA when:**
- Biological realism needed
- Voltage dependence matters
- Shunting inhibition required

Performance Tips
~~~~~~~~~~~~~~~~

1. **Sparse over Dense:** Use sparse connectivity for large networks
2. **Batch initialization:** Initialize all modules together
3. **JIT compile:** Wrap simulation loop with ``@brainstate.compile.jit``
4. **Appropriate precision:** Use float32 unless high precision needed
5. **Minimize communication:** Group projections with same connectivity

Common Patterns
~~~~~~~~~~~~~~~

**Pattern 1: Dale's Principle**

Neurons are either excitatory OR inhibitory (not both).

.. code-block:: python

   # Separate excitatory and inhibitory populations
   E = bp.LIF(800, ...)  # Excitatory
   I = bp.LIF(200, ...)  # Inhibitory

   # E always excitatory (E=0mV)
   # I always inhibitory (E=-80mV)

**Pattern 2: Balanced Networks**

Excitation balanced by inhibition.

.. code-block:: python

   # Strong inhibition to balance excitation
   w_exc = 0.6 * u.mS
   w_inh = 6.7 * u.mS  # ~10× stronger

   # More E neurons than I (4:1 ratio)
   n_exc = 800
   n_inh = 200

**Pattern 3: Recurrent Loops**

Self-connections for persistent activity.

.. code-block:: python

   # Excitatory recurrence (working memory)
   E2E = bp.AlignPostProj(
       comm=brainstate.nn.EventFixedProb(n_exc, n_exc, prob=0.02, weight=0.5*u.mS),
       syn=bp.Expon.desc(n_exc, tau=5*u.ms),
       out=bp.COBA.desc(E=0*u.mV),
       post=E
   )

Troubleshooting
---------------

Issue: Spikes not propagating
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Symptoms:** Postsynaptic neurons don't receive input

**Solutions:**

1. Check spike timing: Call ``get_spike()`` BEFORE updating
2. Verify connectivity: Check ``prob`` and ``weight``
3. Check update order: Projections before neurons

.. code-block:: python

   # CORRECT order
   spk = pre.get_spike()  # Get spikes from previous step
   proj(spk)               # Update projection
   pre(inp)                # Update neurons

   # WRONG order
   pre(inp)                # Update first
   spk = pre.get_spike()  # Then get spikes (too late!)
   proj(spk)

Issue: Network silent or exploding
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Symptoms:** No activity or runaway firing

**Solutions:**

1. Balance E/I weights (I should be ~10× stronger)
2. Check reversal potentials (E=0mV, I=-80mV)
3. Verify threshold and reset values
4. Add external input

.. code-block:: python

   # Balanced weights
   w_exc = 0.5 * u.mS
   w_inh = 5.0 * u.mS  # Strong inhibition

   # Proper reversal potentials
   out_exc = bp.COBA.desc(E=0.0 * u.mV)
   out_inh = bp.COBA.desc(E=-80.0 * u.mV)

Issue: Slow simulation
~~~~~~~~~~~~~~~~~~~~~~

**Solutions:**

1. Use sparse connectivity (EventFixedProb)
2. Use JIT compilation
3. Use CUBA instead of COBA (if appropriate)
4. Reduce connectivity or neurons

.. code-block:: python

   # Fast configuration
   @brainstate.compile.jit
   def simulate_step(net, inp):
       return net(inp)

   # Sparse connectivity
   comm = brainstate.nn.EventFixedProb(1000, 1000, prob=0.02, ...)

Further Reading
---------------

- :doc:`../tutorials/basic/03-network-connections` - Network connections tutorial
- :doc:`architecture` - Overall BrainPy architecture
- :doc:`synapses` - Detailed synapse models
- :doc:`../tutorials/advanced/06-synaptic-plasticity` - Learning in projections
- :doc:`../tutorials/advanced/07-large-scale-simulations` - Scaling projections

Summary
-------

**Key takeaways:**

✅ Projections use Comm-Syn-Out architecture

✅ Communication: Dense (Linear) or Sparse (EventFixedProb)

✅ Synapse: Temporal dynamics (Expon, Alpha, AMPA, GABA, NMDA)

✅ Output: Current-based (CUBA) or Conductance-based (COBA)

✅ Choose components based on scale, realism, and performance needs

✅ Follow Dale's principle and balanced E/I patterns

✅ Get spikes BEFORE updating for correct propagation

**Quick reference:**

.. code-block:: python

   # Standard projection template
   proj = bp.AlignPostProj(
       comm=brainstate.nn.EventFixedProb(n_pre, n_post, prob=0.1, weight=0.5*u.mS),
       syn=bp.Expon.desc(n_post, tau=5.0*u.ms),
       out=bp.COBA.desc(E=0.0*u.mV),
       post=post_neurons
   )

   # Usage in network
   def update(self):
       spk = self.pre.get_spike()  # Get spikes first
       self.proj(spk)               # Update projection
       self.pre(inp)                # Update neurons
       self.post(0*u.nA)
