How to Create Custom Components
================================

This guide shows you how to create custom neurons, synapses, and other components in BrainPy.

.. contents:: Table of Contents
   :local:
   :depth: 2

Quick Start
-----------

**Custom neuron template:**

.. code-block:: python

   import brainpy as bp
   import brainstate
   import brainunit as u
   import jax.numpy as jnp

   class CustomNeuron(bp.Neuron):
       def __init__(self, size, **kwargs):
           super().__init__(size, **kwargs)

           # Parameters
           self.tau = 10.0 * u.ms
           self.V_th = -50.0 * u.mV

           # States
           self.V = brainstate.ShortTermState(jnp.zeros(size))
           self.spike = brainstate.ShortTermState(jnp.zeros(size))

       def reset_state(self, batch_size=None):
           shape = self.size if batch_size is None else (batch_size, self.size)
           self.V.value = jnp.zeros(shape)
           self.spike.value = jnp.zeros(shape)

       def update(self, x):
           dt = brainstate.environ.get_dt()

           # Dynamics
           dV = -self.V.value / self.tau.to_decimal(u.ms) + x.to_decimal(u.nA)
           self.V.value += dV * dt.to_decimal(u.ms)

           # Spike generation
           self.spike.value = (self.V.value >= self.V_th.to_decimal(u.mV)).astype(float)

           # Reset
           self.V.value = jnp.where(
               self.spike.value > 0,
               0.0,  # Reset voltage
               self.V.value
           )

           return self.V.value

       def get_spike(self):
           return self.spike.value

Custom Neurons
--------------

Example 1: Adaptive LIF
~~~~~~~~~~~~~~~~~~~~~~~

**LIF with spike-frequency adaptation:**

.. code-block:: python

   class AdaptiveLIF(bp.Neuron):
       """LIF neuron with adaptation current."""

       def __init__(self, size, tau=10*u.ms, tau_w=100*u.ms,
                    V_th=-50*u.mV, V_reset=-65*u.mV, a=0.1*u.nA,
                    b=0.5*u.nA, **kwargs):
           super().__init__(size, **kwargs)

           self.tau = tau
           self.tau_w = tau_w
           self.V_th = V_th
           self.V_reset = V_reset
           self.a = a  # Adaptation coupling
           self.b = b  # Spike-triggered adaptation

           # States
           self.V = brainstate.ShortTermState(jnp.ones(size) * V_reset.to_decimal(u.mV))
           self.w = brainstate.ShortTermState(jnp.zeros(size))  # Adaptation current
           self.spike = brainstate.ShortTermState(jnp.zeros(size))

       def reset_state(self, batch_size=None):
           shape = self.size if batch_size is None else (batch_size, self.size)
           self.V.value = jnp.ones(shape) * self.V_reset.to_decimal(u.mV)
           self.w.value = jnp.zeros(shape)
           self.spike.value = jnp.zeros(shape)

       def update(self, I_ext):
           dt = brainstate.environ.get_dt()

           # Membrane potential dynamics
           dV = (-self.V.value + self.V_reset.to_decimal(u.mV) + I_ext.to_decimal(u.nA) - self.w.value) / self.tau.to_decimal(u.ms)
           self.V.value += dV * dt.to_decimal(u.ms)

           # Adaptation dynamics
           dw = (self.a.to_decimal(u.nA) * (self.V.value - self.V_reset.to_decimal(u.mV)) - self.w.value) / self.tau_w.to_decimal(u.ms)
           self.w.value += dw * dt.to_decimal(u.ms)

           # Spike generation
           self.spike.value = (self.V.value >= self.V_th.to_decimal(u.mV)).astype(float)

           # Reset and adaptation jump
           self.V.value = jnp.where(
               self.spike.value > 0,
               self.V_reset.to_decimal(u.mV),
               self.V.value
           )
           self.w.value += self.spike.value * self.b.to_decimal(u.nA)

           return self.V.value

       def get_spike(self):
           return self.spike.value

Example 2: Izhikevich Neuron
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   class Izhikevich(bp.Neuron):
       """Izhikevich neuron model."""

       def __init__(self, size, a=0.02, b=0.2, c=-65*u.mV, d=8*u.mV, **kwargs):
           super().__init__(size, **kwargs)

           self.a = a
           self.b = b
           self.c = c
           self.d = d

           # States
           self.V = brainstate.ShortTermState(jnp.ones(size) * c.to_decimal(u.mV))
           self.u = brainstate.ShortTermState(jnp.zeros(size))
           self.spike = brainstate.ShortTermState(jnp.zeros(size))

       def reset_state(self, batch_size=None):
           shape = self.size if batch_size is None else (batch_size, self.size)
           self.V.value = jnp.ones(shape) * self.c.to_decimal(u.mV)
           self.u.value = jnp.zeros(shape)
           self.spike.value = jnp.zeros(shape)

       def update(self, I):
           dt = brainstate.environ.get_dt()

           # Izhikevich dynamics
           dV = (0.04 * self.V.value**2 + 5 * self.V.value + 140 - self.u.value + I.to_decimal(u.nA))
           du = self.a * (self.b * self.V.value - self.u.value)

           self.V.value += dV * dt.to_decimal(u.ms)
           self.u.value += du * dt.to_decimal(u.ms)

           # Spike and reset
           self.spike.value = (self.V.value >= 30).astype(float)
           self.V.value = jnp.where(self.spike.value > 0, self.c.to_decimal(u.mV), self.V.value)
           self.u.value = jnp.where(self.spike.value > 0, self.u.value + self.d.to_decimal(u.mV), self.u.value)

           return self.V.value

       def get_spike(self):
           return self.spike.value

Custom Synapses
---------------

Example: Biexponential Synapse
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   class BiexponentialSynapse(bp.Synapse):
       """Synapse with separate rise and decay."""

       def __init__(self, size, tau_rise=1*u.ms, tau_decay=5*u.ms, **kwargs):
           super().__init__(size, **kwargs)

           self.tau_rise = tau_rise
           self.tau_decay = tau_decay

           # States
           self.h = brainstate.ShortTermState(jnp.zeros(size))  # Rising phase
           self.g = brainstate.ShortTermState(jnp.zeros(size))  # Decaying phase

       def reset_state(self, batch_size=None):
           shape = self.size if batch_size is None else (batch_size, self.size)
           self.h.value = jnp.zeros(shape)
           self.g.value = jnp.zeros(shape)

       def update(self, x):
           dt = brainstate.environ.get_dt()

           # Two-stage dynamics
           dh = -self.h.value / self.tau_rise.to_decimal(u.ms) + x
           dg = -self.g.value / self.tau_decay.to_decimal(u.ms) + self.h.value

           self.h.value += dh * dt.to_decimal(u.ms)
           self.g.value += dg * dt.to_decimal(u.ms)

           return self.g.value

Example: NMDA Synapse
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   class NMDASynapse(bp.Synapse):
       """NMDA receptor with voltage dependence."""

       def __init__(self, size, tau=100*u.ms, a=0.5/u.mM, Mg=1.0*u.mM, **kwargs):
           super().__init__(size, **kwargs)

           self.tau = tau
           self.a = a
           self.Mg = Mg

           self.g = brainstate.ShortTermState(jnp.zeros(size))

       def reset_state(self, batch_size=None):
           shape = self.size if batch_size is None else (batch_size, self.size)
           self.g.value = jnp.zeros(shape)

       def update(self, x, V_post=None):
           """Update with optional postsynaptic voltage."""
           dt = brainstate.environ.get_dt()

           # Conductance dynamics
           dg = -self.g.value / self.tau.to_decimal(u.ms) + x
           self.g.value += dg * dt.to_decimal(u.ms)

           # Voltage-dependent magnesium block
           if V_post is not None:
               mg_block = 1 / (1 + self.Mg.to_decimal(u.mM) * self.a.to_decimal(1/u.mM) * jnp.exp(-0.062 * V_post.to_decimal(u.mV)))
               return self.g.value * mg_block
           else:
               return self.g.value

Custom Learning Rules
---------------------

Example: Simplified STDP
~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   class SimpleSTDP(brainstate.nn.Module):
       """Simplified STDP learning rule."""

       def __init__(self, n_pre, n_post, A_plus=0.01, A_minus=0.01,
                    tau_plus=20*u.ms, tau_minus=20*u.ms):
           super().__init__()

           self.A_plus = A_plus
           self.A_minus = A_minus
           self.tau_plus = tau_plus
           self.tau_minus = tau_minus

           # Learnable weights
           self.W = brainstate.ParamState(jnp.ones((n_pre, n_post)) * 0.5)

           # Eligibility traces
           self.pre_trace = brainstate.ShortTermState(jnp.zeros(n_pre))
           self.post_trace = brainstate.ShortTermState(jnp.zeros(n_post))

       def reset_state(self, batch_size=None):
           shape_pre = self.W.value.shape[0] if batch_size is None else (batch_size, self.W.value.shape[0])
           shape_post = self.W.value.shape[1] if batch_size is None else (batch_size, self.W.value.shape[1])
           self.pre_trace.value = jnp.zeros(shape_pre)
           self.post_trace.value = jnp.zeros(shape_post)

       def update(self, pre_spike, post_spike):
           dt = brainstate.environ.get_dt()

           # Update traces
           self.pre_trace.value += -self.pre_trace.value / self.tau_plus.to_decimal(u.ms) * dt.to_decimal(u.ms) + pre_spike
           self.post_trace.value += -self.post_trace.value / self.tau_minus.to_decimal(u.ms) * dt.to_decimal(u.ms) + post_spike

           # Weight updates
           # LTP: pre spike finds existing post trace
           dw_ltp = self.A_plus * jnp.outer(pre_spike, self.post_trace.value)

           # LTD: post spike finds existing pre trace
           dw_ltd = -self.A_minus * jnp.outer(self.pre_trace.value, post_spike)

           # Update weights
           self.W.value = jnp.clip(self.W.value + dw_ltp + dw_ltd, 0, 1)

           return jnp.dot(pre_spike, self.W.value)

Custom Network Architectures
-----------------------------

Example: Liquid State Machine
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   class LiquidStateMachine(brainstate.nn.Module):
       """Reservoir computing with spiking neurons."""

       def __init__(self, n_input=100, n_reservoir=1000, n_output=10):
           super().__init__()

           # Input projection (trainable)
           self.input_weights = brainstate.ParamState(
               brainstate.random.randn(n_input, n_reservoir) * 0.1
           )

           # Reservoir (fixed random recurrent network)
           self.reservoir = bp.LIF(n_reservoir, V_rest=-65*u.mV, V_th=-50*u.mV, tau=10*u.ms)

           # Fixed random recurrent weights
           w_reservoir = brainstate.random.randn(n_reservoir, n_reservoir) * 0.01
           mask = (brainstate.random.rand(n_reservoir, n_reservoir) < 0.1).astype(float)
           self.reservoir_weights = w_reservoir * mask  # Not a ParamState (fixed)

           # Readout (trainable)
           self.readout = bp.Readout(n_reservoir, n_output)

       def update(self, x):
           # Input to reservoir
           reservoir_input = jnp.dot(x, self.input_weights.value) * u.nA

           # Reservoir recurrence
           spk = self.reservoir.get_spike()
           recurrent_input = jnp.dot(spk, self.reservoir_weights) * u.nA

           # Update reservoir
           self.reservoir(reservoir_input + recurrent_input)

           # Readout from reservoir state
           output = self.readout(self.reservoir.get_spike())

           return output

Custom Input Encoders
----------------------

Example: Temporal Contrast Encoder
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   class TemporalContrastEncoder(brainstate.nn.Module):
       """Encode images as spike timing based on contrast."""

       def __init__(self, n_pixels, max_time=100, threshold=0.1):
           super().__init__()
           self.n_pixels = n_pixels
           self.max_time = max_time
           self.threshold = threshold

       def encode(self, image):
           """Convert image to spike timing.

           Args:
               image: Array of pixel values [0, 1]

           Returns:
               spike_times: When each pixel spikes (or max_time if no spike)
           """
           # Higher intensity → earlier spike
           spike_times = jnp.where(
               image > self.threshold,
               self.max_time * (1 - image),  # Invert: bright pixels spike early
               self.max_time  # Below threshold: no spike
           )

           return spike_times

       def decode_to_spikes(self, spike_times, current_time):
           """Get spikes at current simulation time."""
           spikes = (spike_times == current_time).astype(float)
           return spikes

Best Practices
--------------

✅ **Inherit from base classes**
   - ``bp.Neuron`` for neurons
   - ``bp.Synapse`` for synapses
   - ``brainstate.nn.Module`` for general components

✅ **Use ShortTermState for dynamics**
   - Reset each trial
   - Temporary variables

✅ **Use ParamState for learnable parameters**
   - Trained by optimizers
   - Saved in checkpoints

✅ **Implement reset_state()**
   - Handle batch_size parameter
   - Initialize all ShortTermStates

✅ **Use physical units**
   - All parameters with ``brainunit``
   - Convert for computation with ``.to_decimal()``

✅ **Follow naming conventions**
   - ``V`` for voltage
   - ``spike`` for spike indicator
   - ``g`` for conductance
   - ``w`` for weights

Testing Custom Components
--------------------------

.. code-block:: python

   def test_custom_neuron():
       """Test custom neuron implementation."""

       neuron = CustomNeuron(size=10)
       brainstate.nn.init_all_states(neuron)

       # Test 1: Initialization
       assert neuron.V.value.shape == (10,)
       assert jnp.all(neuron.V.value == 0)

       # Test 2: Response to input
       strong_input = jnp.ones(10) * 10.0 * u.nA
       for _ in range(100):
           neuron(strong_input)

       spike_count = jnp.sum(neuron.spike.value)
       assert spike_count > 0, "Neuron should spike with strong input"

       # Test 3: Batch dimension
       brainstate.nn.init_all_states(neuron, batch_size=5)
       assert neuron.V.value.shape == (5, 10)

       print("✅ Custom neuron tests passed")

   test_custom_neuron()

Complete Example
----------------

**Putting it all together:**

.. code-block:: python

   # Custom components
   class MyNeuron(bp.Neuron):
       # ... (see examples above)
       pass

   class MySynapse(bp.Synapse):
       # ... (see examples above)
       pass

   # Use in network
   class CustomNetwork(brainstate.nn.Module):
       def __init__(self):
           super().__init__()

           self.pre = MyNeuron(size=100)
           self.post = MyNeuron(size=50)

           self.projection = bp.AlignPostProj(
               comm=brainstate.nn.EventFixedProb(100, 50, prob=0.1, weight=0.5*u.mS),
               syn=MySynapse.desc(50),  # Use custom synapse
               out=bp.CUBA.desc(),
               post=self.post
           )

       def update(self, inp):
           spk_pre = self.pre.get_spike()
           self.projection(spk_pre)
           self.pre(inp)
           self.post(0*u.nA)
           return self.post.get_spike()

   # Use network
   net = CustomNetwork()
   brainstate.nn.init_all_states(net)

   for _ in range(100):
       output = net(input_data)

Summary
-------

**Component creation checklist:**

.. code-block:: python

   ✅ Inherit from bp.Neuron, bp.Synapse, or brainstate.nn.Module
   ✅ Define __init__ with parameters
   ✅ Create states (ShortTermState or ParamState)
   ✅ Implement reset_state(batch_size=None)
   ✅ Implement update() method
   ✅ Use physical units throughout
   ✅ Test with different batch sizes

See Also
--------

- :doc:`../core-concepts/state-management` - Understanding states
- :doc:`../core-concepts/neurons` - Built-in neuron models
- :doc:`../core-concepts/synapses` - Built-in synapse models
- :doc:`../tutorials/advanced/06-synaptic-plasticity` - Plasticity examples
