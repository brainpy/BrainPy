How to Debug Networks
=====================

This guide shows you how to identify and fix common issues when developing neural networks with BrainPy.

.. contents:: Table of Contents
   :local:
   :depth: 2

Quick Diagnostic Checklist
---------------------------

When your network isn't working, check these first:

**☐ Is the network receiving input?**
   Print input values, check shapes

**☐ Are neurons firing?**
   Count spikes, check spike rates

**☐ Are projections working?**
   Verify connectivity, check weights

**☐ Is update order correct?**
   Get spikes BEFORE updating neurons

**☐ Are states initialized?**
   Call ``brainstate.nn.init_all_states()``

**☐ Are units correct?**
   All values need physical units (mV, nA, ms)

Common Issues and Solutions
----------------------------

Issue 1: No Spikes / Silent Network
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Symptoms:**

- Network produces no spikes
- All neurons stay at rest potential

**Diagnosis:**

.. code-block:: python

   import brainpy
   import brainstate
   import brainunit as u

   neuron = brainpy.state.LIF(100, V_rest=-65*u.mV, V_th=-50*u.mV, tau=10*u.ms)
   brainstate.nn.init_all_states(neuron)

   # Check 1: Is input being provided?
   inp = brainstate.random.rand(100) * 5.0 * u.nA
   print("Input range:", inp.min(), "to", inp.max())

   # Check 2: Are neurons updating?
   V_before = neuron.V.value.copy()
   neuron(inp)
   V_after = neuron.V.value
   print("Voltage changed:", not jnp.allclose(V_before, V_after))

   # Check 3: Are any neurons near threshold?
   print("Max voltage:", V_after.max())
   print("Threshold:", neuron.V_th.to_decimal(u.mV))
   print("Neurons above -55mV:", jnp.sum(V_after > -55))

   # Check 4: Count spikes
   for i in range(100):
       neuron(inp)
   spike_count = jnp.sum(neuron.spike.value)
   print(f"Spikes in 100 steps: {spike_count}")

**Common Causes:**

1. **Input too weak:**

   .. code-block:: python

      # Too weak
      inp = brainstate.random.rand(100) * 0.1 * u.nA  # Not enough!

      # Better
      inp = brainstate.random.rand(100) * 5.0 * u.nA  # Stronger

2. **Threshold too high:**

   .. code-block:: python

      # Check threshold
      neuron = brainpy.state.LIF(100, V_th=-40*u.mV, ...)  # Harder to spike
      neuron = brainpy.state.LIF(100, V_th=-50*u.mV, ...)  # Easier to spike

3. **Time constant too large:**

   .. code-block:: python

      # Slow integration
      neuron = brainpy.state.LIF(100, tau=100*u.ms, ...)  # Very slow

      # Faster
      neuron = brainpy.state.LIF(100, tau=10*u.ms, ...)  # Normal speed

4. **Missing initialization:**

   .. code-block:: python

      neuron = brainpy.state.LIF(100, ...)
      # MUST initialize!
      brainstate.nn.init_all_states(neuron)

Issue 2: Runaway Activity / Explosion
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Symptoms:**

- All neurons fire constantly
- Membrane potentials go to infinity
- NaN values appear

**Diagnosis:**

.. code-block:: python

   # Check for NaN
   if jnp.any(jnp.isnan(neuron.V.value)):
       print("❌ NaN detected in membrane potential!")

   # Check for explosion
   if jnp.any(jnp.abs(neuron.V.value) > 1000):
       print("❌ Membrane potential exploded!")

   # Check spike rate
   spike_rate = jnp.mean(neuron.spike.value)
   print(f"Spike rate: {spike_rate*100:.1f}%")
   if spike_rate > 0.5:
       print("⚠️ More than 50% of neurons firing every step!")

**Common Causes:**

1. **Excitation-Inhibition imbalance:**

   .. code-block:: python

      # Imbalanced (explosion!)
      w_exc = 5.0 * u.mS  # Too strong
      w_inh = 1.0 * u.mS  # Too weak

      # Balanced
      w_exc = 0.5 * u.mS
      w_inh = 5.0 * u.mS  # Inhibition ~10× stronger

2. **Positive feedback loop:**

   .. code-block:: python

      # Check recurrent excitation
      # E → E with no inhibition can explode

      # Add inhibition
      class BalancedNetwork(brainstate.nn.Module):
          def __init__(self):
              super().__init__()
              self.E = brainpy.state.LIF(800, ...)
              self.I = brainpy.state.LIF(200, ...)

              self.E2E = ...  # Excitatory recurrence
              self.I2E = ...  # MUST have inhibition!

3. **Time step too large:**

   .. code-block:: python

      # Unstable
      brainstate.environ.set(dt=1.0 * u.ms)  # Too large

      # Stable
      brainstate.environ.set(dt=0.1 * u.ms)  # Standard

4. **Wrong reversal potentials:**

   .. code-block:: python

      # WRONG: Inhibition with excitatory reversal
      out_inh = brainpy.state.COBA.desc(E=0*u.mV)  # Should be negative!

      # CORRECT
      out_exc = brainpy.state.COBA.desc(E=0*u.mV)    # Excitation
      out_inh = brainpy.state.COBA.desc(E=-80*u.mV)  # Inhibition

Issue 3: Spikes Not Propagating
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Symptoms:**

- Presynaptic neurons spike
- Postsynaptic neurons don't respond
- Projection seems inactive

**Diagnosis:**

.. code-block:: python

   # Create simple network
   pre = brainpy.state.LIF(10, V_rest=-65*u.mV, V_th=-50*u.mV, tau=10*u.ms)
   post = brainpy.state.LIF(10, V_rest=-65*u.mV, V_th=-50*u.mV, tau=10*u.ms)

   proj = brainpy.state.AlignPostProj(
       comm=brainstate.nn.EventFixedProb(10, 10, prob=0.5, weight=2.0*u.mS),
       syn=brainpy.state.Expon.desc(10, tau=5*u.ms),
       out=brainpy.state.CUBA.desc(),
       post=post
   )

   brainstate.nn.init_all_states([pre, post, proj])

   # Diagnosis
   for i in range(10):
       # CRITICAL: Get spikes BEFORE update
       pre_spikes = pre.get_spike()

       # Strong input to pre
       pre(brainstate.random.rand(10) * 10.0 * u.nA)

       # Check: Did pre spike?
       if jnp.sum(pre_spikes) > 0:
           print(f"Step {i}: {jnp.sum(pre_spikes)} presynaptic spikes")

           # Update projection
           proj(pre_spikes)

           # Check: Did projection produce current?
           print(f"  Synaptic conductance: {proj.syn.g.value.max():.4f}")

       # Update post
       post(0*u.nA)  # Only synaptic input

       # Check: Did post spike?
       post_spikes = post.get_spike()
       print(f"  {jnp.sum(post_spikes)} postsynaptic spikes")

**Common Causes:**

1. **Wrong spike timing:**

   .. code-block:: python

      # WRONG: Spikes from current step
      pre(inp)              # Update first
      spikes = pre.get_spike()  # These are NEW spikes
      proj(spikes)          # But projection needs OLD spikes!

      # CORRECT: Spikes from previous step
      spikes = pre.get_spike()  # Get OLD spikes first
      proj(spikes)              # Update projection
      pre(inp)                  # Then update neurons

2. **Weak connectivity:**

   .. code-block:: python

      # Too sparse
      comm = brainstate.nn.EventFixedProb(..., prob=0.01, weight=0.1*u.mS)

      # Stronger
      comm = brainstate.nn.EventFixedProb(..., prob=0.1, weight=1.0*u.mS)

3. **Missing projection update:**

   .. code-block:: python

      # Forgot to call projection!
      spk = pre.get_spike()
      # proj(spk)  <- MISSING!
      post(0*u.nA)

4. **Wrong postsynaptic target:**

   .. code-block:: python

      # Wrong target
      proj = brainpy.state.AlignPostProj(..., post=wrong_population)

      # Correct target
      proj = brainpy.state.AlignPostProj(..., post=correct_population)

Issue 4: Shape Mismatch Errors
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Symptoms:**

.. code-block:: text

   ValueError: operands could not be broadcast together
   with shapes (100,) (64, 100)

**Common Causes:**

1. **Batch dimension mismatch:**

   .. code-block:: python

      # Network initialized with batch
      brainstate.nn.init_all_states(net, batch_size=64)
      # States shape: (64, 100)

      # But input has no batch
      inp = jnp.zeros(100)  # Shape: (100,) - WRONG!

      # Fix: Add batch dimension
      inp = jnp.zeros((64, 100))  # Shape: (64, 100) - CORRECT

2. **Forgot batch in initialization:**

   .. code-block:: python

      # Initialized without batch
      brainstate.nn.init_all_states(net)  # Shape: (100,)

      # But providing batched input
      inp = jnp.zeros((64, 100))  # Shape: (64, 100)

      # Fix: Initialize with batch
      brainstate.nn.init_all_states(net, batch_size=64)

**Debug shape mismatches:**

.. code-block:: python

   print(f"Input shape: {inp.shape}")
   print(f"Network state shape: {net.neurons.V.value.shape}")
   print(f"Expected: Both should have same batch dimension")

Inspection Tools
----------------

Print State Values
~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Inspect neuron states
   neuron = brainpy.state.LIF(10, ...)
   brainstate.nn.init_all_states(neuron)

   print("Membrane potentials:", neuron.V.value)
   print("Spikes:", neuron.spike.value)
   print("Shape:", neuron.V.value.shape)

   # Statistics
   print(f"V range: [{neuron.V.value.min():.2f}, {neuron.V.value.max():.2f}]")
   print(f"V mean: {neuron.V.value.mean():.2f}")
   print(f"Spike count: {jnp.sum(neuron.spike.value)}")

Visualize Activity
~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import matplotlib.pyplot as plt
   import numpy as np

   # Record activity
   n_steps = 1000
   V_history = []
   spike_history = []

   for i in range(n_steps):
       neuron(inp)
       V_history.append(neuron.V.value.copy())
       spike_history.append(neuron.spike.value.copy())

   V_history = jnp.array(V_history)
   spike_history = jnp.array(spike_history)

   # Plot membrane potential
   plt.figure(figsize=(12, 4))
   plt.plot(V_history[:, 0])  # First neuron
   plt.xlabel('Time step')
   plt.ylabel('Membrane Potential (mV)')
   plt.title('Neuron 0 Membrane Potential')
   plt.show()

   # Plot raster
   plt.figure(figsize=(12, 6))
   times, neurons = jnp.where(spike_history > 0)
   plt.scatter(times, neurons, s=1, c='black')
   plt.xlabel('Time step')
   plt.ylabel('Neuron index')
   plt.title('Spike Raster')
   plt.show()

   # Firing rate over time
   plt.figure(figsize=(12, 4))
   firing_rate = jnp.mean(spike_history, axis=1) * 1000 / 0.1  # Hz
   plt.plot(firing_rate)
   plt.xlabel('Time step')
   plt.ylabel('Population Rate (Hz)')
   plt.title('Population Firing Rate')
   plt.show()

Check Connectivity
~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # For sparse projections
   proj = brainpy.state.AlignPostProj(
       comm=brainstate.nn.EventFixedProb(100, 50, prob=0.1, weight=0.5*u.mS),
       syn=brainpy.state.Expon.desc(50, tau=5*u.ms),
       out=brainpy.state.CUBA.desc(),
       post=post_neurons
   )

   # Check connection count
   print(f"Expected connections: {100 * 50 * 0.1:.0f}")
   # Note: Actual connectivity may vary due to randomness

   # Check weights
   # (Accessing internal connectivity structure depends on implementation)

Monitor Training
~~~~~~~~~~~~~~~~

.. code-block:: python

   # Track loss and metrics
   train_losses = []
   val_accuracies = []

   for epoch in range(num_epochs):
       epoch_losses = []

       for batch in train_loader:
           loss = train_step(net, batch)
           epoch_losses.append(float(loss))

       avg_loss = np.mean(epoch_losses)
       train_losses.append(avg_loss)

       # Validation
       val_acc = evaluate(net, val_loader)
       val_accuracies.append(val_acc)

       print(f"Epoch {epoch}: Loss={avg_loss:.4f}, Val Acc={val_acc:.2%}")

       # Check for issues
       if np.isnan(avg_loss):
           print("❌ NaN loss! Stopping training.")
           break

       if avg_loss > 10 * train_losses[0]:
           print("⚠️ Loss exploding!")

   # Plot training curves
   plt.figure(figsize=(12, 4))
   plt.subplot(1, 2, 1)
   plt.plot(train_losses)
   plt.xlabel('Epoch')
   plt.ylabel('Loss')
   plt.title('Training Loss')

   plt.subplot(1, 2, 2)
   plt.plot(val_accuracies)
   plt.xlabel('Epoch')
   plt.ylabel('Accuracy')
   plt.title('Validation Accuracy')
   plt.show()

Advanced Debugging
------------------

Gradient Checking
~~~~~~~~~~~~~~~~~

.. code-block:: python

   import braintools

   # Check if gradients are being computed
   params = net.states(brainstate.ParamState)

   grads = brainstate.transform.grad(
       loss_fn,
       params,
       return_value=True
   )(net, X, y)

   # Inspect gradients
   for name, grad in grads.items():
       grad_norm = jnp.linalg.norm(grad.value.flatten())
       print(f"{name}: gradient norm = {grad_norm:.6f}")

       if jnp.any(jnp.isnan(grad.value)):
           print(f"  ❌ NaN in gradient!")

       if grad_norm == 0:
           print(f"  ⚠️ Zero gradient - parameter not learning")

       if grad_norm > 1000:
           print(f"  ⚠️ Exploding gradient!")

Trace Execution
~~~~~~~~~~~~~~~

.. code-block:: python

   def debug_step(net, inp):
       """Instrumented simulation step."""
       print(f"\n--- Step Start ---")

       # Before
       print(f"Input range: [{inp.min():.2f}, {inp.max():.2f}]")
       print(f"V before: [{net.neurons.V.value.min():.2f}, {net.neurons.V.value.max():.2f}]")

       # Execute
       output = net(inp)

       # After
       print(f"V after: [{net.neurons.V.value.min():.2f}, {net.neurons.V.value.max():.2f}]")
       print(f"Spikes: {jnp.sum(net.neurons.spike.value)}")
       print(f"Output range: [{output.min():.2f}, {output.max():.2f}]")

       # Checks
       if jnp.any(jnp.isnan(net.neurons.V.value)):
           print("❌ NaN detected!")
           import pdb; pdb.set_trace()  # Drop into debugger

       print(f"--- Step End ---\n")
       return output

   # Use for debugging
   for i in range(10):
       output = debug_step(net, input_data)

Assertion Checks
~~~~~~~~~~~~~~~~

.. code-block:: python

   class SafeNetwork(brainstate.nn.Module):
       """Network with built-in checks."""

       def __init__(self, n_neurons=100):
           super().__init__()
           self.neurons = brainpy.state.LIF(n_neurons, ...)

       def update(self, inp):
           # Pre-checks
           assert inp.shape[-1] == 100, f"Wrong input size: {inp.shape}"
           assert not jnp.any(jnp.isnan(inp)), "NaN in input!"
           assert not jnp.any(jnp.isinf(inp)), "Inf in input!"

           # Execute
           self.neurons(inp)
           output = self.neurons.get_spike()

           # Post-checks
           assert not jnp.any(jnp.isnan(self.neurons.V.value)), "NaN in membrane potential!"
           assert jnp.all(jnp.abs(self.neurons.V.value) < 1000), "Voltage explosion!"

           return output

Unit Testing
~~~~~~~~~~~~

.. code-block:: python

   def test_neuron_spikes():
       """Test that neuron spikes with strong input."""
       neuron = brainpy.state.LIF(1, V_rest=-65*u.mV, V_th=-50*u.mV, tau=10*u.ms)
       brainstate.nn.init_all_states(neuron)

       # Strong constant input should cause spiking
       strong_input = jnp.array([20.0]) * u.nA

       spike_count = 0
       for _ in range(100):
           neuron(strong_input)
           spike_count += int(neuron.spike.value[0])

       assert spike_count > 0, "Neuron didn't spike with strong input!"
       assert spike_count < 100, "Neuron spiked every step (check reset!)"

       print(f"✅ Neuron test passed ({spike_count} spikes)")

   def test_projection():
       """Test that projection propagates spikes."""
       pre = brainpy.state.LIF(10, V_rest=-65*u.mV, V_th=-50*u.mV, tau=10*u.ms)
       post = brainpy.state.LIF(10, V_rest=-65*u.mV, V_th=-50*u.mV, tau=10*u.ms)

       proj = brainpy.state.AlignPostProj(
           comm=brainstate.nn.EventFixedProb(10, 10, prob=1.0, weight=5.0*u.mS),  # 100% connectivity
           syn=brainpy.state.Expon.desc(10, tau=5*u.ms),
           out=brainpy.state.CUBA.desc(),
           post=post
       )

       brainstate.nn.init_all_states([pre, post, proj])

       # Make pre spike
       pre(jnp.ones(10) * 20.0 * u.nA)

       # Projection should activate
       spk = pre.get_spike()
       assert jnp.sum(spk) > 0, "Pre didn't spike!"

       proj(spk)

       # Check synaptic conductance increased
       assert proj.syn.g.value.max() > 0, "Synapse didn't activate!"

       print("✅ Projection test passed")

   # Run tests
   test_neuron_spikes()
   test_projection()

Debugging Checklist
-------------------

When your network doesn't work:

**1. Check Initialization**

.. code-block:: python

   ☐ Called brainstate.nn.init_all_states()?
   ☐ Correct batch_size parameter?
   ☐ All submodules initialized?

**2. Check Input**

.. code-block:: python

   ☐ Input shape matches network?
   ☐ Input has units (nA, mV, etc.)?
   ☐ Input magnitude reasonable?
   ☐ Input not all zeros?

**3. Check Neurons**

.. code-block:: python

   ☐ Threshold reasonable (e.g., -50 mV)?
   ☐ Reset potential below threshold?
   ☐ Time constant reasonable (5-20 ms)?
   ☐ Neurons actually spiking?

**4. Check Projections**

.. code-block:: python

   ☐ Connectivity probability > 0?
   ☐ Weights reasonable magnitude?
   ☐ Correct update order (spikes before update)?
   ☐ Projection actually called?

**5. Check Balance**

.. code-block:: python

   ☐ Inhibition stronger than excitation (~10×)?
   ☐ Reversal potentials correct (E=0, I=-80)?
   ☐ E/I ratio appropriate (4:1)?

**6. Check Training**

.. code-block:: python

   ☐ Loss decreasing?
   ☐ Gradients non-zero?
   ☐ No NaN in gradients?
   ☐ Learning rate appropriate?

Common Error Messages
---------------------

"operands could not be broadcast"
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Meaning:** Shape mismatch

**Fix:** Check batch dimensions

.. code-block:: python

   print(f"Shapes: {x.shape} vs {y.shape}")

"RESOURCE_EXHAUSTED: Out of memory"
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Meaning:** GPU/CPU memory full

**Fix:** Reduce batch size or network size

.. code-block:: python

   # Reduce batch
   brainstate.nn.init_all_states(net, batch_size=16)  # Instead of 64

"Concrete value required"
~~~~~~~~~~~~~~~~~~~~~~~~~

**Meaning:** JIT can't handle dynamic values

**Fix:** Use static shapes

.. code-block:: python

   # Dynamic (bad for JIT)
   n = len(data)  # Changes each call

   # Static (good for JIT)
   n = 100  # Fixed value

"Invalid device"
~~~~~~~~~~~~~~~~

**Meaning:** Trying to use unavailable device

**Fix:** Check available devices

.. code-block:: python

   import jax
   print(jax.devices())

Best Practices
--------------

✅ **Test small first** - Debug with 10 neurons before scaling to 10,000

✅ **Visualize early** - Plot activity to see problems immediately

✅ **Check incrementally** - Test each component before combining

✅ **Use assertions** - Catch problems early with runtime checks

✅ **Print liberally** - Add diagnostic prints during development

✅ **Keep backups** - Save working versions before major changes

✅ **Start simple** - Begin with minimal network, add complexity gradually

✅ **Write tests** - Unit test individual components

❌ **Don't debug by guessing** - Use systematic diagnosis

❌ **Don't skip initialization** - Always call init_all_states

❌ **Don't ignore warnings** - They often indicate real problems

Summary
-------

**Debugging workflow:**

1. **Identify symptom** (no spikes, explosion, etc.)
2. **Isolate component** (neurons, projections, input)
3. **Inspect state** (print values, plot activity)
4. **Form hypothesis** (what might be wrong?)
5. **Test fix** (make one change at a time)
6. **Verify** (ensure problem solved)

**Quick diagnostic code:**

.. code-block:: python

   # Comprehensive diagnostic
   def diagnose_network(net, inp):
       print("=== Network Diagnostic ===")

       # Input
       print(f"Input shape: {inp.shape}")
       print(f"Input range: [{inp.min():.2f}, {inp.max():.2f}]")

       # States
       if hasattr(net, 'neurons'):
           V = net.neurons.V.value
           print(f"Voltage shape: {V.shape}")
           print(f"Voltage range: [{V.min():.2f}, {V.max():.2f}]")

       # Simulation
       output = net(inp)

       # Results
       if hasattr(net, 'neurons'):
           spk_count = jnp.sum(net.neurons.spike.value)
           print(f"Spikes: {spk_count}")

       print(f"Output shape: {output.shape}")
       print(f"Output range: [{output.min():.2f}, {output.max():.2f}]")

       # Checks
       if jnp.any(jnp.isnan(output)):
           print("❌ NaN in output!")
       if jnp.all(output == 0):
           print("⚠️  Output all zeros!")

       print("=========================")
       return output

See Also
--------

- :doc:`../core-concepts/state-management` - Understanding state system
- :doc:`../core-concepts/projections` - Projection architecture
- :doc:`performance-optimization` - Optimization tips
