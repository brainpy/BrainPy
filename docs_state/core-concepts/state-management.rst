State Management: The Foundation of ``brainpy.state``
=====================================================

State management is the core architectural change in ``brainpy.state``. Understanding states is
essential for using BrainPy effectively. This guide provides comprehensive coverage of the state
system built on ``brainstate``.

.. contents:: Table of Contents
   :local:
   :depth: 2

Overview
--------

What is State?
~~~~~~~~~~~~~~

**State** is any variable that persists across function calls and can change over time. In neural simulations:

- Membrane potentials
- Synaptic conductances
- Spike trains
- Learnable weights
- Temporary buffers

**Key insight:** ``brainpy.state`` makes states **explicit** rather than implicit. Every stateful variable is declared and tracked.

Why Explicit State Management?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Problems with implicit state (BrainPy 2.x):**

- Hard to track what changes when
- Difficult to serialize/checkpoint
- Unclear initialization procedures
- Conflicts with JAX functional programming

**Benefits of explicit state (``brainpy.state``):**

✅ Clear variable lifecycle

✅ Easy checkpointing and loading

✅ Functional programming compatible

✅ Better debugging and introspection

✅ Automatic differentiation support

✅ Type safety and validation

The State Hierarchy
~~~~~~~~~~~~~~~~~~~~

BrainPy uses different state types for different purposes:

.. code-block:: text

   State (base class)
   │
   ├── ParamState        ← Learnable parameters (weights, biases)
   ├── ShortTermState    ← Temporary dynamics (V, g, spikes)
   └── LongTermState     ← Persistent but non-learnable (statistics)

Each type has different semantics and handling:

- **ParamState**: Updated by optimizers, saved in checkpoints
- **ShortTermState**: Reset each trial, not saved
- **LongTermState**: Saved but not trained

State Types
-----------

ParamState: Learnable Parameters
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Use for:** Weights, biases, trainable parameters

**Characteristics:**

- Updated by gradient descent
- Saved in model checkpoints
- Persistent across trials
- Registered with optimizers

**Example:**

.. code-block:: python

   import brainstate
   import jax.numpy as jnp

   class LinearLayer(brainstate.nn.Module):
       def __init__(self, in_size, out_size):
           super().__init__()

           # Learnable weight matrix
           self.W = brainstate.ParamState(
               brainstate.random.randn(in_size, out_size) * 0.01
           )

           # Learnable bias vector
           self.b = brainstate.ParamState(
               jnp.zeros(out_size)
           )

       def update(self, x):
           # Use parameters in computation
           return jnp.dot(x, self.W.value) + self.b.value

   # Access all parameters
   layer = LinearLayer(100, 50)
   params = layer.states(brainstate.ParamState)
   # Returns: {'W': ParamState(...), 'b': ParamState(...)}

**Common uses:**

- Synaptic weights
- Neural biases
- Time constants (if learning them)
- Connectivity matrices (if plastic)

ShortTermState: Temporary Dynamics
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Use for:** Variables that reset each trial

**Characteristics:**

- Reset at trial start
- Not saved in checkpoints
- Represent current dynamics
- Fastest state type

**Example:**

.. code-block:: python

   import brainpy as bp
   import brainunit as u

   class LIFNeuron(brainstate.nn.Module):
       def __init__(self, size):
           super().__init__()

           self.size = size
           self.V_rest = -65.0 * u.mV
           self.V_th = -50.0 * u.mV

           # Membrane potential (resets each trial)
           self.V = brainstate.ShortTermState(
               jnp.ones(size) * self.V_rest.to_decimal(u.mV)
           )

           # Spike indicator (resets each trial)
           self.spike = brainstate.ShortTermState(
               jnp.zeros(size)
           )

       def reset_state(self, batch_size=None):
           """Called at trial start."""
           if batch_size is None:
               self.V.value = jnp.ones(self.size) * self.V_rest.to_decimal(u.mV)
               self.spike.value = jnp.zeros(self.size)
           else:
               self.V.value = jnp.ones((batch_size, self.size)) * self.V_rest.to_decimal(u.mV)
               self.spike.value = jnp.zeros((batch_size, self.size))

       def update(self, I):
           # Update membrane potential
           # ... (LIF dynamics)
           self.V.value = new_V
           self.spike.value = new_spike

**Common uses:**

- Membrane potentials
- Synaptic conductances
- Spike indicators
- Refractory counters
- Temporary buffers

LongTermState: Persistent Non-Learnable
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Use for:** Statistics, counters, persistent metadata

**Characteristics:**

- Not reset each trial
- Saved in checkpoints
- Not updated by optimizers
- Accumulates over time

**Example:**

.. code-block:: python

   class NeuronWithStatistics(brainstate.nn.Module):
       def __init__(self, size):
           super().__init__()

           self.V = brainstate.ShortTermState(jnp.zeros(size))

           # Running spike count (persists across trials)
           self.total_spikes = brainstate.LongTermState(
               jnp.zeros(size, dtype=jnp.int32)
           )

           # Running average firing rate
           self.avg_rate = brainstate.LongTermState(
               jnp.zeros(size)
           )

       def update(self, I):
           # ... update dynamics ...

           # Accumulate statistics
           self.total_spikes.value += self.spike.value.astype(jnp.int32)

**Common uses:**

- Spike counters
- Running averages
- Homeostatic variables
- Simulation metadata
- Custom statistics

State Initialization
--------------------

Automatic Initialization
~~~~~~~~~~~~~~~~~~~~~~~~

BrainPy provides ``init_all_states()`` for automatic initialization.

**Basic usage:**

.. code-block:: python

   import brainstate

   # Create network
   net = MyNetwork()

   # Initialize all states (single trial)
   brainstate.nn.init_all_states(net)

   # Initialize with batch dimension
   brainstate.nn.init_all_states(net, batch_size=32)

**What it does:**

1. Finds all modules in the hierarchy
2. Calls ``reset_state()`` on each module
3. Handles nested structures automatically
4. Sets up batch dimensions if requested

**Example with network:**

.. code-block:: python

   class EINetwork(brainstate.nn.Module):
       def __init__(self):
           super().__init__()
           self.E = bp.LIF(800, V_rest=-65*u.mV, V_th=-50*u.mV, tau=10*u.ms)
           self.I = bp.LIF(200, V_rest=-65*u.mV, V_th=-50*u.mV, tau=10*u.ms)
           # ... projections ...

   net = EINetwork()

   # This initializes E, I, and all projections
   brainstate.nn.init_all_states(net, batch_size=10)

Manual Initialization
~~~~~~~~~~~~~~~~~~~~~

For custom initialization, override ``reset_state()``.

.. code-block:: python

   class CustomNeuron(brainstate.nn.Module):
       def __init__(self, size, V_init_range=(-70, -60)):
           super().__init__()
           self.size = size
           self.V_init_range = V_init_range

           self.V = brainstate.ShortTermState(jnp.zeros(size))

       def reset_state(self, batch_size=None):
           """Custom initialization: random voltage in range."""

           # Generate random initial voltages
           low, high = self.V_init_range
           if batch_size is None:
               init_V = brainstate.random.uniform(low, high, size=self.size)
           else:
               init_V = brainstate.random.uniform(low, high, size=(batch_size, self.size))

           self.V.value = init_V

**Best practices:**

- Always check ``batch_size`` parameter
- Handle both single and batched cases
- Initialize all ShortTermStates
- Don't initialize ParamStates (they're learnable)
- Don't initialize LongTermStates (they persist)

Initializers for Parameters
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Use ``braintools.init`` for parameter initialization.

.. code-block:: python

   import braintools.init as init

   class Network(brainstate.nn.Module):
       def __init__(self, in_size, out_size):
           super().__init__()

           # Xavier/Glorot initialization
           self.W1 = brainstate.ParamState(
               init.XavierNormal()(shape=(in_size, 100))
           )

           # Kaiming/He initialization (for ReLU)
           self.W2 = brainstate.ParamState(
               init.KaimingNormal()(shape=(100, out_size))
           )

           # Zero initialization
           self.b = brainstate.ParamState(
               init.Constant(0.0)(shape=(out_size,))
           )

           # Orthogonal initialization (for RNNs)
           self.W_rec = brainstate.ParamState(
               init.Orthogonal()(shape=(100, 100))
           )

**Available initializers:**

- ``Constant(value)`` - Fill with constant
- ``Normal(mean, std)`` - Gaussian distribution
- ``Uniform(low, high)`` - Uniform distribution
- ``XavierNormal()`` - Xavier/Glorot normal
- ``XavierUniform()`` - Xavier/Glorot uniform
- ``KaimingNormal()`` - He normal (for ReLU)
- ``KaimingUniform()`` - He uniform
- ``Orthogonal()`` - Orthogonal matrix (for RNNs)
- ``Identity()`` - Identity matrix

State Access and Manipulation
------------------------------

Reading State Values
~~~~~~~~~~~~~~~~~~~~

Access the current value with ``.value``.

.. code-block:: python

   neuron = bp.LIF(100, V_rest=-65*u.mV, V_th=-50*u.mV, tau=10*u.ms)
   brainstate.nn.init_all_states(neuron)

   # Read current membrane potential
   current_V = neuron.V.value

   # Read shape
   print(current_V.shape)  # (100,)

   # Read specific neurons
   V_neuron_0 = neuron.V.value[0]

Writing State Values
~~~~~~~~~~~~~~~~~~~~

Update state by assigning to ``.value``.

.. code-block:: python

   # Set new value (entire array)
   neuron.V.value = jnp.ones(100) * -60.0

   # Update subset
   neuron.V.value = neuron.V.value.at[0:10].set(-55.0)

   # Increment
   neuron.V.value = neuron.V.value + 0.1

**Important:** Always assign to ``.value``, not the state object itself!

.. code-block:: python

   # CORRECT
   neuron.V.value = new_V

   # WRONG (creates new object, doesn't update state)
   neuron.V = new_V

Collecting States
~~~~~~~~~~~~~~~~~

Get all states of a specific type from a module.

.. code-block:: python

   # Get all parameters
   params = net.states(brainstate.ParamState)
   # Returns: dict with parameter names as keys

   # Get all short-term states
   short_term = net.states(brainstate.ShortTermState)

   # Get all states (any type)
   all_states = net.states()

**Example:**

.. code-block:: python

   class SimpleNet(brainstate.nn.Module):
       def __init__(self):
           super().__init__()
           self.W = brainstate.ParamState(jnp.ones((10, 10)))
           self.V = brainstate.ShortTermState(jnp.zeros(10))

   net = SimpleNet()

   params = net.states(brainstate.ParamState)
   # {'W': ParamState(...)}

   states = net.states(brainstate.ShortTermState)
   # {'V': ShortTermState(...)}

State in Training
-----------------

Gradient Computation
~~~~~~~~~~~~~~~~~~~~

Use ``brainstate.transform.grad()`` to compute gradients w.r.t. parameters.

.. code-block:: python

   def loss_fn(params, net, X, y):
       """Loss function parameterized by params."""
       # params is automatically used by net
       output = net(X)
       return jnp.mean((output - y) ** 2)

   # Get parameters
   params = net.states(brainstate.ParamState)

   # Compute gradients
   grads = brainstate.transform.grad(loss_fn, params)(net, X, y)

   # grads has same structure as params
   # grads = {'W': gradient_for_W, 'b': gradient_for_b, ...}

**Key points:**

- Gradients computed only for ParamState
- ShortTermState treated as constants
- Gradient structure matches parameter structure

Optimizer Updates
~~~~~~~~~~~~~~~~~

Register parameters with optimizer and update.

.. code-block:: python

   import braintools

   # Create optimizer
   optimizer = braintools.optim.Adam(learning_rate=1e-3)

   # Register trainable parameters
   params = net.states(brainstate.ParamState)
   optimizer.register_trainable_weights(params)

   # Training loop
   for epoch in range(num_epochs):
       for batch in data_loader:
           X, y = batch

           # Compute gradients
           grads = brainstate.transform.grad(
               loss_fn,
               params,
               return_value=False
           )(net, X, y)

           # Update parameters
           optimizer.update(grads)

**The optimizer automatically:**

- Updates all registered parameters
- Applies learning rate
- Handles momentum/adaptive rates
- Maintains optimizer state (momentum buffers, etc.)

State Persistence
~~~~~~~~~~~~~~~~~

Training doesn't reset ShortTermState between batches (unless you do it manually).

.. code-block:: python

   # Training with state reset each example
   for X, y in data_loader:
       # Reset dynamics for new example
       brainstate.nn.init_all_states(net)

       # Forward pass (dynamics evolve)
       output = net(X)

       # Backward pass
       grads = compute_grads(...)
       optimizer.update(grads)

   # Training with persistent state (e.g., RNN)
   for X, y in data_loader:
       # Don't reset - state carries over
       output = net(X)
       grads = compute_grads(...)
       optimizer.update(grads)

Batching
--------

Batch Dimensions
~~~~~~~~~~~~~~~~

States can have a batch dimension for parallel trials.

**Single trial:**

.. code-block:: python

   neuron = bp.LIF(100, ...)  # 100 neurons
   brainstate.nn.init_all_states(neuron)
   # neuron.V.value.shape = (100,)

**Batched trials:**

.. code-block:: python

   neuron = bp.LIF(100, ...)  # 100 neurons
   brainstate.nn.init_all_states(neuron, batch_size=32)
   # neuron.V.value.shape = (32, 100)

**Usage:**

.. code-block:: python

   # Input also needs batch dimension
   inp = brainstate.random.rand(32, 100) * 2.0 * u.nA

   # Update operates on all batches in parallel
   neuron(inp)

   # Output has batch dimension
   spikes = neuron.get_spike()  # shape: (32, 100)

Benefits of Batching
~~~~~~~~~~~~~~~~~~~~

**1. Parallelism:** GPU processes all batches simultaneously

**2. Statistical averaging:** Reduce noise in gradients

**3. Exploration:** Try different initial conditions

**4. Efficiency:** Amortize compilation cost

**Example: Parameter sweep with batching**

.. code-block:: python

   # Test 10 different input currents in parallel
   batch_size = 10
   neuron = bp.LIF(100, ...)
   brainstate.nn.init_all_states(neuron, batch_size=batch_size)

   # Different input for each batch
   currents = jnp.linspace(0, 5, batch_size).reshape(-1, 1) * u.nA
   inp = jnp.broadcast_to(currents, (batch_size, 100))

   # Simulate
   for _ in range(1000):
       neuron(inp)

   # Analyze each trial separately
   spike_counts = jnp.sum(neuron.spike.value, axis=1)  # (10,)

Checkpointing and Serialization
--------------------------------

Saving Models
~~~~~~~~~~~~~

Save model state to disk.

.. code-block:: python

   import pickle

   # Get all states to save
   state_dict = {
       'params': net.states(brainstate.ParamState),
       'long_term': net.states(brainstate.LongTermState),
       'epoch': current_epoch,
       'optimizer_state': optimizer.state_dict()  # If applicable
   }

   # Save to file
   with open('checkpoint.pkl', 'wb') as f:
       pickle.dump(state_dict, f)

**Note:** Don't save ShortTermState (it resets each trial).

Loading Models
~~~~~~~~~~~~~~

Restore model state from disk.

.. code-block:: python

   # Load checkpoint
   with open('checkpoint.pkl', 'rb') as f:
       state_dict = pickle.load(f)

   # Create fresh model
   net = MyNetwork()
   brainstate.nn.init_all_states(net)

   # Restore parameters
   params = state_dict['params']
   for name, param_state in params.items():
       # Find corresponding parameter in net
       # and copy value
       net_params = net.states(brainstate.ParamState)
       if name in net_params:
           net_params[name].value = param_state.value

   # Restore long-term states similarly
   # ...

   # Restore optimizer if continuing training
   optimizer.load_state_dict(state_dict['optimizer_state'])

Best Practices for Checkpointing
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**1. Save regularly during training**

.. code-block:: python

   if epoch % save_interval == 0:
       save_checkpoint(net, optimizer, epoch, path)

**2. Keep multiple checkpoints**

.. code-block:: python

   # Save with epoch number
   save_path = f'checkpoint_epoch_{epoch}.pkl'

**3. Save best model separately**

.. code-block:: python

   if val_loss < best_val_loss:
       best_val_loss = val_loss
       save_checkpoint(net, optimizer, epoch, 'best_model.pkl')

**4. Include metadata**

.. code-block:: python

   state_dict = {
       'params': ...,
       'epoch': epoch,
       'best_val_loss': best_val_loss,
       'config': model_config,  # Hyperparameters
       'timestamp': datetime.now()
   }

Common Patterns
---------------

Pattern 1: Resetting Between Trials
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Simulate multiple trials
   for trial in range(num_trials):
       # Reset dynamics
       brainstate.nn.init_all_states(net)

       # Run trial
       for t in range(trial_length):
           inp = get_input(trial, t)
           output = net(inp)
           record(output)

Pattern 2: Accumulating Statistics
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   class NeuronWithStats(brainstate.nn.Module):
       def __init__(self, size):
           super().__init__()
           self.V = brainstate.ShortTermState(jnp.zeros(size))

           # Accumulate across trials
           self.total_spikes = brainstate.LongTermState(
               jnp.zeros(size, dtype=jnp.int32)
           )
           self.n_steps = brainstate.LongTermState(0)

       def update(self, I):
           # ... dynamics ...

           # Accumulate
           self.total_spikes.value += self.spike.value.astype(jnp.int32)
           self.n_steps.value += 1

       def get_firing_rate(self):
           """Average firing rate across all trials."""
           dt = brainstate.environ.get_dt()
           total_time = self.n_steps.value * dt.to_decimal(u.second)
           return self.total_spikes.value / total_time

Pattern 3: Conditional Updates
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   class AdaptiveNeuron(brainstate.nn.Module):
       def __init__(self, size):
           super().__init__()
           self.V = brainstate.ShortTermState(jnp.zeros(size))
           self.threshold = brainstate.ParamState(jnp.ones(size) * (-50.0))

       def update(self, I):
           # Dynamics
           # ...

           # Homeostatic threshold adaptation
           spike_rate = compute_spike_rate(self.spike.value)

           # Adjust threshold based on activity
           target_rate = 5.0  # Hz
           adjustment = 0.01 * (spike_rate - target_rate)

           # Update learnable threshold
           self.threshold.value -= adjustment

Pattern 4: Hierarchical States
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   class HierarchicalNetwork(brainstate.nn.Module):
       def __init__(self):
           super().__init__()

           # Submodules have their own states
           self.layer1 = MyLayer(100, 50)
           self.layer2 = MyLayer(50, 10)

       def update(self, x):
           # Each layer manages its own states
           h1 = self.layer1(x)
           h2 = self.layer2(h1)
           return h2

   net = HierarchicalNetwork()

   # Collect ALL states from hierarchy
   all_params = net.states(brainstate.ParamState)
   # Includes params from layer1 AND layer2

   # Initialize ALL states in hierarchy
   brainstate.nn.init_all_states(net)
   # Calls reset_state() on net, layer1, and layer2

Advanced Topics
---------------

Custom State Types
~~~~~~~~~~~~~~~~~~

Create custom state types for specialized needs.

.. code-block:: python

   class RandomState(brainstate.State):
       """State that re-randomizes on reset."""

       def __init__(self, shape, low=0.0, high=1.0):
           super().__init__(jnp.zeros(shape))
           self.shape = shape
           self.low = low
           self.high = high

       def reset(self):
           """Re-randomize on reset."""
           self.value = brainstate.random.uniform(
               self.low, self.high, size=self.shape
           )

State Sharing
~~~~~~~~~~~~~

Share state between modules (use with caution).

.. code-block:: python

   class SharedState(brainstate.nn.Module):
       def __init__(self):
           super().__init__()

           # Shared weight matrix
           shared_W = brainstate.ParamState(jnp.ones((100, 100)))

           self.module1 = ModuleA(shared_W)
           self.module2 = ModuleB(shared_W)

       # module1 and module2 both modify the same weights

**When to use:** Siamese networks, weight tying, parameter sharing

**Caution:** Makes dependencies implicit, harder to debug

State Inspection
~~~~~~~~~~~~~~~~

Debug by inspecting state values.

.. code-block:: python

   # Print all parameter shapes
   params = net.states(brainstate.ParamState)
   for name, state in params.items():
       print(f"{name}: {state.value.shape}")

   # Check for NaN values
   for name, state in params.items():
       if jnp.any(jnp.isnan(state.value)):
           print(f"NaN detected in {name}!")

   # Compute statistics
   V_values = neuron.V.value
   print(f"V range: [{V_values.min():.2f}, {V_values.max():.2f}]")
   print(f"V mean: {V_values.mean():.2f}")

Troubleshooting
---------------

Issue: States not updating
~~~~~~~~~~~~~~~~~~~~~~~~~~

**Symptoms:** Values stay constant

**Solutions:**

1. Assign to ``.value``, not the state itself
2. Check you're updating the right variable
3. Verify update function is called

.. code-block:: python

   # WRONG
   self.V = new_V  # Creates new object!

   # CORRECT
   self.V.value = new_V  # Updates state

Issue: Batch dimension errors
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Symptoms:** Shape mismatch errors

**Solutions:**

1. Initialize with ``batch_size`` parameter
2. Ensure inputs have batch dimension
3. Check ``reset_state()`` handles batching

.. code-block:: python

   # Initialize with batching
   brainstate.nn.init_all_states(net, batch_size=32)

   # Input needs batch dimension
   inp = jnp.zeros((32, 100))  # (batch, neurons)

Issue: Gradients are None
~~~~~~~~~~~~~~~~~~~~~~~~~~

**Symptoms:** No gradients for parameters

**Solutions:**

1. Ensure parameters are ``ParamState``
2. Check parameters are used in loss computation
3. Verify gradient function call

.. code-block:: python

   # Parameters must be ParamState
   self.W = brainstate.ParamState(init_W)  # Correct

   # Compute gradients for parameters only
   params = net.states(brainstate.ParamState)
   grads = brainstate.transform.grad(loss_fn, params)(...)

Issue: Memory leak during training
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Symptoms:** Memory grows over time

**Solutions:**

1. Don't accumulate history in Python lists
2. Clear unnecessary references
3. Use ``jnp.array`` operations (not Python append)

.. code-block:: python

   # BAD - accumulates in Python memory
   history = []
   for t in range(10000):
       output = net(inp)
       history.append(output)  # Memory leak!

   # GOOD - use fixed-size buffer or don't store
   for t in range(10000):
       output = net(inp)
       # Process immediately, don't store

Further Reading
---------------

- :doc:`architecture` - Overall BrainPy architecture
- :doc:`neurons` - Neuron models and their states
- :doc:`synapses` - Synapse models and their states
- :doc:`../tutorials/advanced/05-snn-training` - Training with states
- BrainState documentation: https://brainstate.readthedocs.io/

Summary
-------

**Key takeaways:**

✅ **Three state types:**
   - ``ParamState``: Learnable parameters
   - ``ShortTermState``: Temporary dynamics
   - ``LongTermState``: Persistent statistics

✅ **Initialization:**
   - Use ``brainstate.nn.init_all_states(module)``
   - Implement ``reset_state()`` for custom logic
   - Handle batch dimensions

✅ **Access:**
   - Read/write with ``.value``
   - Collect with ``.states(StateType)``
   - Never assign to state object directly

✅ **Training:**
   - Gradients computed for ``ParamState``
   - Register with optimizer
   - Update with ``optimizer.update(grads)``

✅ **Checkpointing:**
   - Save ``ParamState`` and ``LongTermState``
   - Don't save ``ShortTermState``
   - Include metadata and optimizer state

**Quick reference:**

.. code-block:: python

   # Define states
   class MyModule(brainstate.nn.Module):
       def __init__(self):
           super().__init__()
           self.W = brainstate.ParamState(init_W)           # Learnable
           self.V = brainstate.ShortTermState(init_V)       # Resets
           self.count = brainstate.LongTermState(init_c)    # Persists

       def reset_state(self, batch_size=None):
           """Initialize ShortTermState."""
           shape = self.size if batch_size is None else (batch_size, self.size)
           self.V.value = jnp.zeros(shape)

   # Initialize
   brainstate.nn.init_all_states(module, batch_size=32)

   # Access
   params = module.states(brainstate.ParamState)
   module.V.value = new_V

   # Train
   grads = brainstate.transform.grad(loss, params)(...)
   optimizer.update(grads)
