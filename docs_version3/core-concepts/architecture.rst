Architecture Overview
====================

BrainPy 3.0 represents a complete architectural redesign built on top of the ``brainstate`` framework. This document explains the design principles and architectural components that make BrainPy 3.0 powerful and flexible.

Design Philosophy
-----------------

BrainPy 3.0 is built around several core principles:

**State-Based Programming**
   All dynamical variables are managed as explicit states, enabling automatic differentiation, efficient compilation, and clear data flow.

**Modular Composition**
   Complex models are built by composing simple, reusable components. Each component has a well-defined interface and responsibility.

**Scientific Accuracy**
   Integration with ``brainunit`` ensures physical correctness and prevents unit-related errors.

**Performance by Default**
   JIT compilation and optimization are built into the framework, not an afterthought.

**Extensibility**
   Adding new neuron models, synapse types, or learning rules is straightforward and follows clear patterns.

Architectural Layers
--------------------

BrainPy 3.0 is organized into several layers:

.. code-block:: text

    ┌─────────────────────────────────────────┐
    │         User Models & Networks          │  ← Your code
    ├─────────────────────────────────────────┤
    │      BrainPy Components Layer           │  ← Neurons, Synapses, Projections
    ├─────────────────────────────────────────┤
    │       BrainState Framework              │  ← State management, compilation
    ├─────────────────────────────────────────┤
    │       JAX + XLA Backend                 │  ← JIT compilation, autodiff
    └─────────────────────────────────────────┘

1. JAX + XLA Backend
~~~~~~~~~~~~~~~~~~~~

The foundation layer provides:

- Just-In-Time (JIT) compilation
- Automatic differentiation
- Hardware acceleration (CPU/GPU/TPU)
- Functional transformations (vmap, grad, etc.)

2. BrainState Framework
~~~~~~~~~~~~~~~~~~~~~~~~

Built on JAX, ``brainstate`` provides:

- State management system
- Module composition
- Compilation and optimization
- Program transformations (for_loop, etc.)

3. BrainPy Components
~~~~~~~~~~~~~~~~~~~~~

High-level neuroscience-specific components:

- Neuron models (LIF, ALIF, etc.)
- Synapse models (Expon, Alpha, etc.)
- Projection architectures
- Learning rules and plasticity

4. User Models
~~~~~~~~~~~~~~

Your custom networks and experiments built using BrainPy components.

State Management System
-----------------------

The Foundation: brainstate.State
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Everything in BrainPy 3.0 revolves around **states**:

.. code-block:: python

    import brainstate

    # Create a state
    voltage = brainstate.State(0.0)  # Single value
    weights = brainstate.State([[0.1, 0.2], [0.3, 0.4]])  # Matrix

States are special containers that:

- Track their values across time
- Support automatic differentiation
- Enable efficient compilation
- Handle batching automatically

State Types
~~~~~~~~~~~

BrainPy uses different state types for different purposes:

**ParamState** - Trainable Parameters
   Used for weights, time constants, and other trainable parameters.

   .. code-block:: python

       class MyNeuron(brainstate.nn.Module):
           def __init__(self):
               super().__init__()
               self.tau = brainstate.ParamState(10.0)  # Trainable
               self.weight = brainstate.ParamState([[0.1, 0.2]])

**ShortTermState** - Temporary Variables
   Used for membrane potentials, synaptic currents, and other dynamics.

   .. code-block:: python

       class MyNeuron(brainstate.nn.Module):
           def __init__(self, size):
               super().__init__()
               self.V = brainstate.ShortTermState(jnp.zeros(size))  # Dynamic
               self.spike = brainstate.ShortTermState(jnp.zeros(size))

State Initialization
~~~~~~~~~~~~~~~~~~~~

States can be initialized with various strategies:

.. code-block:: python

    import braintools
    import brainunit as u

    # Constant initialization
    V = brainstate.ShortTermState(
        braintools.init.Constant(-65.0, unit=u.mV)(size)
    )

    # Normal distribution
    V = brainstate.ShortTermState(
        braintools.init.Normal(-65.0, 5.0, unit=u.mV)(size)
    )

    # Uniform distribution
    weights = brainstate.ParamState(
        braintools.init.Uniform(0.0, 1.0)(shape)
    )

Module System
-------------

Base Class: brainstate.nn.Module
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

All BrainPy components inherit from ``brainstate.nn.Module``:

.. code-block:: python

    class MyComponent(brainstate.nn.Module):
        def __init__(self, size):
            super().__init__()
            # Initialize states
            self.state1 = brainstate.ShortTermState(...)
            self.param1 = brainstate.ParamState(...)

        def update(self, input):
            # Define dynamics
            pass

Benefits of Module:

- Automatic state registration
- Nested module support
- State collection and filtering
- Serialization support

Module Composition
~~~~~~~~~~~~~~~~~~

Modules can contain other modules:

.. code-block:: python

    import brainpy

    class Network(brainstate.nn.Module):
        def __init__(self):
            super().__init__()
            self.neurons = brainpy.LIF(100)          # Neuron module
            self.synapse = brainpy.Expon(100)        # Synapse module
            self.projection = brainpy.AlignPostProj(...)  # Projection module

        def update(self, input):
            # Compose behavior
            self.projection(spikes)
            self.neurons(input)

Component Architecture
----------------------

Neurons
~~~~~~~

Neurons model the dynamics of neural populations:

.. code-block:: python

    class Neuron(brainstate.nn.Module):
        def __init__(self, size, ...):
            super().__init__()
            # Membrane potential
            self.V = brainstate.ShortTermState(...)
            # Spike output
            self.spike = brainstate.ShortTermState(...)

        def update(self, input_current):
            # Update membrane potential
            # Generate spikes
            pass

Key responsibilities:

- Maintain membrane potential
- Generate spikes when threshold is crossed
- Reset after spiking
- Integrate input currents

Synapses
~~~~~~~~

Synapses model temporal filtering of spike trains:

.. code-block:: python

    class Synapse(brainstate.nn.Module):
        def __init__(self, size, tau, ...):
            super().__init__()
            # Synaptic conductance/current
            self.g = brainstate.ShortTermState(...)

        def update(self, spike_input):
            # Update synaptic variable
            # Return filtered output
            pass

Key responsibilities:

- Filter spike inputs temporally
- Model synaptic dynamics (exponential, alpha, etc.)
- Provide smooth currents to postsynaptic neurons

Projections: The Comm-Syn-Out Pattern
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Projections connect populations using a three-stage architecture:

.. code-block:: text

    Presynaptic Spikes → [Comm] → [Syn] → [Out] → Postsynaptic Neurons
                          │         │       │
                      Connectivity  │    Current
                      & Weights   Dynamics  Injection

**Communication (Comm)**
   Handles spike transmission, connectivity, and weights.

   .. code-block:: python

       comm = brainstate.nn.EventFixedProb(
           pre_size, post_size, prob=0.1, weight=0.5
       )

**Synaptic Dynamics (Syn)**
   Temporal filtering of transmitted spikes.

   .. code-block:: python

       syn = brainpy.Expon.desc(post_size, tau=5*u.ms)

**Output Mechanism (Out)**
   How synaptic variables affect postsynaptic neurons.

   .. code-block:: python

       out = brainpy.CUBA.desc()  # Current-based
       # or
       out = brainpy.COBA.desc()  # Conductance-based

**Complete Projection**

.. code-block:: python

    projection = brainpy.AlignPostProj(
        comm=comm,
        syn=syn,
        out=out,
        post=postsynaptic_neurons
    )

This separation provides:

- Clear responsibility boundaries
- Easy component swapping
- Reusable building blocks
- Better testing and debugging

Compilation and Execution
--------------------------

Time-Stepped Simulation
~~~~~~~~~~~~~~~~~~~~~~~

BrainPy uses discrete time steps:

.. code-block:: python

    import brainunit as u

    # Set global time step
    brainstate.environ.set(dt=0.1 * u.ms)

    # Define simulation duration
    times = u.math.arange(0*u.ms, 1000*u.ms, brainstate.environ.get_dt())

    # Run simulation
    results = brainstate.transform.for_loop(
        network.update,
        times,
        pbar=brainstate.transform.ProgressBar(10)
    )

JIT Compilation
~~~~~~~~~~~~~~~

Functions are compiled for performance:

.. code-block:: python

    @brainstate.compile.jit
    def simulate_step(input):
        return network.update(input)

    # First call: compile
    result = simulate_step(input)  # Slow (compilation)

    # Subsequent calls: fast
    result = simulate_step(input)  # Fast (compiled)

Compilation benefits:

- 10-100x speedup over Python
- Automatic GPU/TPU dispatch
- Memory optimization
- Fusion of operations

Gradient Computation
~~~~~~~~~~~~~~~~~~~~

For training, gradients are computed automatically:

.. code-block:: python

    def loss_fn():
        predictions = network.forward(inputs)
        return compute_loss(predictions, targets)

    # Compute gradients
    grads, loss = brainstate.transform.grad(
        loss_fn,
        network.states(brainstate.ParamState),
        return_value=True
    )()

    # Update parameters
    optimizer.update(grads)

Physical Units System
---------------------

Integration with brainunit
~~~~~~~~~~~~~~~~~~~~~~~~~~

BrainPy 3.0 integrates ``brainunit`` for scientific accuracy:

.. code-block:: python

    import brainunit as u

    # Define with units
    tau = 10 * u.ms
    threshold = -50 * u.mV
    current = 5 * u.nA

    # Units are checked automatically
    neuron = brainpy.LIF(100, tau=tau, V_th=threshold)

Benefits:

- Prevents unit errors (e.g., ms vs s)
- Self-documenting code
- Automatic unit conversions
- Scientific correctness

Unit Operations
~~~~~~~~~~~~~~~

.. code-block:: python

    # Arithmetic with units
    total_time = 100 * u.ms + 0.5 * u.second  # → 600 ms

    # Unit conversion
    time_in_seconds = (100 * u.ms).to_decimal(u.second)  # → 0.1

    # Unit checking (automatic in BrainPy operations)
    voltage = -65 * u.mV
    current = 2 * u.nA
    resistance = voltage / current  # Automatically gives MΩ

Ecosystem Integration
---------------------

BrainPy 3.0 integrates tightly with its ecosystem:

braintools
~~~~~~~~~~

Utilities and tools:

.. code-block:: python

    import braintools

    # Optimizers
    optimizer = braintools.optim.Adam(lr=1e-3)

    # Initializers
    init = braintools.init.KaimingNormal()

    # Surrogate gradients
    spike_fn = braintools.surrogate.ReluGrad()

    # Metrics
    loss = braintools.metric.cross_entropy(pred, target)

brainunit
~~~~~~~~~

Physical units:

.. code-block:: python

    import brainunit as u

    # All standard SI units
    time = 10 * u.ms
    voltage = -65 * u.mV
    current = 2 * u.nA

brainstate
~~~~~~~~~~

Core framework (used automatically):

.. code-block:: python

    import brainstate

    # Module system
    class Net(brainstate.nn.Module): ...

    # Compilation
    @brainstate.compile.jit
    def fn(): ...

    # Transformations
    result = brainstate.transform.for_loop(...)

Data Flow Example
-----------------

Here's how data flows through a typical BrainPy 3.0 simulation:

.. code-block:: python

    # 1. Define network
    class EINetwork(brainstate.nn.Module):
        def __init__(self):
            super().__init__()
            self.E = brainpy.LIF(800)  # States: V, spike
            self.I = brainpy.LIF(200)  # States: V, spike
            self.E2E = brainpy.AlignPostProj(...)  # States: g (in synapse)
            self.E2I = brainpy.AlignPostProj(...)
            self.I2E = brainpy.AlignPostProj(...)
            self.I2I = brainpy.AlignPostProj(...)

        def update(self, input):
            # Get spikes from last time step
            e_spikes = self.E.get_spike()
            i_spikes = self.I.get_spike()

            # Update projections (spikes → synaptic currents)
            self.E2E(e_spikes)  # Updates E2E.syn.g
            self.E2I(e_spikes)
            self.I2E(i_spikes)
            self.I2I(i_spikes)

            # Update neurons (currents → new V and spikes)
            self.E(input)  # Updates E.V and E.spike
            self.I(input)  # Updates I.V and I.spike

            return e_spikes, i_spikes

    # 2. Initialize
    net = EINetwork()
    brainstate.nn.init_all_states(net)

    # 3. Compile
    @brainstate.compile.jit
    def step(input):
        return net.update(input)

    # 4. Simulate
    times = u.math.arange(0*u.ms, 1000*u.ms, 0.1*u.ms)
    results = brainstate.transform.for_loop(step, times)

State Flow:

.. code-block:: text

    Time t:
    ┌──────────────────────────────────────────┐
    │  States at t-1:                          │
    │    E.V[t-1], E.spike[t-1]               │
    │    I.V[t-1], I.spike[t-1]               │
    │    E2E.syn.g[t-1], ...                  │
    └──────────────────────────────────────────┘
                    ↓
    ┌──────────────────────────────────────────┐
    │  Projection Updates:                     │
    │    E2E.syn.g[t] = f(g[t-1], E.spike[t-1])│
    │    ... (other projections)               │
    └──────────────────────────────────────────┘
                    ↓
    ┌──────────────────────────────────────────┐
    │  Neuron Updates:                         │
    │    E.V[t] = f(V[t-1], Σ currents[t])   │
    │    E.spike[t] = E.V[t] >= V_th          │
    │    ... (other neurons)                   │
    └──────────────────────────────────────────┘
                    ↓
    Time t+1...

Performance Considerations
--------------------------

Memory Management
~~~~~~~~~~~~~~~~~

- States are preallocated
- In-place updates when possible
- Efficient batching support
- Automatic garbage collection

Compilation Strategy
~~~~~~~~~~~~~~~~~~~~

- Compile simulation loops
- Batch operations when possible
- Use ``for_loop`` for long sequences
- Leverage JAX's XLA optimization

Hardware Acceleration
~~~~~~~~~~~~~~~~~~~~~

- Automatic GPU dispatch for large arrays
- TPU support for massive parallelism
- Efficient CPU fallback for small problems

Summary
-------

BrainPy 3.0's architecture provides:

✅ **Clear Abstractions**: Neurons, synapses, and projections with well-defined roles

✅ **State Management**: Explicit, efficient handling of dynamical variables

✅ **Modularity**: Compose complex models from simple components

✅ **Performance**: JIT compilation and hardware acceleration

✅ **Scientific Accuracy**: Integrated physical units

✅ **Extensibility**: Easy to add custom components

✅ **Modern Design**: Built on proven frameworks (JAX, brainstate)

Next Steps
----------

- Learn about specific components: :doc:`neurons`, :doc:`synapses`, :doc:`projections`
- Understand state management in depth: :doc:`state-management`
- See practical examples: :doc:`../tutorials/basic/01-lif-neuron`
- Explore the ecosystem: `brainstate docs <https://brainstate.readthedocs.io/>`_
