Core Concepts Overview
======================

``brainpy.state`` introduces a modern, state-based architecture built on top of ``brainstate``. This overview will
help you understand the key concepts and design philosophy.

What's New
----------

``brainpy.state`` has been completely rewritten to provide:

- **State-based programming**: Built on ``brainstate`` for efficient state management
- **Modular architecture**: Clear separation of concerns (communication, dynamics, outputs)
- **Physical units**: Integration with ``brainunit`` for scientifically accurate simulations
- **Modern API**: Cleaner, more intuitive interfaces
- **Better performance**: Optimized JIT compilation and memory management

Key Architectural Components
-----------------------------

``brainpy.state`` is organized around several core concepts:

1. State Management
~~~~~~~~~~~~~~~~~~~

Everything in ``brainpy.state`` revolves around **states**. States are variables that persist across time steps:

- ``brainstate.State``: Base state container
- ``brainstate.ParamState``: Trainable parameters
- ``brainstate.ShortTermState``: Temporary variables

States enable:

- Automatic differentiation for training
- Efficient memory management
- Batching and parallelization

2. Neurons
~~~~~~~~~~

Neurons are the fundamental computational units:

.. code-block:: python

   import brainpy
   import brainunit as u

   # Create a population of 100 LIF neurons
   neurons = brainpy.state.LIF(100, tau=10*u.ms, V_th=-50*u.mV)

Key neuron models:

- ``brainpy.state.IF``: Integrate-and-Fire
- ``brainpy.state.LIF``: Leaky Integrate-and-Fire
- ``brainpy.state.LIFRef``: LIF with refractory period
- ``brainpy.state.ALIF``: Adaptive LIF

3. Synapses
~~~~~~~~~~~

Synapses model the dynamics of neural connections:

.. code-block:: python

   # Exponential synapse
   synapse = brainpy.state.Expon(100, tau=5*u.ms)

   # Alpha synapse (more realistic)
   synapse = brainpy.state.Alpha(100, tau=5*u.ms)

Synapse models:

- ``brainpy.state.Expon``: Single exponential decay
- ``brainpy.state.Alpha``: Double exponential (alpha function)
- ``brainpy.state.AMPA``: Excitatory receptor dynamics
- ``brainpy.state.GABAa``: Inhibitory receptor dynamics

4. Projections
~~~~~~~~~~~~~~

Projections connect neural populations:

.. code-block:: python

   projection = brainpy.state.AlignPostProj(
       comm=brainstate.nn.EventFixedProb(N_pre, N_post, prob=0.1, weight=0.5),
       syn=brainpy.state.Expon.desc(N_post, tau=5*u.ms),
       out=brainpy.state.CUBA.desc(),
       post=neurons
   )

The projection architecture separates:

- **Communication**: How spikes are transmitted (connectivity, weights)
- **Synaptic dynamics**: How synapses respond (temporal filtering)
- **Output mechanism**: How synaptic currents affect neurons (CUBA/COBA)

5. Networks
~~~~~~~~~~~

Networks combine neurons and projections:

.. code-block:: python

   import brainstate

   class EINet(brainstate.nn.Module):
       def __init__(self):
           super().__init__()
           self.E = brainpy.state.LIF(800)
           self.I = brainpy.state.LIF(200)
           self.E2E = brainpy.state.AlignPostProj(...)
           self.E2I = brainpy.state.AlignPostProj(...)
           # ... more projections

       def update(self, input):
           # Define network dynamics
           pass

Computational Model
-------------------

Time-Stepped Simulation
~~~~~~~~~~~~~~~~~~~~~~~

BrainPy uses discrete time steps for simulation:

.. code-block:: python

   import brainstate
   import brainunit as u

   # Set simulation time step
   brainstate.environ.set(dt=0.1 * u.ms)

   # Run simulation
   times = u.math.arange(0*u.ms, 1000*u.ms, brainstate.environ.get_dt())
   results = brainstate.transform.for_loop(network.update, times)

JIT Compilation
~~~~~~~~~~~~~~~

BrainPy leverages JAX for Just-In-Time compilation:

.. code-block:: python

   @brainstate.transform.jit
   def simulate():
       return network.update(input)

   # First call compiles, subsequent calls are fast
   result = simulate()

Benefits:

- Near-C performance
- Automatic GPU/TPU dispatch
- Optimized memory usage

Physical Units
~~~~~~~~~~~~~~

BrainPy 3.0 integrates ``brainunit`` for scientific accuracy:

.. code-block:: python

   import brainunit as u

   # Define parameters with units
   tau = 10 * u.ms
   V_threshold = -50 * u.mV
   current = 5 * u.nA

   # Units are checked automatically
   neurons = brainpy.state.LIF(100, tau=tau, V_th=V_threshold)

This prevents unit-related bugs and makes code self-documenting.

Training and Learning
---------------------

BrainPy 3.0 supports gradient-based training:

.. code-block:: python

   import braintools

   # Define optimizer
   optimizer = braintools.optim.Adam(lr=1e-3)
   optimizer.register_trainable_weights(net.states(brainstate.ParamState))

   # Define loss function
   def loss_fn():
       predictions = brainstate.transform.for_loop(net.update, inputs)
       return loss(predictions, targets)

   # Training step
   @brainstate.transform.jit
   def train_step():
       grads, loss = brainstate.transform.grad(
           loss_fn,
           net.states(brainstate.ParamState),
           return_value=True
       )()
       optimizer.update(grads)
       return loss

Key features:

- Surrogate gradients for spiking neurons
- Automatic differentiation
- Various optimizers (Adam, SGD, etc.)

Ecosystem Components
--------------------

``brainpy.state`` is part of a larger ecosystem:

brainstate
~~~~~~~~~~

The foundation for state management and compilation:

- State-based IR construction
- JIT compilation
- Program augmentation (batching, etc.)

brainunit
~~~~~~~~~

Physical units system:

- SI units support
- Automatic unit checking
- Unit conversions

braintools
~~~~~~~~~~

Utilities and tools:

- Optimizers (``braintools.optim``)
- Initialization (``braintools.init``)
- Metrics and losses (``braintools.metric``)
- Surrogate gradients (``braintools.surrogate``)
- Visualization (``braintools.visualize``)

Design Philosophy
-----------------

``brainpy.state`` follows these principles:

1. **Explicit over implicit**: Clear, readable code
2. **Modular composition**: Build complex models from simple components
3. **Performance by default**: JIT compilation and optimization built-in
4. **Scientific accuracy**: Physical units and biologically realistic models
5. **Extensibility**: Easy to add custom components

Next Steps
----------

Now that you understand the core concepts:

- Try the :doc:`5-minute tutorial <5min-tutorial>` to get hands-on experience
- Read the :doc:`detailed core concepts <../core-concepts/architecture>` documentation
- Explore :doc:`basic tutorials <../tutorials/basic/01-lif-neuron>` to learn each component
- Check out the :doc:`examples gallery <../examples/gallery>` for real-world models
