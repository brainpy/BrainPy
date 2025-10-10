Neurons
=======

Neurons are the fundamental computational units in ``brainpy.state``. This document explains how neurons work, what models are available, and how to use and create them.

Overview
--------

In ``brainpy.state``, neurons model the dynamics of neural populations. Each neuron model:

- Maintains **membrane potential** (voltage)
- Integrates **input currents**
- Generates **spikes** when threshold is crossed
- **Resets** after spiking (various strategies)

All neuron models inherit from the base ``Neuron`` class and follow consistent interfaces.

Basic Usage
-----------

Creating Neurons
~~~~~~~~~~~~~~~~

.. code-block:: python

    import brainpy
    import brainunit as u

    # Create a population of 100 LIF neurons
    neurons = brainpy.state.LIF(
        size=100,
        V_rest=-65. * u.mV,
        V_th=-50. * u.mV,
        V_reset=-65. * u.mV,
        tau=10. * u.ms
    )

Initializing States
~~~~~~~~~~~~~~~~~~~

Before simulation, initialize neuron states:

.. code-block:: python

    import brainstate

    # Initialize all states to default values
    brainstate.nn.init_all_states(neurons)

    # Or with specific batch size
    brainstate.nn.init_all_states(neurons, batch_size=32)

Running Neurons
~~~~~~~~~~~~~~~

Update neurons by calling them with input current:

.. code-block:: python

    # Single time step
    input_current = 2.0 * u.nA
    neurons(input_current)

    # Access results
    voltage = neurons.V.value          # Membrane potential
    spikes = neurons.get_spike()       # Spike output

Available Neuron Models
-----------------------

IF (Integrate-and-Fire)
~~~~~~~~~~~~~~~~~~~~~~~

The simplest spiking neuron model.

**Mathematical Model:**

.. math::

    \\tau \\frac{dV}{dt} = -V + R \\cdot I(t)

**Spike condition:** If :math:`V \\geq V_{th}`, emit spike and reset.

**Example:**

.. code-block:: python

    neuron = brainpy.state.IF(
        size=100,
        V_rest=0. * u.mV,
        V_th=1. * u.mV,
        V_reset=0. * u.mV,
        tau=20. * u.ms,
        R=1. * u.ohm
    )

**Parameters:**

- ``size``: Number of neurons
- ``V_rest``: Resting potential
- ``V_th``: Spike threshold
- ``V_reset``: Reset potential after spike
- ``tau``: Membrane time constant
- ``R``: Input resistance

**Use cases:**

- Simple rate coding
- Fast simulations
- Theoretical studies

LIF (Leaky Integrate-and-Fire)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The most commonly used spiking neuron model.

**Mathematical Model:**

.. math::

    \\tau \\frac{dV}{dt} = -(V - V_{rest}) + R \\cdot I(t)

**Spike condition:** If :math:`V \\geq V_{th}`, emit spike and reset.

**Example:**

.. code-block:: python

    neuron = brainpy.state.LIF(
        size=100,
        V_rest=-65. * u.mV,
        V_th=-50. * u.mV,
        V_reset=-65. * u.mV,
        tau=10. * u.ms,
        R=1. * u.ohm,
        V_initializer=braintools.init.Normal(-65., 5., unit=u.mV)
    )

**Parameters:**

All IF parameters, plus:

- ``V_initializer``: How to initialize membrane potential

**Key Features:**

- Leak toward resting potential
- Realistic temporal integration
- Well-studied dynamics

**Use cases:**

- General spiking neural networks
- Cortical neuron modeling
- Learning and training

LIFRef (LIF with Refractory Period)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

LIF neuron with absolute refractory period.

**Mathematical Model:**

Same as LIF, but after spiking:

- Neuron is "frozen" for refractory period
- No integration during refractory period
- More biologically realistic

**Example:**

.. code-block:: python

    neuron = brainpy.state.LIFRef(
        size=100,
        V_rest=-65. * u.mV,
        V_th=-50. * u.mV,
        V_reset=-65. * u.mV,
        tau=10. * u.ms,
        tau_ref=2. * u.ms,  # Refractory period
        R=1. * u.ohm
    )

**Additional Parameters:**

- ``tau_ref``: Refractory period duration

**Key Features:**

- Absolute refractory period
- Prevents immediate re-firing
- More realistic spike timing

**Use cases:**

- Precise temporal coding
- Biological realism
- Rate regulation

ALIF (Adaptive Leaky Integrate-and-Fire)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

LIF with spike-frequency adaptation.

**Mathematical Model:**

.. math::

    \\tau \\frac{dV}{dt} &= -(V - V_{rest}) - R \\cdot w + R \\cdot I(t)

    \\tau_w \\frac{dw}{dt} &= -w

When spike occurs: :math:`w \\leftarrow w + \\beta`

**Example:**

.. code-block:: python

    neuron = brainpy.state.ALIF(
        size=100,
        V_rest=-65. * u.mV,
        V_th=-50. * u.mV,
        V_reset=-65. * u.mV,
        tau=10. * u.ms,
        tau_w=200. * u.ms,   # Adaptation time constant
        beta=0.01,           # Adaptation strength
        R=1. * u.ohm
    )

**Additional Parameters:**

- ``tau_w``: Adaptation time constant
- ``beta``: Adaptation increment per spike

**Key Features:**

- Spike-frequency adaptation
- Reduced firing with sustained input
- More complex dynamics

**Use cases:**

- Cortical neuron modeling
- Sensory adaptation
- Complex temporal patterns

Reset Modes
-----------

BrainPy supports different reset behaviors after spiking:

Soft Reset (Default)
~~~~~~~~~~~~~~~~~~~~

Subtract threshold from membrane potential:

.. math::

    V \\leftarrow V - V_{th}

.. code-block:: python

    neuron = brainpy.state.LIF(..., spk_reset='soft')

**Properties:**

- Preserves extra charge above threshold
- Allows rapid re-firing
- Common in machine learning

Hard Reset
~~~~~~~~~~

Reset to fixed potential:

.. math::

    V \\leftarrow V_{reset}

.. code-block:: python

    neuron = brainpy.state.LIF(..., spk_reset='hard')

**Properties:**

- Discards extra charge
- More biologically realistic
- Prevents immediate re-firing

Choosing Reset Mode
~~~~~~~~~~~~~~~~~~~~

- **Soft reset**: Machine learning, rate coding, fast oscillations
- **Hard reset**: Biological modeling, temporal coding, realism

Spike Functions
---------------

For training spiking neural networks, use surrogate gradients:

.. code-block:: python

    import braintools

    neuron = brainpy.state.LIF(
        size=100,
        ...,
        spk_fun=braintools.surrogate.ReluGrad()
    )

Available surrogate functions:

- ``ReluGrad()``: ReLU-like gradient
- ``SigmoidGrad()``: Sigmoid-like gradient
- ``GaussianGrad()``: Gaussian-like gradient
- ``SuperSpike()``: SuperSpike surrogate

See :doc:`../tutorials/advanced/03-snn-training` for training details.

Advanced Features
-----------------

Initialization Strategies
~~~~~~~~~~~~~~~~~~~~~~~~~

Different ways to initialize membrane potential:

.. code-block:: python

    import braintools

    # Constant initialization
    neuron = brainpy.state.LIF(
        size=100,
        V_initializer=braintools.init.Constant(-65., unit=u.mV)
    )

    # Normal distribution
    neuron = brainpy.state.LIF(
        size=100,
        V_initializer=braintools.init.Normal(-65., 5., unit=u.mV)
    )

    # Uniform distribution
    neuron = brainpy.state.LIF(
        size=100,
        V_initializer=braintools.init.Uniform(-70., -60., unit=u.mV)
    )

Accessing Neuron States
~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    # Membrane potential (with units)
    voltage = neuron.V.value  # Quantity with units

    # Spike output (binary or real-valued)
    spikes = neuron.get_spike()

    # Access underlying array (without units)
    voltage_array = neuron.V.value.to_decimal(u.mV)

Batched Simulation
~~~~~~~~~~~~~~~~~~

Simulate multiple trials in parallel:

.. code-block:: python

    # Initialize with batch dimension
    brainstate.nn.init_all_states(neuron, batch_size=32)

    # Input shape: (batch_size,) or (batch_size, size)
    input_current = jnp.ones((32, 100)) * 2.0 * u.nA
    neuron(input_current)

    # Output shape: (batch_size, size)
    spikes = neuron.get_spike()

Complete Example
----------------

Here's a complete example simulating a LIF neuron:

.. code-block:: python

    import brainpy
    import brainstate
    import brainunit as u
    import matplotlib.pyplot as plt

    # Set time step
    brainstate.environ.set(dt=0.1 * u.ms)

    # Create neuron
    neuron = brainpy.state.LIF(
        size=1,
        V_rest=-65. * u.mV,
        V_th=-50. * u.mV,
        V_reset=-65. * u.mV,
        tau=10. * u.ms,
        spk_reset='hard'
    )

    # Initialize
    brainstate.nn.init_all_states(neuron)

    # Simulation parameters
    duration = 200. * u.ms
    dt = brainstate.environ.get_dt()
    times = u.math.arange(0. * u.ms, duration, dt)

    # Input current (step input)
    def get_input(t):
        return 2.0 * u.nA if t > 50*u.ms else 0.0 * u.nA

    # Run simulation
    voltages = []
    spikes = []

    for t in times:
        neuron(get_input(t))
        voltages.append(neuron.V.value)
        spikes.append(neuron.get_spike())

    # Plot results
    voltages = u.math.asarray(voltages)
    times_plot = times.to_decimal(u.ms)
    voltages_plot = voltages.to_decimal(u.mV)

    plt.figure(figsize=(10, 4))
    plt.plot(times_plot, voltages_plot)
    plt.axhline(y=-50, color='r', linestyle='--', label='Threshold')
    plt.xlabel('Time (ms)')
    plt.ylabel('Membrane Potential (mV)')
    plt.title('LIF Neuron Response')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

Creating Custom Neurons
------------------------

You can create custom neuron models by inheriting from ``Neuron``:

.. code-block:: python

    import brainstate
    from brainpy.state import Neuron

    class MyNeuron(Neuron):
        def __init__(self, size, tau, V_th, **kwargs):
            super().__init__(size, **kwargs)

            # Store parameters
            self.tau = tau
            self.V_th = V_th

            # Initialize states
            self.V = brainstate.ShortTermState(
                braintools.init.Constant(0., unit=u.mV)(size)
            )
            self.spike = brainstate.ShortTermState(
                jnp.zeros(size)
            )

        def update(self, x):
            # Get time step
            dt = brainstate.environ.get_dt()

            # Update membrane potential (custom dynamics)
            dV = (-self.V.value + x) / self.tau * dt
            V_new = self.V.value + dV

            # Check for spikes
            spike = (V_new >= self.V_th).astype(float)

            # Reset
            V_new = jnp.where(spike > 0, 0. * u.mV, V_new)

            # Update states
            self.V.value = V_new
            self.spike.value = spike

            return spike

        def get_spike(self):
            return self.spike.value

Usage:

.. code-block:: python

    neuron = MyNeuron(size=100, tau=10*u.ms, V_th=1*u.mV)
    brainstate.nn.init_all_states(neuron)
    neuron(input_current)

Performance Tips
----------------

1. **Use JIT compilation** for repeated simulations:

   .. code-block:: python

       @brainstate.transform.jit
       def simulate_step(input):
           neuron(input)
           return neuron.V.value

2. **Batch multiple trials** for parallelism:

   .. code-block:: python

       brainstate.nn.init_all_states(neuron, batch_size=100)

3. **Use appropriate data types**:

   .. code-block:: python

       # Float32 is usually sufficient and faster
       brainstate.environ.set(dtype=jnp.float32)

4. **Preallocate arrays** when recording:

   .. code-block:: python

       n_steps = len(times)
       voltages = jnp.zeros((n_steps, neuron.size))

Common Patterns
---------------

Rate Coding
~~~~~~~~~~~

Neurons encoding information in firing rate:

.. code-block:: python

    neuron = brainpy.state.LIF(100, tau=10*u.ms, spk_reset='soft')
    # Use soft reset for higher firing rates

Temporal Coding
~~~~~~~~~~~~~~~

Neurons encoding information in spike timing:

.. code-block:: python

    neuron = brainpy.state.LIFRef(
        100,
        tau=10*u.ms,
        tau_ref=2*u.ms,
        spk_reset='hard'
    )
    # Use refractory period for precise timing

Burst Firing
~~~~~~~~~~~~

Neurons with bursting behavior:

.. code-block:: python

    neuron = brainpy.state.ALIF(
        100,
        tau=10*u.ms,
        tau_w=200*u.ms,
        beta=0.01,
        spk_reset='soft'
    )
    # Adaptation creates bursting patterns

Summary
-------

Neurons in ``brainpy.state``:

✅ **Multiple models**: IF, LIF, LIFRef, ALIF

✅ **Physical units**: All parameters with proper units

✅ **Flexible reset**: Soft or hard reset modes

✅ **Training-ready**: Surrogate gradients for learning

✅ **High performance**: JIT compilation and batching

✅ **Extensible**: Easy to create custom models

Next Steps
----------

- Learn about :doc:`synapses` to connect neurons
- Explore :doc:`projections` for network connectivity
- Follow :doc:`../tutorials/basic/01-lif-neuron` for hands-on practice
- See :doc:`../examples/classical-networks/ei-balanced` for network examples
