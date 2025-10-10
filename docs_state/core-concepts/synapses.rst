Synapses
========

Synapses model the temporal dynamics of neural connections in ``brainpy.state``. This document explains
how synapses work, what models are available, and how to use them effectively.

Overview
--------

Synapses provide temporal filtering of spike trains, transforming discrete spikes into continuous currents or conductances. They model:

- **Postsynaptic potentials** (PSPs)
- **Temporal integration** of spike trains
- **Synaptic dynamics** (rise and decay)

In BrainPy's architecture, synapses are part of the projection system:

.. code-block:: text

    Spikes → [Connectivity] → [Synapse] → [Output] → Neurons
                                  ↑
                          Temporal filtering

Basic Usage
-----------

Creating Synapses
~~~~~~~~~~~~~~~~~

Synapses are typically created as part of projections:

.. code-block:: python

    import brainpy
    import brainunit as u

    # Create synapse descriptor
    syn = brainpy.state.Expon.desc(
        size=100,           # Number of synapses
        tau=5. * u.ms       # Time constant
    )

    # Use in projection
    projection = brainpy.state.AlignPostProj(
        comm=...,
        syn=syn,            # Synapse here
        out=...,
        post=neurons
    )

Synapse Lifecycle
~~~~~~~~~~~~~~~~~

1. **Creation**: Define synapse with `.desc()` method
2. **Integration**: Include in projection
3. **Update**: Called automatically by projection
4. **Access**: Read synaptic variables as needed

.. code-block:: python

    # During simulation
    projection(presynaptic_spikes)  # Updates synapse internally

    # Access synaptic variable
    synaptic_current = projection.syn.g.value

Available Synapse Models
------------------------

Expon (Single Exponential)
~~~~~~~~~~~~~~~~~~~~~~~~~~~

The simplest and most commonly used synapse model.

**Mathematical Model:**

.. math::

    \\tau \\frac{dg}{dt} = -g

When spike arrives: :math:`g \\leftarrow g + 1`

**Impulse Response:**

.. math::

    g(t) = \\exp(-t/\\tau)

**Example:**

.. code-block:: python

    syn = brainpy.state.Expon.desc(
        size=100,
        tau=5. * u.ms,
        g_initializer=braintools.init.Constant(0. * u.mS)
    )

**Parameters:**

- ``size``: Number of synapses
- ``tau``: Decay time constant
- ``g_initializer``: Initial synaptic variable (optional)

**Key Features:**

- Single time constant
- Fast computation
- Instantaneous rise

**Use cases:**

- General-purpose modeling
- Fast simulations
- When precise kinetics are not critical

**Behavior:**

.. code-block:: python

    # Response to single spike at t=0
    # g(t) = exp(-t/τ)
    # Fast rise, exponential decay

Alpha Synapse
~~~~~~~~~~~~~

A more realistic model with non-instantaneous rise time.

**Mathematical Model:**

.. math::

    \\tau \\frac{dh}{dt} &= -h

    \\tau \\frac{dg}{dt} &= -g + h

When spike arrives: :math:`h \\leftarrow h + 1`

**Impulse Response:**

.. math::

    g(t) = \\frac{t}{\\tau} \\exp(-t/\\tau)

**Example:**

.. code-block:: python

    syn = brainpy.state.Alpha.desc(
        size=100,
        tau=5. * u.ms,
        g_initializer=braintools.init.Constant(0. * u.mS)
    )

**Parameters:**

Same as Expon, but produces alpha-shaped response.

**Key Features:**

- Smooth rise and fall
- Biologically realistic
- Peak at t = τ

**Use cases:**

- Biological realism
- Detailed cortical modeling
- When kinetics matter

**Behavior:**

.. code-block:: python

    # Response to single spike at t=0
    # g(t) = (t/τ) * exp(-t/τ)
    # Gradual rise to peak at τ, then decay

AMPA (Excitatory)
~~~~~~~~~~~~~~~~~

Models AMPA receptor dynamics for excitatory synapses.

**Mathematical Model:**

Similar to Alpha, but with parameters tuned for AMPA receptors.

**Example:**

.. code-block:: python

    syn = brainpy.state.AMPA.desc(
        size=100,
        tau=2. * u.ms,  # Fast AMPA kinetics
        g_initializer=braintools.init.Constant(0. * u.mS)
    )

**Key Features:**

- Fast kinetics (τ ≈ 2 ms)
- Excitatory receptor
- Biologically parameterized

**Use cases:**

- Excitatory synapses
- Cortical pyramidal neurons
- Biological realism

GABAa (Inhibitory)
~~~~~~~~~~~~~~~~~~

Models GABAa receptor dynamics for inhibitory synapses.

**Mathematical Model:**

Similar to Alpha, but with parameters tuned for GABAa receptors.

**Example:**

.. code-block:: python

    syn = brainpy.state.GABAa.desc(
        size=100,
        tau=10. * u.ms,  # Slower GABAa kinetics
        g_initializer=braintools.init.Constant(0. * u.mS)
    )

**Key Features:**

- Slower kinetics (τ ≈ 10 ms)
- Inhibitory receptor
- Biologically parameterized

**Use cases:**

- Inhibitory synapses
- GABAergic interneurons
- Biological realism

Synaptic Variables
------------------

The Descriptor Pattern
~~~~~~~~~~~~~~~~~~~~~~~

BrainPy synapses use a descriptor pattern:

.. code-block:: python

    # Create descriptor (not yet instantiated)
    syn_desc = brainpy.state.Expon.desc(size=100, tau=5*u.ms)

    # Instantiated within projection
    projection = brainpy.state.AlignPostProj(..., syn=syn_desc, ...)

    # Access instantiated synapse
    actual_synapse = projection.syn
    g_value = actual_synapse.g.value

Why Descriptors?
~~~~~~~~~~~~~~~~

- **Deferred instantiation**: Created when needed
- **Reusability**: Same descriptor for multiple projections
- **Flexibility**: Configure before instantiation

Accessing Synaptic State
~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    # Within projection
    projection = brainpy.state.AlignPostProj(
        comm=...,
        syn=brainpy.state.Expon.desc(100, tau=5*u.ms),
        out=...,
        post=neurons
    )

    # After simulation step
    synaptic_var = projection.syn.g.value  # Current value with units

    # Convert to array for plotting
    g_array = synaptic_var.to_decimal(u.mS)

Synaptic Dynamics Visualization
--------------------------------

Comparing Different Models
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    import brainpy as bp
    import brainstate
    import brainunit as u
    import matplotlib.pyplot as plt
    import jax.numpy as jnp

    brainstate.environ.set(dt=0.1 * u.ms)

    # Create different synapses
    expon = brainpy.state.Expon(100, tau=5*u.ms)
    alpha = brainpy.state.Alpha(100, tau=5*u.ms)
    ampa = brainpy.state.AMPA(100, tau=2*u.ms)
    gaba = brainpy.state.GABAa(100, tau=10*u.ms)

    # Initialize
    for syn in [expon, alpha, ampa, gaba]:
        brainstate.nn.init_all_states(syn)

    # Single spike at t=0
    spike_input = jnp.zeros(100)
    spike_input = spike_input.at[0].set(1.0)

    # Simulate
    times = u.math.arange(0*u.ms, 50*u.ms, 0.1*u.ms)
    responses = {
        'Expon': [],
        'Alpha': [],
        'AMPA': [],
        'GABAa': []
    }

    for syn, name in zip([expon, alpha, ampa, gaba],
                         ['Expon', 'Alpha', 'AMPA', 'GABAa']):
        brainstate.nn.init_all_states(syn)
        for i, t in enumerate(times):
            if i == 0:
                syn(spike_input)
            else:
                syn(jnp.zeros(100))
            responses[name].append(syn.g.value[0])

    # Plot
    plt.figure(figsize=(10, 6))
    for name, response in responses.items():
        response_array = u.math.asarray(response)
        plt.plot(times.to_decimal(u.ms),
                response_array.to_decimal(u.mS),
                label=name, linewidth=2)

    plt.xlabel('Time (ms)')
    plt.ylabel('Synaptic Variable (mS)')
    plt.title('Comparison of Synapse Models (Single Spike)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

Integration with Projections
-----------------------------

Complete Example
~~~~~~~~~~~~~~~~

.. code-block:: python

    import brainpy as bp
    import brainstate
    import brainunit as u

    # Create neurons
    pre_neurons = brainpy.state.LIF(80, V_th=-50*u.mV, tau=10*u.ms)
    post_neurons = brainpy.state.LIF(100, V_th=-50*u.mV, tau=10*u.ms)

    # Create projection with exponential synapse
    projection = brainpy.state.AlignPostProj(
        comm=brainstate.nn.EventFixedProb(
            80, 100, prob=0.1, weight=0.5*u.mS
        ),
        syn=brainpy.state.Expon.desc(100, tau=5*u.ms),
        out=brainpy.state.CUBA.desc(),
        post=post_neurons
    )

    # Initialize
    brainstate.nn.init_all_states(pre_neurons)
    brainstate.nn.init_all_states(post_neurons)

    # Simulation
    def update(input_current):
        # Update presynaptic neurons
        pre_neurons(input_current)

        # Get spikes and propagate through projection
        spikes = pre_neurons.get_spike()
        projection(spikes)

        # Update postsynaptic neurons
        post_neurons(0 * u.nA)

        return post_neurons.get_spike()

    # Run
    times = u.math.arange(0*u.ms, 100*u.ms, 0.1*u.ms)
    results = brainstate.transform.for_loop(
        lambda t: update(2*u.nA),
        times
    )

Short-Term Plasticity
---------------------

Synapses can be combined with short-term plasticity (STP):

.. code-block:: python

    # Create projection with STP
    projection = brainpy.state.AlignPostProj(
        comm=brainstate.nn.EventFixedProb(80, 100, prob=0.1, weight=0.5*u.mS),
        syn=brainpy.state.STP.desc(
            brainpy.state.Expon.desc(100, tau=5*u.ms),  # Underlying synapse
            tau_f=200*u.ms,   # Facilitation time constant
            tau_d=150*u.ms,   # Depression time constant
            U=0.2             # Utilization of synaptic efficacy
        ),
        out=brainpy.state.CUBA.desc(),
        post=post_neurons
    )

See :doc:`plasticity` for more details on STP.

Custom Synapses
---------------

Creating Custom Synapse Models
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

You can create custom synapse models by inheriting from ``Synapse``:

.. code-block:: python

    import brainstate
    from brainpy.state import Synapse

    class MyCustomSynapse(Synapse):
        def __init__(self, size, tau1, tau2, **kwargs):
            super().__init__(size, **kwargs)

            self.tau1 = tau1
            self.tau2 = tau2

            # Synaptic variable
            self.g = brainstate.ShortTermState(
                braintools.init.Constant(0., unit=u.mS)(size)
            )

        def update(self, spike_input):
            dt = brainstate.environ.get_dt()

            # Custom dynamics (double exponential)
            dg = (-self.g.value / self.tau1 +
                  spike_input / self.tau2)
            self.g.value = self.g.value + dg * dt

            return self.g.value

        @classmethod
        def desc(cls, size, tau1, tau2, **kwargs):
            """Descriptor for deferred instantiation."""
            def create():
                return cls(size, tau1, tau2, **kwargs)
            return create

Usage:

.. code-block:: python

    # Create descriptor
    syn_desc = MyCustomSynapse.desc(
        size=100,
        tau1=5*u.ms,
        tau2=10*u.ms
    )

    # Use in projection
    projection = brainpy.state.AlignPostProj(..., syn=syn_desc, ...)

Choosing the Right Synapse
---------------------------

Decision Guide
~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 20 30 25 25

   * - Model
     - When to Use
     - Pros
     - Cons
   * - Expon
     - General purpose, speed
     - Fast, simple
     - Unrealistic rise
   * - Alpha
     - Biological realism
     - Realistic kinetics
     - Slower computation
   * - AMPA
     - Excitatory, fast
     - Biologically accurate
     - Specific use case
   * - GABAa
     - Inhibitory, slow
     - Biologically accurate
     - Specific use case

Recommendations
~~~~~~~~~~~~~~~

**For machine learning / SNNs:**
   Use ``Expon`` for speed and simplicity.

**For biological modeling:**
   Use ``Alpha``, ``AMPA``, or ``GABAa`` for realism.

**For cortical networks:**
   - Excitatory: ``AMPA`` (τ ≈ 2 ms)
   - Inhibitory: ``GABAa`` (τ ≈ 10 ms)

**For custom dynamics:**
   Implement custom synapse class.

Performance Considerations
--------------------------

Computational Cost
~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 25 25 50

   * - Model
     - Relative Cost
     - Notes
   * - Expon
     - 1x (baseline)
     - Single state variable
   * - Alpha
     - 2x
     - Two state variables
   * - AMPA/GABAa
     - 2x
     - Similar to Alpha

Optimization Tips
~~~~~~~~~~~~~~~~~

1. **Use Expon when possible**: Fastest option

2. **Batch operations**: Multiple synapses together

   .. code-block:: python

       # Good: Single projection with 1000 synapses
       proj = brainpy.state.AlignPostProj(..., syn=brainpy.state.Expon.desc(1000, ...))

       # Bad: 1000 separate projections
       projs = [brainpy.state.AlignPostProj(..., syn=brainpy.state.Expon.desc(1, ...))
                for _ in range(1000)]

3. **JIT compilation**: Always use for simulations

   .. code-block:: python

       @brainstate.transform.jit
       def step():
           projection(spikes)
           neurons(0*u.nA)

Common Patterns
---------------

Excitatory-Inhibitory Balance
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    # Excitatory projection (fast)
    E_proj = brainpy.state.AlignPostProj(
        comm=...,
        syn=brainpy.state.Expon.desc(post_size, tau=2*u.ms),
        out=brainpy.state.CUBA.desc(),
        post=neurons
    )

    # Inhibitory projection (slow)
    I_proj = brainpy.state.AlignPostProj(
        comm=...,
        syn=brainpy.state.Expon.desc(post_size, tau=10*u.ms),
        out=brainpy.state.CUBA.desc(),
        post=neurons
    )

Multiple Receptor Types
~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    # AMPA (fast excitatory)
    ampa_proj = brainpy.state.AlignPostProj(
        ..., syn=brainpy.state.AMPA.desc(size, tau=2*u.ms), ...
    )

    # NMDA (slow excitatory) - custom
    nmda_proj = brainpy.state.AlignPostProj(
        ..., syn=CustomNMDA.desc(size, tau=100*u.ms), ...
    )

    # GABAa (fast inhibitory)
    gaba_proj = brainpy.state.AlignPostProj(
        ..., syn=brainpy.state.GABAa.desc(size, tau=10*u.ms), ...
    )

Summary
-------

Synapses in ``brainpy.state``:

✅ **Multiple models**: Expon, Alpha, AMPA, GABAa

✅ **Temporal filtering**: Convert spikes to continuous signals

✅ **Descriptor pattern**: Flexible, reusable configuration

✅ **Integration ready**: Seamless use in projections

✅ **Extensible**: Easy custom synapse models

✅ **Physical units**: Proper unit handling throughout

Next Steps
----------

- Learn about :doc:`projections` for complete connectivity
- Explore :doc:`plasticity` for learning rules
- Follow :doc:`../tutorials/basic/02-synapse-models` for practice
- See :doc:`../examples/classical-networks/ei-balanced` for network examples
