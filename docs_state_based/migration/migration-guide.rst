Migration Guide: BrainPy 2.x to 3.0
====================================

This guide helps you migrate your code from BrainPy 2.x to BrainPy 3.0. BrainPy 3.0 represents a complete rewrite built on ``brainstate``, with significant architectural changes and API improvements.

Overview of Changes
-------------------

BrainPy 3.0 introduces several major changes:

**Architecture**
   - Built on ``brainstate`` framework
   - State-based programming model
   - Integrated physical units (``brainunit``)
   - Modular projection architecture

**API Changes**
   - New neuron and synapse interfaces
   - Projection system redesign
   - Updated simulation APIs
   - Training framework changes

**Performance**
   - Improved JIT compilation
   - Better memory efficiency
   - Enhanced GPU/TPU support

Compatibility Layer
-------------------

BrainPy 3.0 includes ``brainpy.version2`` for backward compatibility:

.. code-block:: python

    # Old code (BrainPy 2.x) - still works with deprecation warning
    import brainpy as bp
    # bp.math, bp.layers, etc. redirect to bp.version2

    # Explicit version2 usage (recommended during migration)
    import brainpy.version2 as bp2

    # New BrainPy 3.0 API
    import brainpy  # Use new 3.0 features

Migration Strategy
------------------

Recommended Approach
~~~~~~~~~~~~~~~~~~~~

1. **Gradual Migration**: Use ``brainpy.version2`` for old code while writing new code with 3.0 API
2. **Test Thoroughly**: Ensure numerical equivalence between versions
3. **Update Incrementally**: Migrate module by module, not all at once
4. **Use Both**: Mix version2 and 3.0 code during transition

.. code-block:: python

    # During migration
    import brainpy              # New 3.0 API
    import brainpy.version2 as bp2    # Old 2.x API

    # Old model
    old_network = bp2.dyn.Network(...)

    # New model
    new_network = brainpy.LIF(...)

    # Can coexist in same codebase

Key API Changes
---------------

Imports and Modules
~~~~~~~~~~~~~~~~~~~

**BrainPy 2.x:**

.. code-block:: python

    import brainpy as bp
    import brainpy.math as bm
    import brainpy.layers as layers
    import brainpy.dyn as dyn
    from brainpy import neurons, synapses

**BrainPy 3.0:**

.. code-block:: python

    import brainpy as bp           # Core neurons, synapses, projections
    import brainstate              # State management, modules
    import brainunit as u          # Physical units
    import braintools              # Utilities, optimizers, etc.

Neuron Models
~~~~~~~~~~~~~

**BrainPy 2.x:**

.. code-block:: python

    # Old API
    neurons = bp.neurons.LIF(
        size=100,
        V_rest=-65.,
        V_th=-50.,
        V_reset=-60.,
        tau=10.,
        V_initializer=bp.init.Normal(-60., 5.)
    )

**BrainPy 3.0:**

.. code-block:: python

    # New API - with units!
    import brainunit as u
    import braintools

    neurons = brainpy.LIF(
        size=100,
        V_rest=-65. * u.mV,        # Units required
        V_th=-50. * u.mV,
        V_reset=-60. * u.mV,
        tau=10. * u.ms,
        V_initializer=braintools.init.Normal(-60., 5., unit=u.mV)
    )

**Key Changes:**

- Simpler import: ``brainpy.LIF`` instead of ``bp.neurons.LIF``
- Physical units are mandatory
- Initializers from ``braintools.init``
- Must use ``brainstate.nn.init_all_states()`` before simulation

Synapse Models
~~~~~~~~~~~~~~

**BrainPy 2.x:**

.. code-block:: python

    # Old API
    syn = bp.synapses.Exponential(
        pre=pre_neurons,
        post=post_neurons,
        conn=bp.connect.FixedProb(0.1),
        tau=5.,
        output=bp.synouts.CUBA()
    )

**BrainPy 3.0:**

.. code-block:: python

    # New API - using projection architecture
    import brainstate

    projection = brainpy.AlignPostProj(
        comm=brainstate.nn.EventFixedProb(
            pre_size, post_size, prob=0.1, weight=0.5*u.mS
        ),
        syn=brainpy.Expon.desc(post_size, tau=5.*u.ms),
        out=brainpy.CUBA.desc(),
        post=post_neurons
    )

**Key Changes:**

- Synapse, connectivity, and output are separated
- Use descriptor pattern (``.desc()``)
- Projections handle the complete pathway
- Physical units throughout

Network Definition
~~~~~~~~~~~~~~~~~~

**BrainPy 2.x:**

.. code-block:: python

    # Old API
    class EINet(bp.DynamicalSystem):
        def __init__(self):
            super().__init__()
            self.E = bp.neurons.LIF(800)
            self.I = bp.neurons.LIF(200)
            self.E2E = bp.synapses.Exponential(...)
            self.E2I = bp.synapses.Exponential(...)
            # ...

        def update(self, tdi, x):
            self.E(x)
            self.I(x)
            self.E2E()
            # ...

**BrainPy 3.0:**

.. code-block:: python

    # New API
    import brainstate

    class EINet(brainstate.nn.Module):
        def __init__(self):
            super().__init__()
            self.E = brainpy.LIF(800, ...)
            self.I = brainpy.LIF(200, ...)
            self.E2E = brainpy.AlignPostProj(...)
            self.E2I = brainpy.AlignPostProj(...)
            # ...

        def update(self, x):
            spikes_e = self.E.get_spike()
            spikes_i = self.I.get_spike()

            self.E2E(spikes_e)
            self.E2I(spikes_e)
            # ...

            self.E(x)
            self.I(x)

**Key Changes:**

- Inherit from ``brainstate.nn.Module`` instead of ``bp.DynamicalSystem``
- No ``tdi`` argument (time info from ``brainstate.environ``)
- Explicit spike handling with ``get_spike()``
- Update order: projections first, then neurons

Running Simulations
~~~~~~~~~~~~~~~~~~~

**BrainPy 2.x:**

.. code-block:: python

    # Old API
    runner = bp.DSRunner(network, monitors=['E.spike'])
    runner.run(duration=1000.)

    # Access results
    spikes = runner.mon['E.spike']

**BrainPy 3.0:**

.. code-block:: python

    # New API
    import brainunit as u

    # Set time step
    brainstate.environ.set(dt=0.1 * u.ms)

    # Initialize
    brainstate.nn.init_all_states(network)

    # Run simulation
    times = u.math.arange(0*u.ms, 1000*u.ms, brainstate.environ.get_dt())
    results = brainstate.transform.for_loop(
        network.update,
        times,
        pbar=brainstate.transform.ProgressBar(10)
    )

**Key Changes:**

- No ``DSRunner`` class
- Use ``brainstate.transform.for_loop`` for simulation
- Must initialize states explicitly
- Manual recording of variables
- Physical units for time

Training
~~~~~~~~

**BrainPy 2.x:**

.. code-block:: python

    # Old API
    trainer = bp.BPTT(
        network,
        loss_fun=loss_fn,
        optimizer=bp.optim.Adam(lr=1e-3)
    )
    trainer.fit(train_data, epochs=100)

**BrainPy 3.0:**

.. code-block:: python

    # New API
    import braintools

    # Define optimizer
    optimizer = braintools.optim.Adam(lr=1e-3)
    optimizer.register_trainable_weights(
        network.states(brainstate.ParamState)
    )

    # Training loop
    @brainstate.compile.jit
    def train_step(inputs, targets):
        def loss_fn():
            predictions = brainstate.compile.for_loop(network.update, inputs)
            return compute_loss(predictions, targets)

        grads, loss = brainstate.transform.grad(
            loss_fn,
            network.states(brainstate.ParamState),
            return_value=True
        )()
        optimizer.update(grads)
        return loss

    # Train
    for epoch in range(100):
        loss = train_step(train_inputs, train_targets)

**Key Changes:**

- No ``BPTT`` or ``Trainer`` classes
- Manual training loop implementation
- Explicit gradient computation
- More control, more flexibility

Common Migration Patterns
--------------------------

Pattern 1: Simple Neuron Population
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**2.x Code:**

.. code-block:: python

    neurons = bp.neurons.LIF(100, V_rest=-65., V_th=-50., tau=10.)
    runner = bp.DSRunner(neurons)
    runner.run(100., inputs=2.0)

**3.0 Code:**

.. code-block:: python

    import brainunit as u
    import brainstate

    brainstate.environ.set(dt=0.1*u.ms)
    neurons = brainpy.LIF(100, V_rest=-65.*u.mV, V_th=-50.*u.mV, tau=10.*u.ms)
    brainstate.nn.init_all_states(neurons)

    times = u.math.arange(0*u.ms, 100*u.ms, brainstate.environ.get_dt())
    results = brainstate.transform.for_loop(
        lambda t: neurons(2.0*u.nA),
        times
    )

Pattern 2: E-I Network
~~~~~~~~~~~~~~~~~~~~~~

**2.x Code:**

.. code-block:: python

    E = bp.neurons.LIF(800)
    I = bp.neurons.LIF(200)
    E2E = bp.synapses.Exponential(E, E, bp.connect.FixedProb(0.02))
    E2I = bp.synapses.Exponential(E, I, bp.connect.FixedProb(0.02))
    I2E = bp.synapses.Exponential(I, E, bp.connect.FixedProb(0.02))
    I2I = bp.synapses.Exponential(I, I, bp.connect.FixedProb(0.02))

    net = bp.Network(E, I, E2E, E2I, I2E, I2I)
    runner = bp.DSRunner(net)
    runner.run(1000.)

**3.0 Code:**

.. code-block:: python

    import brainpy as bp
    import brainstate
    import brainunit as u

    class EINet(brainstate.nn.Module):
        def __init__(self):
            super().__init__()
            self.E = brainpy.LIF(800, V_th=-50.*u.mV, tau=10.*u.ms)
            self.I = brainpy.LIF(200, V_th=-50.*u.mV, tau=10.*u.ms)

            self.E2E = brainpy.AlignPostProj(
                comm=brainstate.nn.EventFixedProb(800, 800, 0.02, 0.1*u.mS),
                syn=brainpy.Expon.desc(800, tau=5.*u.ms),
                out=brainpy.CUBA.desc(),
                post=self.E
            )
            # ... similar for E2I, I2E, I2I

        def update(self, inp):
            e_spk = self.E.get_spike()
            i_spk = self.I.get_spike()
            self.E2E(e_spk)
            # ... other projections
            self.E(inp)
            self.I(inp)

    brainstate.environ.set(dt=0.1*u.ms)
    net = EINet()
    brainstate.nn.init_all_states(net)

    times = u.math.arange(0*u.ms, 1000*u.ms, 0.1*u.ms)
    results = brainstate.transform.for_loop(
        lambda t: net.update(1.*u.nA),
        times
    )

Troubleshooting
---------------

Common Issues
~~~~~~~~~~~~~

**Issue 1: ImportError**

.. code-block:: python

    # Error: ModuleNotFoundError: No module named 'brainpy.math'
    import brainpy.math as bm  # Old import

    # Solution: Use version2 or update to new API
    import brainpy.version2.math as bm  # Temporary
    # or
    import brainunit as u  # New API

**Issue 2: Unit Errors**

.. code-block:: python

    # Error: Units required but not provided
    neuron = bp.LIF(100, tau=10.)  # Missing units

    # Solution: Add units
    import brainunit as u
    neuron = bp.LIF(100, tau=10.*u.ms)

**Issue 3: State Initialization**

.. code-block:: python

    # Error: States not initialized
    neuron = brainpy.LIF(100, ...)
    neuron(input)  # May fail or give wrong results

    # Solution: Initialize states
    import brainstate
    neuron = brainpy.LIF(100, ...)
    brainstate.nn.init_all_states(neuron)
    neuron(input)  # Now works correctly

**Issue 4: Projection Update Order**

.. code-block:: python

    # Wrong: Neurons before projections
    def update(self, inp):
        self.neurons(inp)
        self.projection(self.neurons.get_spike())  # Uses current spikes

    # Correct: Projections before neurons
    def update(self, inp):
        spikes = self.neurons.get_spike()  # Get previous spikes
        self.projection(spikes)             # Update synapses
        self.neurons(inp)                   # Update neurons

Testing Migration
-----------------

Numerical Equivalence
~~~~~~~~~~~~~~~~~~~~~

When migrating, verify that new code produces equivalent results:

.. code-block:: python

    # Old code results
    import brainpy.version2 as bp2
    old_network = bp2.neurons.LIF(100, ...)
    old_runner = bp2.DSRunner(old_network)
    old_runner.run(100.)
    old_voltages = old_runner.mon['V']

    # New code results
    import brainpy as bp
    import brainstate
    new_network = brainpy.LIF(100, ...)
    brainstate.nn.init_all_states(new_network)
    # ... run simulation ...
    # new_voltages = ...

    # Compare
    import numpy as np
    np.allclose(old_voltages, new_voltages, rtol=1e-5)

Feature Parity Checklist
-------------------------

Before completing migration, verify:

☐ All neuron models migrated
☐ All synapse models migrated
☐ Network structure preserved
☐ Simulation produces equivalent results
☐ Training works (if applicable)
☐ Visualization updated
☐ Unit tests pass
☐ Documentation updated

Getting Help
------------

If you encounter issues during migration:

- Check the `API documentation <../api/index.html>`_
- Review `examples <../examples/gallery.html>`_
- Search `GitHub issues <https://github.com/brainpy/BrainPy/issues>`_
- Ask on GitHub Discussions
- Read the `brainstate documentation <https://brainstate.readthedocs.io/>`_

Benefits of Migration
---------------------

Migrating to BrainPy 3.0 provides:

✅ **Better Performance**: Optimized compilation and execution

✅ **Physical Units**: Automatic unit checking prevents errors

✅ **Cleaner API**: More intuitive and consistent interfaces

✅ **Modularity**: Easier to compose and reuse components

✅ **Modern Architecture**: Built on proven frameworks

✅ **Better Tooling**: Improved ecosystem integration

Summary
-------

Migration from BrainPy 2.x to 3.0 requires:

1. Understanding new architecture (state-based, modular)
2. Adding physical units to all parameters
3. Updating import statements
4. Refactoring network definitions
5. Changing simulation and training code
6. Testing for numerical equivalence

The ``brainpy.version2`` compatibility layer enables gradual migration, allowing you to update your codebase incrementally.

Next Steps
----------

- Start with the :doc:`../quickstart/5min-tutorial` to learn 3.0 basics
- Review :doc:`../core-concepts/architecture` for design understanding
- Follow :doc:`../tutorials/basic/01-lif-neuron` for hands-on practice
- Study :doc:`../examples/gallery` for complete migration examples
