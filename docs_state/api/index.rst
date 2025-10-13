API Reference
=============

Complete API reference for BrainPy 3.0.

.. note::
   BrainPy 3.0 is built on top of `brainstate <https://brainstate.readthedocs.io/>`_,
   `brainunit <https://brainunit.readthedocs.io/>`_, and `braintools <https://braintools.readthedocs.io/>`_.

Organization
------------

The API is organized into the following categories:

.. grid:: 1 2 2 3

   .. grid-item-card:: :material-regular:`psychology;2em` Neurons
      :link: neurons.html

      Spiking neuron models (LIF, ALIF, Izhikevich, HH, etc.)

   .. grid-item-card:: :material-regular:`timeline;2em` Synapses
      :link: synapses.html

      Synaptic dynamics (Expon, Alpha, AMPA, GABA, NMDA)

   .. grid-item-card:: :material-regular:`account_tree;2em` Projections
      :link: projections.html

      Connect neural populations (AlignPostProj, DeltaProj, etc.)

   .. grid-item-card:: :material-regular:`output;2em` Synaptic Outputs
      :link: synouts.html

      Convert conductances to currents (COBA, CUBA, MgBlock)

   .. grid-item-card:: :material-regular:`psychology_alt;2em` Short-Term Plasticity
      :link: stp.html

      Short-term synaptic plasticity (STP, STD)

   .. grid-item-card:: :material-regular:`sensors;2em` Readouts
      :link: readouts.html

      Readout layers (LeakyRateReadout, LeakySpikeReadout)

   .. grid-item-card:: :material-regular:`input;2em` Input Generators
      :link: inputs.html

      Spike and current generators (PoissonSpike, SpikeTime)

Quick Reference
---------------

**Most commonly used classes:**

Neurons
~~~~~~~

.. code-block:: python

   import brainpy
   import brainunit as u

   # Leaky Integrate-and-Fire
   brainpy.state.LIF(100, V_rest=-70*u.mV, V_th=-50*u.mV, tau=20*u.ms)

   # Adaptive LIF
   brainpy.state.ALIF(100, tau=20*u.ms, tau_w=144*u.ms, a=4*u.nS, b=0.0805*u.nA)

   # Izhikevich
   brainpy.state.Izhikevich(100, a=0.02/u.ms, b=0.2/u.ms, c=-65*u.mV, d=8*u.mV/u.ms)

   # Hodgkin-Huxley
   brainpy.state.HH(100, ENa=50*u.mV, EK=-77*u.mV, EL=-54.387*u.mV)

Synapses
~~~~~~~~

.. code-block:: python

   # Exponential synapse
   brainpy.state.Expon.desc(100, tau=5*u.ms)

   # Dual exponential synapse
   brainpy.state.DualExpon.desc(100, tau_rise=1*u.ms, tau_decay=10*u.ms)

   # Alpha synapse
   brainpy.state.Alpha.desc(100, tau=8*u.ms)

   # AMPA receptor
   brainpy.state.AMPA.desc(100, alpha=0.98/(u.ms*u.mM), beta=0.18/u.ms)

   # GABAa receptor
   brainpy.state.GABAa.desc(100, alpha=0.53/(u.ms*u.mM), beta=0.18/u.ms)

   # NMDA receptor
   brainpy.state.BioNMDA.desc(100, alpha1=2.0/u.ms, beta1=0.01/u.ms)

Projections
~~~~~~~~~~~

.. code-block:: python

   # AlignPost projection with communication, synapse, and output
   brainpy.state.AlignPostProj(
       comm=brainstate.nn.EventFixedProb(n_pre, n_post, prob=0.1, weight=0.5*u.mS),
       syn=brainpy.state.Expon.desc(n_post, tau=5*u.ms),
       out=brainpy.state.COBA.desc(E=0*u.mV),
       post=post_neurons
   )

   # Delta projection for direct input
   brainpy.state.DeltaProj(
       comm=lambda x: x * 10*u.mV,
       post=post_neurons
   )

   # Current projection
   brainpy.state.CurrentProj(
       comm=lambda x: x * 0.5,
       out=brainpy.state.CUBA.desc(scale=1*u.nA),
       post=post_neurons
   )

Synaptic Outputs
~~~~~~~~~~~~~~~~

.. code-block:: python

   # Conductance-based output
   brainpy.state.COBA.desc(E=0*u.mV)  # excitatory reversal potential

   # Current-based output
   brainpy.state.CUBA.desc(scale=1*u.mV)

   # NMDA with magnesium block
   brainpy.state.MgBlock.desc(E=0*u.mV, cc_Mg=1.2*u.mM, alpha=0.062, beta=3.57)

Short-Term Plasticity
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Short-term plasticity (facilitation + depression)
   brainpy.state.STP.desc(100, U=0.15, tau_f=1500*u.ms, tau_d=200*u.ms)

   # Short-term depression only
   brainpy.state.STD.desc(100, tau=200*u.ms, U=0.07)

Input Generators
~~~~~~~~~~~~~~~~

.. code-block:: python

   # Spike times
   brainpy.state.SpikeTime(100, times=[10, 20, 30]*u.ms, indices=[0, 10, 20])

   # Poisson spike generator
   brainpy.state.PoissonSpike(100, freqs=50*u.Hz)

   # Poisson encoder (dynamic rates)
   encoder = brainpy.state.PoissonEncoder(100)
   spikes = encoder.update(rates)  # rates: array of firing rates

   # Poisson input to a state variable
   brainpy.state.PoissonInput(
       target=neuron.V,
       indices=None,
       num_input=200,
       freq=50*u.Hz,
       weight=0.1*u.mV
   )

Readout Layers
~~~~~~~~~~~~~~

.. code-block:: python

   # Leaky rate-based readout
   brainpy.state.LeakyRateReadout(in_size=100, out_size=10, tau=5*u.ms)

   # Leaky spiking readout
   brainpy.state.LeakySpikeReadout(in_size=100, tau=5*u.ms, V_th=1*u.mV)

Documentation Sections
----------------------

.. toctree::
   :maxdepth: 2

   neurons
   synapses
   projections
   synouts
   stp
   readouts
   inputs

Import Structure
----------------

BrainPy uses a clear import hierarchy:

.. code-block:: python

   import brainpy as bp              # Core BrainPy
   import brainstate                 # State management and modules
   import brainunit as u             # Physical units
   import braintools                 # Training utilities

   # Neurons and synapses
   neuron = brainpy.state.LIF(100, ...)
   synapse = brainpy.state.Expon.desc(100, tau=5*u.ms)

   # State management
   state = brainstate.ShortTermState(...)
   brainstate.nn.init_all_states(net)

   # Units
   current = 2.0 * u.nA
   voltage = -65 * u.mV
   time = 10 * u.ms

   # Training
   optimizer = braintools.optim.Adam(lr=1e-3)
   loss = braintools.metric.softmax_cross_entropy(...)

Type Conventions
----------------

**States:**

- ``ShortTermState`` - Temporary dynamics (V, g, spikes)
- ``ParamState`` - Learnable parameters (weights, biases)
- ``LongTermState`` - Persistent statistics

**Units:**

All physical quantities use ``brainunit``:

- Voltage: ``u.mV``
- Current: ``u.nA``, ``u.pA``
- Time: ``u.ms``, ``u.second``
- Conductance: ``u.mS``, ``u.nS``
- Concentration: ``u.mM``

**Shapes:**

- Single trial: ``(n_neurons,)``
- Batched: ``(batch_size, n_neurons)``
- Connectivity: ``(n_pre, n_post)``

See Also
--------

**External Documentation:**

- `BrainState Documentation <https://brainstate.readthedocs.io/>`_ - State management
- `BrainUnit Documentation <https://brainunit.readthedocs.io/>`_ - Physical units
- `BrainTools Documentation <https://braintools.readthedocs.io/>`_ - Training utilities
- `JAX Documentation <https://jax.readthedocs.io/>`_ - Underlying computation
