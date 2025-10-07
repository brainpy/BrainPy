API Reference
=============

Complete API reference for BrainPy 3.0.

.. note::
   BrainPy 3.0 is built on top of `brainstate <https://brainstate.readthedocs.io/>`_,
   `brainunit <https://brainunit.readthedocs.io/>`_, and `braintools <https://braintools.readthedocs.io/>`_.

Organization
------------

The API is organized into the following categories:

.. grid:: 1 2 2 2

   .. grid-item-card:: :material-regular:`psychology;2em` Neurons
      :link: neurons.html

      Spiking neuron models (LIF, ALIF, Izhikevich, etc.)

   .. grid-item-card:: :material-regular:`timeline;2em` Synapses
      :link: synapses.html

      Synaptic dynamics (Expon, Alpha, AMPA, GABA, NMDA)

   .. grid-item-card:: :material-regular:`account_tree;2em` Projections
      :link: projections.html

      Connect neural populations (AlignPostProj, AlignPreProj)

   .. grid-item-card:: :material-regular:`hub;2em` Networks
      :link: networks.html

      Network building blocks and utilities

   .. grid-item-card:: :material-regular:`school;2em` Training
      :link: training.html

      Gradient-based learning utilities

   .. grid-item-card:: :material-regular:`input;2em` Input/Output
      :link: input-output.html

      Input generation and output processing

Quick Reference
---------------

**Most commonly used classes:**

Neurons
~~~~~~~

.. code-block:: python

   import brainpy as bp

   # Leaky Integrate-and-Fire
   bp.LIF(size, V_rest, V_th, V_reset, tau, R, ...)

   # Adaptive LIF
   bp.ALIF(size, V_rest, V_th, V_reset, tau, tau_w, a, b, ...)

   # Izhikevich
   bp.Izhikevich(size, a, b, c, d, ...)

Synapses
~~~~~~~~

.. code-block:: python

   # Exponential
   bp.Expon.desc(size, tau)

   # Alpha
   bp.Alpha.desc(size, tau)

   # AMPA receptor
   bp.AMPA.desc(size, tau)

   # GABA_a receptor
   bp.GABAa.desc(size, tau)

Projections
~~~~~~~~~~~

.. code-block:: python

   # Standard projection
   bp.AlignPostProj(
       comm=brainstate.nn.EventFixedProb(n_pre, n_post, prob, weight),
       syn=bp.Expon.desc(n_post, tau),
       out=bp.COBA.desc(E),
       post=post_neurons
   )

Networks
~~~~~~~~

.. code-block:: python

   # Module base class
   class MyNetwork(brainstate.nn.Module):
       def __init__(self):
           super().__init__()
           # ... define components

       def update(self, x):
           # ... network dynamics
           return output

Training
~~~~~~~~

.. code-block:: python

   import braintools

   # Optimizer
   optimizer = braintools.optim.Adam(lr=1e-3)

   # Gradients
   grads = brainstate.transform.grad(loss_fn, params)(...)

   # Update
   optimizer.update(grads)

Documentation Sections
----------------------

.. toctree::
   :maxdepth: 2

   neurons
   synapses
   projections
   networks
   training
   input-output

Import Structure
----------------

BrainPy uses a clear import hierarchy:

.. code-block:: python

   import brainpy as bp              # Core BrainPy
   import brainstate                 # State management and modules
   import brainunit as u             # Physical units
   import braintools                 # Training utilities

   # Neurons and synapses
   neuron = bp.LIF(100, ...)
   synapse = bp.Expon.desc(100, tau=5*u.ms)

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
