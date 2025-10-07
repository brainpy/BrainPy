Neuron Models
=============

Spiking neuron models in BrainPy.

.. currentmodule:: brainpy

Base Class
----------

Neuron
~~~~~~

.. class:: Neuron(size, **kwargs)

   Base class for all neuron models.

   All neuron models inherit from this class and implement the ``update()`` method
   for their specific dynamics.

   **Parameters:**

   - ``size`` (int) - Number of neurons in the population
   - ``**kwargs`` - Additional keyword arguments

   **Key Methods:**

   .. method:: update(x)

      Update neuron dynamics for one time step.

      :param x: Input current with units (e.g., ``2.0 * u.nA``)
      :type x: Array with brainunit
      :returns: Updated state (typically membrane potential)

   .. method:: get_spike()

      Get current spike state.

      :returns: Binary spike indicator (1 = spike, 0 = no spike)
      :rtype: Array of shape ``(size,)`` or ``(batch_size, size)``

   .. method:: reset_state(batch_size=None)

      Reset neuron state for new trial.

      :param batch_size: Optional batch dimension
      :type batch_size: int or None

   **Common Attributes:**

   - ``V`` (ShortTermState) - Membrane potential
   - ``spike`` (ShortTermState) - Spike indicator
   - ``size`` (int) - Number of neurons

   **Example:**

   .. code-block:: python

      # Subclass to create custom neuron
      class CustomNeuron(bp.Neuron):
          def __init__(self, size, tau=10*u.ms):
              super().__init__(size)
              self.tau = tau
              self.V = brainstate.ShortTermState(jnp.zeros(size))

          def update(self, x):
              # Custom dynamics
              pass

Integrate-and-Fire Models
--------------------------

IF
~~

.. class:: IF(size, V_rest=-65*u.mV, V_th=-50*u.mV, V_reset=-65*u.mV, **kwargs)

   Basic Integrate-and-Fire neuron.

   **Model:**

   .. math::

      \\frac{dV}{dt} = I_{ext}

   Spikes when :math:`V \\geq V_{th}`, then resets to :math:`V_{reset}`.

   **Parameters:**

   - ``size`` (int) - Number of neurons
   - ``V_rest`` (Quantity[mV]) - Resting potential (default: -65 mV)
   - ``V_th`` (Quantity[mV]) - Spike threshold (default: -50 mV)
   - ``V_reset`` (Quantity[mV]) - Reset potential (default: -65 mV)

   **States:**

   - ``V`` (ShortTermState) - Membrane potential [mV]
   - ``spike`` (ShortTermState) - Spike indicator [0 or 1]

   **Example:**

   .. code-block:: python

      import brainpy as bp
      import brainstate
      import brainunit as u

      neuron = bp.IF(100, V_rest=-65*u.mV, V_th=-50*u.mV)
      brainstate.nn.init_all_states(neuron)

      # Simulate
      for t in range(1000):
          inp = brainstate.random.rand(100) * 2.0 * u.nA
          neuron(inp)
          spikes = neuron.get_spike()

LIF
~~~

.. class:: LIF(size, V_rest=-65*u.mV, V_th=-50*u.mV, V_reset=-65*u.mV, tau=10*u.ms, R=1*u.ohm, **kwargs)

   Leaky Integrate-and-Fire neuron.

   **Model:**

   .. math::

      \\tau \\frac{dV}{dt} = -(V - V_{rest}) + R I_{ext}

   Most commonly used spiking neuron model.

   **Parameters:**

   - ``size`` (int) - Number of neurons
   - ``V_rest`` (Quantity[mV]) - Resting potential (default: -65 mV)
   - ``V_th`` (Quantity[mV]) - Spike threshold (default: -50 mV)
   - ``V_reset`` (Quantity[mV]) - Reset potential (default: -65 mV)
   - ``tau`` (Quantity[ms]) - Membrane time constant (default: 10 ms)
   - ``R`` (Quantity[ohm]) - Input resistance (default: 1 Ω)

   **States:**

   - ``V`` (ShortTermState) - Membrane potential [mV]
   - ``spike`` (ShortTermState) - Spike indicator [0 or 1]

   **Example:**

   .. code-block:: python

      # Standard LIF
      neuron = bp.LIF(
          size=100,
          V_rest=-65*u.mV,
          V_th=-50*u.mV,
          V_reset=-65*u.mV,
          tau=10*u.ms
      )

      brainstate.nn.init_all_states(neuron)

      # Compute F-I curve
      currents = u.math.linspace(0, 5, 20) * u.nA
      rates = []

      for I in currents:
          brainstate.nn.init_all_states(neuron)
          spike_count = 0
          for _ in range(1000):
              neuron(jnp.ones(100) * I)
              spike_count += jnp.sum(neuron.get_spike())
          rate = spike_count / (1000 * 0.1 * 1e-3) / 100  # Hz
          rates.append(rate)

   **See Also:**

   - :doc:`../core-concepts/neurons` - Detailed LIF guide
   - :doc:`../tutorials/basic/01-lif-neuron` - LIF tutorial

LIFRef
~~~~~~

.. class:: LIFRef(size, V_rest=-65*u.mV, V_th=-50*u.mV, V_reset=-65*u.mV, tau=10*u.ms, tau_ref=2*u.ms, **kwargs)

   LIF with refractory period.

   **Model:**

   .. math::

      \\tau \\frac{dV}{dt} = -(V - V_{rest}) + R I_{ext} \\quad \\text{(if not refractory)}

   After spike, neuron is unresponsive for ``tau_ref`` milliseconds.

   **Parameters:**

   - ``size`` (int) - Number of neurons
   - ``V_rest`` (Quantity[mV]) - Resting potential (default: -65 mV)
   - ``V_th`` (Quantity[mV]) - Spike threshold (default: -50 mV)
   - ``V_reset`` (Quantity[mV]) - Reset potential (default: -65 mV)
   - ``tau`` (Quantity[ms]) - Membrane time constant (default: 10 ms)
   - ``tau_ref`` (Quantity[ms]) - Refractory period (default: 2 ms)

   **States:**

   - ``V`` (ShortTermState) - Membrane potential [mV]
   - ``spike`` (ShortTermState) - Spike indicator [0 or 1]
   - ``t_last_spike`` (ShortTermState) - Time since last spike [ms]

   **Example:**

   .. code-block:: python

      neuron = bp.LIFRef(
          size=100,
          tau=10*u.ms,
          tau_ref=2*u.ms  # 2ms refractory period
      )

Adaptive Models
---------------

ALIF
~~~~

.. class:: ALIF(size, V_rest=-65*u.mV, V_th=-50*u.mV, V_reset=-65*u.mV, tau=10*u.ms, tau_w=100*u.ms, a=0*u.nA, b=0.5*u.nA, **kwargs)

   Adaptive Leaky Integrate-and-Fire neuron.

   **Model:**

   .. math::

      \\tau \\frac{dV}{dt} &= -(V - V_{rest}) + R I_{ext} - R w \\\\
      \\tau_w \\frac{dw}{dt} &= a(V - V_{rest}) - w

   After spike: :math:`w \\rightarrow w + b`

   Implements spike-frequency adaptation through adaptation current :math:`w`.

   **Parameters:**

   - ``size`` (int) - Number of neurons
   - ``V_rest`` (Quantity[mV]) - Resting potential (default: -65 mV)
   - ``V_th`` (Quantity[mV]) - Spike threshold (default: -50 mV)
   - ``V_reset`` (Quantity[mV]) - Reset potential (default: -65 mV)
   - ``tau`` (Quantity[ms]) - Membrane time constant (default: 10 ms)
   - ``tau_w`` (Quantity[ms]) - Adaptation time constant (default: 100 ms)
   - ``a`` (Quantity[nA]) - Subthreshold adaptation (default: 0 nA)
   - ``b`` (Quantity[nA]) - Spike-triggered adaptation (default: 0.5 nA)

   **States:**

   - ``V`` (ShortTermState) - Membrane potential [mV]
   - ``w`` (ShortTermState) - Adaptation current [nA]
   - ``spike`` (ShortTermState) - Spike indicator [0 or 1]

   **Example:**

   .. code-block:: python

      # Adapting neuron
      neuron = bp.ALIF(
          size=100,
          tau=10*u.ms,
          tau_w=100*u.ms,  # Slow adaptation
          a=0.1*u.nA,      # Subthreshold coupling
          b=0.5*u.nA       # Spike-triggered jump
      )

      # Constant input → decreasing firing rate
      brainstate.nn.init_all_states(neuron)
      rates = []

      for t in range(2000):
          neuron(jnp.ones(100) * 5.0 * u.nA)
          if t % 100 == 0:
              rate = jnp.mean(neuron.get_spike())
              rates.append(rate)
              # rates will decrease over time due to adaptation

Izhikevich
~~~~~~~~~~

.. class:: Izhikevich(size, a=0.02, b=0.2, c=-65*u.mV, d=8*u.mV, **kwargs)

   Izhikevich neuron model.

   **Model:**

   .. math::

      \\frac{dV}{dt} &= 0.04 V^2 + 5V + 140 - u + I \\\\
      \\frac{du}{dt} &= a(bV - u)

   If :math:`V \\geq 30`, then :math:`V \\rightarrow c, u \\rightarrow u + d`

   Can reproduce many different firing patterns by varying parameters.

   **Parameters:**

   - ``size`` (int) - Number of neurons
   - ``a`` (float) - Time scale of recovery variable (default: 0.02)
   - ``b`` (float) - Sensitivity of recovery to V (default: 0.2)
   - ``c`` (Quantity[mV]) - After-spike reset value of V (default: -65 mV)
   - ``d`` (Quantity[mV]) - After-spike increment of u (default: 8 mV)

   **States:**

   - ``V`` (ShortTermState) - Membrane potential [mV]
   - ``u`` (ShortTermState) - Recovery variable [mV]
   - ``spike`` (ShortTermState) - Spike indicator [0 or 1]

   **Common Parameter Sets:**

   .. code-block:: python

      # Regular spiking
      neuron_rs = bp.Izhikevich(100, a=0.02, b=0.2, c=-65*u.mV, d=8*u.mV)

      # Intrinsically bursting
      neuron_ib = bp.Izhikevich(100, a=0.02, b=0.2, c=-55*u.mV, d=4*u.mV)

      # Chattering
      neuron_ch = bp.Izhikevich(100, a=0.02, b=0.2, c=-50*u.mV, d=2*u.mV)

      # Fast spiking
      neuron_fs = bp.Izhikevich(100, a=0.1, b=0.2, c=-65*u.mV, d=2*u.mV)

   **Example:**

   .. code-block:: python

      neuron = bp.Izhikevich(100, a=0.02, b=0.2, c=-65*u.mV, d=8*u.mV)
      brainstate.nn.init_all_states(neuron)

      for t in range(1000):
          inp = brainstate.random.rand(100) * 15.0 * u.nA
          neuron(inp)

Exponential Models
------------------

ExpIF
~~~~~

.. class:: ExpIF(size, V_rest=-65*u.mV, V_th=-50*u.mV, V_reset=-65*u.mV, tau=10*u.ms, delta_T=2*u.mV, **kwargs)

   Exponential Integrate-and-Fire neuron.

   **Model:**

   .. math::

      \\tau \\frac{dV}{dt} = -(V - V_{rest}) + \\Delta_T e^{\\frac{V - V_{th}}{\\Delta_T}} + R I_{ext}

   Features exponential spike generation.

   **Parameters:**

   - ``size`` (int) - Number of neurons
   - ``V_rest`` (Quantity[mV]) - Resting potential (default: -65 mV)
   - ``V_th`` (Quantity[mV]) - Spike threshold (default: -50 mV)
   - ``V_reset`` (Quantity[mV]) - Reset potential (default: -65 mV)
   - ``tau`` (Quantity[ms]) - Membrane time constant (default: 10 ms)
   - ``delta_T`` (Quantity[mV]) - Spike slope factor (default: 2 mV)

AdExIF
~~~~~~

.. class:: AdExIF(size, V_rest=-65*u.mV, V_th=-50*u.mV, V_reset=-65*u.mV, tau=10*u.ms, tau_w=100*u.ms, delta_T=2*u.mV, a=0*u.nA, b=0.5*u.nA, **kwargs)

   Adaptive Exponential Integrate-and-Fire neuron.

   Combines exponential spike generation with adaptation.

   **Parameters:**

   Similar to ExpIF plus ALIF adaptation parameters (``tau_w``, ``a``, ``b``).

Usage Patterns
--------------

**Creating Neuron Populations:**

.. code-block:: python

   # Single population
   neurons = bp.LIF(1000, V_rest=-65*u.mV, V_th=-50*u.mV, tau=10*u.ms)

   # Multiple populations with different parameters
   E_neurons = bp.LIF(800, tau=15*u.ms)  # Excitatory: slower
   I_neurons = bp.LIF(200, tau=10*u.ms)  # Inhibitory: faster

**Batched Simulation:**

.. code-block:: python

   neuron = bp.LIF(100, ...)
   brainstate.nn.init_all_states(neuron, batch_size=32)

   # Input shape: (32, 100)
   inp = brainstate.random.rand(32, 100) * 2.0 * u.nA
   neuron(inp)

   # Output shape: (32, 100)
   spikes = neuron.get_spike()

**Custom Neurons:**

.. code-block:: python

   class CustomLIF(bp.Neuron):
       def __init__(self, size, tau=10*u.ms):
           super().__init__(size)
           self.tau = tau
           self.V = brainstate.ShortTermState(jnp.zeros(size))
           self.spike = brainstate.ShortTermState(jnp.zeros(size))

       def reset_state(self, batch_size=None):
           shape = self.size if batch_size is None else (batch_size, self.size)
           self.V.value = jnp.zeros(shape)
           self.spike.value = jnp.zeros(shape)

       def update(self, I):
           # Custom dynamics
           pass

       def get_spike(self):
           return self.spike.value

See Also
--------

- :doc:`../core-concepts/neurons` - Detailed neuron model guide
- :doc:`../tutorials/basic/01-lif-neuron` - LIF neuron tutorial
- :doc:`../how-to-guides/custom-components` - Creating custom neurons
- :doc:`synapses` - Synaptic models
- :doc:`projections` - Connecting neurons
