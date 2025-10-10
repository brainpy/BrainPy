Input and Output
================

Utilities for generating inputs and processing outputs.

Input Encoding
--------------

Poisson Spike Trains
~~~~~~~~~~~~~~~~~~~~

Generate Poisson-distributed spikes:

.. code-block:: python

   def poisson_input(rates, dt):
       """Generate Poisson spike train.

       Args:
           rates: Firing rates in Hz (array)
           dt: Time step (Quantity[ms])

       Returns:
           Binary spike array
       """
       probs = rates * dt.to_decimal(u.second)
       return (brainstate.random.rand(*rates.shape) < probs).astype(float)

   # Usage
   rates = jnp.ones(100) * 50  # 50 Hz
   spikes = poisson_input(rates, dt=0.1*u.ms)

Rate Coding
~~~~~~~~~~~

Encode values as firing rates:

.. code-block:: python

   def rate_encode(values, max_rate, dt):
       """Encode values as spike rates.

       Args:
           values: Values to encode [0, 1]
           max_rate: Maximum firing rate (Quantity[Hz])
           dt: Time step (Quantity[ms])
       """
       rates = values * max_rate.to_decimal(u.Hz)
       probs = rates * dt.to_decimal(u.second)
       return (brainstate.random.rand(len(values)) < probs).astype(float)

   # Usage
   pixel_values = jnp.array([0.2, 0.8, 0.5, ...])  # Normalized pixels
   spikes = rate_encode(pixel_values, max_rate=100*u.Hz, dt=0.1*u.ms)

Population Coding
~~~~~~~~~~~~~~~~~

Encode with population of tuned neurons:

.. code-block:: python

   def population_encode(value, n_neurons, pref_values, sigma, max_rate, dt):
       """Population coding with Gaussian tuning curves.

       Args:
           value: Value to encode (scalar)
           n_neurons: Number of neurons in population
           pref_values: Preferred values of neurons
           sigma: Tuning width
           max_rate: Maximum firing rate
           dt: Time step
       """
       # Gaussian tuning curves
       responses = jnp.exp(-0.5 * ((value - pref_values) / sigma)**2)
       rates = responses * max_rate.to_decimal(u.Hz)
       probs = rates * dt.to_decimal(u.second)
       return (brainstate.random.rand(n_neurons) < probs).astype(float)

   # Usage
   pref_values = jnp.linspace(0, 1, 20)
   spikes = population_encode(
       value=0.5,
       n_neurons=20,
       pref_values=pref_values,
       sigma=0.1,
       max_rate=100*u.Hz,
       dt=0.1*u.ms
   )

Temporal Contrast
~~~~~~~~~~~~~~~~~

Encode based on image gradients (event cameras):

.. code-block:: python

   def temporal_contrast_encode(image, prev_image, threshold=0.1, polarity=True):
       """Encode based on temporal contrast.

       Args:
           image: Current image
           prev_image: Previous image
           threshold: Change threshold
           polarity: If True, separate ON/OFF channels

       Returns:
           Spike events
       """
       diff = image - prev_image

       if polarity:
           on_spikes = (diff > threshold).astype(float)
           off_spikes = (diff < -threshold).astype(float)
           return on_spikes, off_spikes
       else:
           spikes = (jnp.abs(diff) > threshold).astype(float)
           return spikes

Output Decoding
---------------

Population Vector Decoding
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   def population_decode(spike_counts, pref_values):
       """Decode value from population spikes.

       Args:
           spike_counts: Spike counts from population
           pref_values: Preferred values of neurons

       Returns:
           Decoded value
       """
       total_activity = jnp.sum(spike_counts)
       if total_activity > 0:
           decoded = jnp.sum(spike_counts * pref_values) / total_activity
           return decoded
       return 0.0

   # Usage
   spike_counts = jnp.array([5, 12, 20, 15, 3, ...])  # From 20 neurons
   pref_values = jnp.linspace(0, 1, 20)
   decoded_value = population_decode(spike_counts, pref_values)

Spike Count
~~~~~~~~~~~

.. code-block:: python

   # Count total spikes over time window
   spike_count = jnp.sum(spike_history, axis=0)

   # Firing rate (Hz)
   duration = n_steps * dt.to_decimal(u.second)
   firing_rate = spike_count / duration

Readout Layer
~~~~~~~~~~~~~

Use ``bp.Readout`` for trainable spike-to-output conversion:

.. code-block:: python

   readout = bp.Readout(n_neurons, n_outputs)

   # Accumulate over time
   def run_and_readout(net, inputs, n_steps):
       brainstate.nn.init_all_states(net)

       outputs = []
       for t in range(n_steps):
           net(inputs)
           spikes = net.get_spike()
           output = readout(spikes)
           outputs.append(output)

       # Sum over time for classification
       logits = jnp.sum(jnp.array(outputs), axis=0)
       return logits

State Recording
---------------

Record Activity
~~~~~~~~~~~~~~~

.. code-block:: python

   # Record states during simulation
   V_history = []
   spike_history = []

   for t in range(n_steps):
       neuron(input_current)

       V_history.append(neuron.V.value.copy())
       spike_history.append(neuron.spike.value.copy())

   V_history = jnp.array(V_history)
   spike_history = jnp.array(spike_history)

Spike Raster Plot
~~~~~~~~~~~~~~~~~

.. code-block:: python

   import matplotlib.pyplot as plt

   # Get spike times and neuron indices
   times, neurons = jnp.where(spike_history > 0)

   # Plot
   plt.figure(figsize=(12, 6))
   plt.scatter(times * 0.1, neurons, s=1, c='black')
   plt.xlabel('Time (ms)')
   plt.ylabel('Neuron index')
   plt.title('Spike Raster')
   plt.show()

Firing Rate Over Time
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Population firing rate
   firing_rate = jnp.mean(spike_history, axis=1) * (1000 / dt.to_decimal(u.ms))

   plt.figure(figsize=(12, 4))
   plt.plot(times, firing_rate)
   plt.xlabel('Time (ms)')
   plt.ylabel('Population Rate (Hz)')
   plt.show()

Complete Example
----------------

.. code-block:: python

   import brainpy as bp
   import brainstate
   import brainunit as u
   import jax.numpy as jnp

   # Setup
   n_input = 784  # MNIST pixels
   n_hidden = 100
   n_output = 10
   dt = 0.1 * u.ms
   brainstate.environ.set(dt=dt)

   # Network
   class EncoderDecoderSNN(brainstate.nn.Module):
       def __init__(self):
           super().__init__()
           self.hidden = bp.LIF(n_hidden, V_rest=-65*u.mV, V_th=-50*u.mV, tau=10*u.ms)
           self.readout = bp.Readout(n_hidden, n_output)

       def update(self, x):
           self.hidden(x)
           return self.readout(self.hidden.get_spike())

   net = EncoderDecoderSNN()
   brainstate.nn.init_all_states(net)

   # Input encoding (rate coding)
   image = jnp.random.rand(784)  # Normalized image
   encoded = rate_encode(image, max_rate=100*u.Hz, dt=dt) * 2.0 * u.nA

   # Simulate
   outputs = []
   for t in range(100):
       output = net(encoded)
       outputs.append(output)

   # Output decoding
   logits = jnp.sum(jnp.array(outputs), axis=0)
   prediction = jnp.argmax(logits)

See Also
--------

- :doc:`../tutorials/basic/04-input-output` - Input/output tutorial
- :doc:`../tutorials/advanced/05-snn-training` - Training with encoded inputs
- :doc:`neurons` - Neuron models
