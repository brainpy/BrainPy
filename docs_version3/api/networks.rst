Network Components
==================

Building blocks for neural networks.

.. currentmodule:: brainstate.nn

Module System
-------------

Module
~~~~~~

.. class:: Module

   Base class for all network components.

   **Key Methods:**

   .. method:: update(*args, **kwargs)

      Forward pass / simulation step.

   .. method:: reset_state(batch_size=None)

      Reset component state.

   .. method:: states(state_type=None)

      Get states of specific type.

      :param state_type: ParamState, ShortTermState, or LongTermState
      :returns: Dictionary of states

   **Example:**

   .. code-block:: python

      class MyNetwork(brainstate.nn.Module):
          def __init__(self):
              super().__init__()
              self.neurons = bp.LIF(100, ...)
              self.weights = brainstate.ParamState(jnp.ones((100, 100)))

          def update(self, x):
              self.neurons(x)
              return self.neurons.get_spike()

State Initialization
--------------------

init_all_states
~~~~~~~~~~~~~~~

.. function:: init_all_states(module, batch_size=None)

   Initialize all states in a module hierarchy.

   **Parameters:**

   - ``module`` - Network module
   - ``batch_size`` (int or None) - Optional batch dimension

   **Example:**

   .. code-block:: python

      net = MyNetwork()
      brainstate.nn.init_all_states(net)  # Single trial
      brainstate.nn.init_all_states(net, batch_size=32)  # Batched

Readout Layers
--------------

Readout
~~~~~~~

.. class:: Readout(in_size, out_size)

   Convert spikes to continuous outputs.

   **Example:**

   .. code-block:: python

      readout = bp.Readout(n_hidden, n_output)

      # Usage
      spikes = hidden_neurons.get_spike()
      logits = readout(spikes)

Linear
~~~~~~

.. class:: Linear(in_size, out_size, w_init=None, b_init=None)

   Fully connected linear layer.

   **Parameters:**

   - ``in_size`` (int) - Input dimension
   - ``out_size`` (int) - Output dimension
   - ``w_init`` - Weight initializer
   - ``b_init`` - Bias initializer (None for no bias)

   **Example:**

   .. code-block:: python

      fc = brainstate.nn.Linear(
          100, 50,
          w_init=brainstate.init.KaimingNormal()
      )

      output = fc(input_data)

See Also
--------

- :doc:`../core-concepts/architecture` - Architecture overview
- :doc:`../core-concepts/state-management` - State system
- :doc:`../tutorials/basic/03-network-connections` - Network tutorial
