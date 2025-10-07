Training Utilities
==================

Tools for training spiking neural networks.

.. currentmodule:: braintools

Optimizers
----------

From ``braintools.optim``:

Adam
~~~~

.. class:: optim.Adam(learning_rate=0.001, beta1=0.9, beta2=0.999, eps=1e-8)

   Adam optimizer.

   **Parameters:**

   - ``learning_rate`` (float) - Learning rate
   - ``beta1`` (float) - First moment decay
   - ``beta2`` (float) - Second moment decay
   - ``eps`` (float) - Numerical stability

   **Methods:**

   .. method:: register_trainable_weights(params)

      Register parameters to optimize.

   .. method:: update(grads)

      Update parameters with gradients.

   **Example:**

   .. code-block:: python

      optimizer = braintools.optim.Adam(learning_rate=1e-3)
      params = net.states(brainstate.ParamState)
      optimizer.register_trainable_weights(params)

      # Training loop
      grads = brainstate.transform.grad(loss_fn, params)(...)
      optimizer.update(grads)

SGD
~~~

.. class:: optim.SGD(learning_rate=0.01, momentum=0.0)

   Stochastic gradient descent with momentum.

RMSprop
~~~~~~~

.. class:: optim.RMSprop(learning_rate=0.001, decay=0.9, eps=1e-8)

   RMSprop optimizer.

Gradient Computation
--------------------

From ``brainstate.transform``:

grad
~~~~

.. function:: transform.grad(fun, argnums=0, has_aux=False, return_value=False)

   Compute gradients of a function.

   **Parameters:**

   - ``fun`` - Function to differentiate
   - ``argnums`` - Which arguments to differentiate
   - ``has_aux`` - Whether function returns auxiliary data
   - ``return_value`` - Also return function value

   **Example:**

   .. code-block:: python

      def loss_fn(params, net, X, y):
          output = net(X)
          return jnp.mean((output - y)**2)

      params = net.states(brainstate.ParamState)

      # Get gradients
      grads = brainstate.transform.grad(loss_fn, params)(net, X, y)

      # Get gradients and loss
      grads, loss = brainstate.transform.grad(
          loss_fn, params, return_value=True
      )(net, X, y)

value_and_grad
~~~~~~~~~~~~~~

.. function:: transform.value_and_grad(fun, argnums=0)

   Compute both value and gradient (more efficient than separate calls).

Surrogate Gradients
-------------------

From ``braintools.surrogate``:

ReluGrad
~~~~~~~~

.. class:: surrogate.ReluGrad(alpha=1.0)

   ReLU surrogate gradient for spike function.

   **Example:**

   .. code-block:: python

      neuron = bp.LIF(
          100,
          spike_fun=braintools.surrogate.ReluGrad()
      )

sigmoid
~~~~~~~

.. function:: surrogate.sigmoid(alpha=4.0)

   Sigmoid surrogate gradient.

slayer_grad
~~~~~~~~~~~

.. function:: surrogate.slayer_grad(alpha=4.0)

   SLAYER/SuperSpike surrogate gradient.

Loss Functions
--------------

From ``braintools.metric``:

softmax_cross_entropy_with_integer_labels
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. function:: metric.softmax_cross_entropy_with_integer_labels(logits, labels)

   Cross-entropy loss for classification.

   **Parameters:**

   - ``logits`` - Network outputs (batch_size, num_classes)
   - ``labels`` - Integer labels (batch_size,)

   **Returns:**

   Loss per example (batch_size,)

   **Example:**

   .. code-block:: python

      logits = net(inputs)  # (32, 10)
      labels = jnp.array([0, 1, 2, ...])  # (32,)

      loss = braintools.metric.softmax_cross_entropy_with_integer_labels(
          logits, labels
      ).mean()

Training Workflow
-----------------

**Complete example:**

.. code-block:: python

   import brainpy as bp
   import brainstate
   import braintools

   # 1. Create network
   net = TrainableSNN()
   brainstate.nn.init_all_states(net, batch_size=32)

   # 2. Create optimizer
   optimizer = braintools.optim.Adam(learning_rate=1e-3)
   params = net.states(brainstate.ParamState)
   optimizer.register_trainable_weights(params)

   # 3. Define loss function
   def loss_fn(params, net, X, y):
       brainstate.nn.init_all_states(net)
       logits = run_network(net, X)  # Simulate and accumulate
       loss = braintools.metric.softmax_cross_entropy_with_integer_labels(
           logits, y
       ).mean()
       return loss

   # 4. Training loop
   for epoch in range(num_epochs):
       for X_batch, y_batch in data_loader:
           # Compute gradients
           grads, loss = brainstate.transform.grad(
               loss_fn, params, return_value=True
           )(net, X_batch, y_batch)

           # Update parameters
           optimizer.update(grads)

           print(f"Loss: {loss:.4f}")

See Also
--------

- :doc:`../tutorials/advanced/05-snn-training` - SNN training tutorial
- :doc:`../how-to-guides/save-load-models` - Model checkpointing
- :doc:`../how-to-guides/gpu-tpu-usage` - GPU acceleration
