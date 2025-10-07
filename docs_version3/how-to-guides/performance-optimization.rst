How to Optimize Performance
============================

This guide shows you how to make your BrainPy simulations run faster.

.. contents:: Table of Contents
   :local:
   :depth: 2

Quick Wins
----------

**Top 5 optimizations (80% of speedup):**

1. ✅ **Use JIT compilation** - 10-100× speedup
2. ✅ **Use sparse connectivity** - 10-100× memory reduction
3. ✅ **Batch operations** - 2-10× speedup on GPU
4. ✅ **Use GPU/TPU** - 10-100× speedup for large networks
5. ✅ **Minimize Python loops** - Use JAX operations instead

JIT Compilation
---------------

**Essential for performance!**

.. code-block:: python

   import brainstate

   # Slow (no JIT)
   def slow_step(net, inp):
       return net(inp)

   # Fast (with JIT)
   @brainstate.compile.jit
   def fast_step(net, inp):
       return net(inp)

   # Warmup (compilation)
   _ = fast_step(net, inp)

   # 10-100× faster than slow_step
   output = fast_step(net, inp)

**Rules for JIT:**
- Static shapes (no dynamic array sizes)
- Pure functions (no side effects)
- Avoid Python loops over data

Sparse Connectivity
-------------------

**Biological networks are sparse (~1-10% connectivity)**

.. code-block:: python

   # Dense: 10,000 × 10,000 = 100M connections (400MB)
   comm_dense = brainstate.nn.Linear(10000, 10000)

   # Sparse: 10,000 × 10,000 × 0.01 = 1M connections (4MB)
   comm_sparse = brainstate.nn.EventFixedProb(
       10000, 10000,
       prob=0.01,  # 1% connectivity
       weight=0.5*u.mS
   )

**Memory savings:** 100× for 1% connectivity

Batching
--------

**Process multiple trials in parallel:**

.. code-block:: python

   # Sequential: 10 trials one by one
   for trial in range(10):
       brainstate.nn.init_all_states(net)
       run_trial(net)

   # Parallel: 10 trials simultaneously
   brainstate.nn.init_all_states(net, batch_size=10)
   run_batched(net)  # 5-10× faster on GPU

**Optimal batch sizes:**
- CPU: 1-16
- GPU: 32-256
- TPU: 128-512

GPU Usage
---------

**Automatic when available:**

.. code-block:: python

   import jax
   print(jax.devices())  # Check for GPU

   # BrainPy automatically uses GPU
   net = bp.LIF(10000, ...)
   # Runs on GPU if available

**See:** :doc:`gpu-tpu-usage` for details

Avoid Python Loops
------------------

**Replace Python loops with JAX operations:**

.. code-block:: python

   # SLOW: Python loop
   result = []
   for i in range(1000):
       result.append(net(inp))

   # FAST: JAX loop
   def body_fun(i):
       return net(inp)

   results = brainstate.transform.for_loop(body_fun, jnp.arange(1000))

Use Appropriate Precision
--------------------------

**Float32 is usually sufficient:**

.. code-block:: python

   # Default (float32) - fast
   weights = jnp.ones((1000, 1000))  # 4 bytes/element

   # Float64 - 2× slower, 2× memory
   weights = jnp.ones((1000, 1000), dtype=jnp.float64)  # 8 bytes/element

Minimize State Storage
----------------------

**Don't accumulate history:**

.. code-block:: python

   # BAD: Stores all history in Python list
   history = []
   for t in range(10000):
       output = net(inp)
       history.append(output)  # Memory leak!

   # GOOD: Process on the fly
   for t in range(10000):
       output = net(inp)
       metrics = compute_metrics(output)  # Don't store raw data

Optimize Network Architecture
------------------------------

**1. Use simpler neuron models when possible:**

.. code-block:: python

   # Complex (slow but realistic)
   neuron = bp.HH(1000, ...)  # Hodgkin-Huxley

   # Simple (fast)
   neuron = bp.LIF(1000, ...)  # Leaky Integrate-and-Fire

**2. Use CUBA instead of COBA when possible:**

.. code-block:: python

   # Slower (conductance-based)
   out = bp.COBA.desc(E=0*u.mV)

   # Faster (current-based)
   out = bp.CUBA.desc()

**3. Reduce connectivity:**

.. code-block:: python

   # Dense
   prob = 0.1  # 10% connectivity

   # Sparse
   prob = 0.02  # 2% connectivity (5× fewer connections)

Profile Before Optimizing
--------------------------

**Identify actual bottlenecks:**

.. code-block:: python

   import time

   # Time different components
   start = time.time()
   for _ in range(100):
       net(inp)
   print(f"Network update: {time.time() - start:.2f}s")

   start = time.time()
   for _ in range(100):
       output = process_output(net.get_spike())
   print(f"Output processing: {time.time() - start:.2f}s")

**Don't optimize blindly - measure first!**

Performance Checklist
---------------------

**For maximum performance:**

.. code-block:: python

   ✅ JIT compiled (@brainstate.compile.jit)
   ✅ Sparse connectivity (EventFixedProb with prob < 0.1)
   ✅ Batched (batch_size ≥ 32 on GPU)
   ✅ GPU enabled (check jax.devices())
   ✅ Static shapes (no dynamic array sizes)
   ✅ Minimal history storage
   ✅ Appropriate neuron models (LIF vs HH)
   ✅ Float32 precision

Common Bottlenecks
------------------

**Issue 1: First run very slow**
   → JIT compilation happens on first call (warmup)

**Issue 2: CPU-GPU transfers**
   → Keep data on GPU between operations

**Issue 3: Small batch sizes**
   → Increase batch_size for better GPU utilization

**Issue 4: Python loops**
   → Replace with JAX operations (for_loop, vmap)

**Issue 5: Dense connectivity**
   → Use sparse (EventFixedProb) for large networks

Complete Optimization Example
------------------------------

.. code-block:: python

   import brainpy as bp
   import brainstate
   import brainunit as u
   import jax

   # Optimized network
   class OptimizedNetwork(brainstate.nn.Module):
       def __init__(self, n_neurons=10000):
           super().__init__()

           # Simple neuron model
           self.neurons = bp.LIF(n_neurons, V_rest=-65*u.mV, V_th=-50*u.mV, tau=10*u.ms)

           # Sparse connectivity
           self.recurrent = bp.AlignPostProj(
               comm=brainstate.nn.EventFixedProb(
                   n_neurons, n_neurons,
                   prob=0.01,  # Sparse!
                   weight=0.5*u.mS
               ),
               syn=bp.Expon.desc(n_neurons, tau=5*u.ms),
               out=bp.CUBA.desc(),  # Simple output
               post=self.neurons
           )

       def update(self, inp):
           spk = self.neurons.get_spike()
           self.recurrent(spk)
           self.neurons(inp)
           return spk

   # Initialize
   net = OptimizedNetwork()
   brainstate.nn.init_all_states(net, batch_size=64)  # Batched

   # JIT compile
   @brainstate.compile.jit
   def simulate_step(net, inp):
       return net(inp)

   # Warmup
   inp = brainstate.random.rand(64, 10000) * 2.0 * u.nA
   _ = simulate_step(net, inp)

   # Fast simulation
   import time
   start = time.time()
   for _ in range(1000):
       output = simulate_step(net, inp)
   elapsed = time.time() - start

   print(f"Optimized: {1000/elapsed:.1f} steps/s")
   print(f"Throughput: {64*1000/elapsed:.1f} trials/s")

Benchmark Results
-----------------

**Typical speedups from optimization:**

.. list-table::
   :header-rows: 1

   * - Optimization
     - Speedup
     - Cumulative
   * - Baseline (Python loops, dense)
     - 1×
     - 1×
   * - + JIT compilation
     - 10-50×
     - 10-50×
   * - + Sparse connectivity
     - 2-10×
     - 20-500×
   * - + GPU
     - 5-20×
     - 100-10,000×
   * - + Batching
     - 2-5×
     - 200-50,000×

**Real example:** 10,000 neuron network
- Baseline (CPU, no JIT): 0.5 steps/s
- Optimized (GPU, JIT, sparse, batched): 5,000 steps/s
- **Total speedup: 10,000×**

See Also
--------

- :doc:`../tutorials/advanced/07-large-scale-simulations`
- :doc:`gpu-tpu-usage`
- :doc:`debugging-networks`
