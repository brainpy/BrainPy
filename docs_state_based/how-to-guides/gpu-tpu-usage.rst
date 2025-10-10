How to Use GPU and TPU
======================

This guide shows you how to leverage GPU and TPU acceleration for faster simulations and training with BrainPy.

.. contents:: Table of Contents
   :local:
   :depth: 2

Quick Start
-----------

**Check available devices:**

.. code-block:: python

   import jax
   print("Available devices:", jax.devices())
   print("Default backend:", jax.default_backend())

**BrainPy automatically uses available accelerators** - no code changes needed!

.. code-block:: python

   import brainpy as bp
   import brainstate

   # This automatically runs on GPU if available
   net = bp.LIF(10000, V_rest=-65*u.mV, V_th=-50*u.mV, tau=10*u.ms)
   brainstate.nn.init_all_states(net)

   for _ in range(1000):
       net(brainstate.random.rand(10000) * 2.0 * u.nA)

Installation
------------

CPU-Only (Default)
~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   pip install brainpy[cpu]

GPU (CUDA 12)
~~~~~~~~~~~~~

.. code-block:: bash

   # CUDA 12
   pip install brainpy[cuda12]

   # Or CUDA 11
   pip install brainpy[cuda11]

**Requirements:**

- NVIDIA GPU (compute capability ≥ 3.5)
- CUDA Toolkit installed
- cuDNN libraries

TPU (Google Cloud)
~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   pip install brainpy[tpu]

**Requirements:**

- Google Cloud TPU instance
- TPU runtime configured

Verify Installation
~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import jax
   import jax.numpy as jnp

   # Check JAX can see GPU/TPU
   print("Devices:", jax.devices())

   # Test computation
   x = jnp.ones((1000, 1000))
   y = jnp.dot(x, x)
   print("✅ JAX computation works!")

   # Check device placement
   print("Result device:", y.device())

Expected output (GPU):

.. code-block:: text

   Devices: [cuda(id=0)]
   ✅ JAX computation works!
   Result device: cuda:0

Understanding Device Placement
-------------------------------

Automatic Placement
~~~~~~~~~~~~~~~~~~~

**JAX automatically places computations on the best available device:**

1. TPU (if available)
2. GPU (if available)
3. CPU (fallback)

.. code-block:: python

   import brainpy as bp
   import brainstate

   # Automatically uses GPU if available
   net = bp.LIF(1000, ...)
   brainstate.nn.init_all_states(net)

   # All operations run on GPU
   net(input_data)

Manual Device Selection
~~~~~~~~~~~~~~~~~~~~~~~

Force computation on specific device:

.. code-block:: python

   import jax

   # Run on specific GPU
   with jax.default_device(jax.devices('gpu')[0]):
       net = bp.LIF(1000, ...)
       brainstate.nn.init_all_states(net)
       result = net(input_data)

   # Run on CPU
   with jax.default_device(jax.devices('cpu')[0]):
       net_cpu = bp.LIF(1000, ...)
       brainstate.nn.init_all_states(net_cpu)
       result_cpu = net_cpu(input_data)

Check Data Location
~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Check where data lives
   neuron = bp.LIF(100, ...)
   brainstate.nn.init_all_states(neuron)

   print("Voltage device:", neuron.V.value.device())
   # Output: cuda:0 (if on GPU)

Optimizing for GPU
-------------------

Use JIT Compilation
~~~~~~~~~~~~~~~~~~~

**Essential for GPU performance!**

.. code-block:: python

   import brainstate

   net = bp.LIF(10000, ...)
   brainstate.nn.init_all_states(net)

   # WITHOUT JIT (slow on GPU)
   for _ in range(1000):
       net(input_data)  # Many small kernel launches

   # WITH JIT (fast on GPU)
   @brainstate.compile.jit
   def simulate_step(net, inp):
       return net(inp)

   # Warmup (compilation)
   _ = simulate_step(net, input_data)

   # Fast execution
   for _ in range(1000):
       output = simulate_step(net, input_data)

**Speedup:** 10-100× with JIT on GPU

Batch Operations
~~~~~~~~~~~~~~~~

**Process multiple trials in parallel:**

.. code-block:: python

   # Single trial (underutilizes GPU)
   net = bp.LIF(1000, ...)
   brainstate.nn.init_all_states(net)  # Shape: (1000,)

   # Multiple trials in parallel (efficient GPU usage)
   net_batched = bp.LIF(1000, ...)
   brainstate.nn.init_all_states(net_batched, batch_size=64)  # Shape: (64, 1000)

   # GPU processes all 64 trials simultaneously
   inp = brainstate.random.rand(64, 1000) * 2.0 * u.nA
   output = net_batched(inp)

**GPU Utilization:**

- Small batches (1-10): ~10-30% GPU usage
- Medium batches (32-128): ~60-80% GPU usage
- Large batches (256+): ~90-100% GPU usage

Appropriate Problem Size
~~~~~~~~~~~~~~~~~~~~~~~~

**GPU overhead is worth it for large problems:**

.. list-table:: When to Use GPU
   :header-rows: 1

   * - Network Size
     - GPU Speedup
     - Recommendation
   * - < 1,000 neurons
     - 0.5-2×
     - Use CPU
   * - 1,000-10,000
     - 2-10×
     - GPU beneficial
   * - 10,000-100,000
     - 10-50×
     - GPU strongly recommended
   * - > 100,000
     - 50-100×
     - GPU essential

Minimize Data Transfer
~~~~~~~~~~~~~~~~~~~~~~

**Avoid moving data between CPU and GPU:**

.. code-block:: python

   # BAD: Frequent CPU-GPU transfers
   for i in range(1000):
       inp_cpu = np.random.rand(1000)  # On CPU
       inp_gpu = jnp.array(inp_cpu)    # Transfer to GPU
       output_gpu = net(inp_gpu)        # Compute on GPU
       output_cpu = np.array(output_gpu)  # Transfer to CPU
       # CPU-GPU transfer dominates time!

   # GOOD: Keep data on GPU
   @brainstate.compile.jit
   def simulate_step(net, key):
       inp = brainstate.random.uniform(key, (1000,)) * 2.0  # Generated on GPU
       return net(inp)  # Stays on GPU

   key = brainstate.random.split_key()
   for i in range(1000):
       output = simulate_step(net, key)  # All on GPU

Use Sparse Operations
~~~~~~~~~~~~~~~~~~~~~

**Sparse connectivity is crucial for large networks:**

.. code-block:: python

   # Dense (memory intensive on GPU)
   dense_proj = bp.AlignPostProj(
       comm=brainstate.nn.Linear(10000, 10000),  # 400MB just for weights!
       syn=bp.Expon.desc(10000, tau=5*u.ms),
       out=bp.CUBA.desc(),
       post=post_neurons
   )

   # Sparse (memory efficient)
   sparse_proj = bp.AlignPostProj(
       comm=brainstate.nn.EventFixedProb(
           pre_size=10000,
           post_size=10000,
           prob=0.01,  # 1% connectivity
           weight=0.5*u.mS
       ),  # Only 4MB for weights!
       syn=bp.Expon.desc(10000, tau=5*u.ms),
       out=bp.CUBA.desc(),
       post=post_neurons
   )

Multi-GPU Usage
---------------

Data Parallelism
~~~~~~~~~~~~~~~~

**Run different trials on different GPUs:**

.. code-block:: python

   import jax

   # Check available GPUs
   gpus = jax.devices('gpu')
   print(f"Found {len(gpus)} GPUs")

   # Split work across GPUs
   def run_on_gpu(gpu_id, n_trials):
       with jax.default_device(gpus[gpu_id]):
           net = bp.LIF(1000, ...)
           brainstate.nn.init_all_states(net, batch_size=n_trials)

           results = []
           for _ in range(100):
               output = net(input_data)
               results.append(output)

           return results

   # Run on multiple GPUs in parallel
   from concurrent.futures import ThreadPoolExecutor

   with ThreadPoolExecutor(max_workers=len(gpus)) as executor:
       futures = [
           executor.submit(run_on_gpu, i, 32)
           for i in range(len(gpus))
       ]
       all_results = [f.result() for f in futures]

Using JAX pmap
~~~~~~~~~~~~~~

**Parallel map across devices:**

.. code-block:: python

   from jax import pmap
   import jax.numpy as jnp

   # Create model
   net = bp.LIF(1000, ...)

   @pmap
   def parallel_simulate(inputs):
       """Run on multiple devices in parallel."""
       brainstate.nn.init_all_states(net)
       return net(inputs)

   # Split inputs across devices
   n_devices = len(jax.devices())
   inputs = jnp.ones((n_devices, 1000))  # One batch per device

   # Run in parallel
   outputs = parallel_simulate(inputs)
   # outputs.shape = (n_devices, output_size)

TPU-Specific Optimization
--------------------------

TPU Characteristics
~~~~~~~~~~~~~~~~~~~

**TPUs are optimized for:**

✅ Large matrix multiplications (e.g., dense layers)

✅ High batch sizes (128+)

✅ Float32 operations (bf16 also good)

❌ Small operations (overhead dominates)

❌ Sparse operations (less optimized than GPU)

❌ Dynamic shapes (requires recompilation)

Optimal TPU Usage
~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Configure for TPU
   import brainstate

   # Large batches for TPU
   batch_size = 256  # TPUs like large batches

   net = bp.LIF(1000, ...)
   brainstate.nn.init_all_states(net, batch_size=batch_size)

   # JIT is essential
   @brainstate.compile.jit
   def train_step(net, inputs, labels):
       # Dense operations work well
       # Avoid sparse operations on TPU
       return loss

   # Static shapes (avoid dynamic)
   inputs = jnp.ones((batch_size, 1000))  # Fixed shape

   # Run
   for batch in data_loader:
       loss = train_step(net, batch_inputs, batch_labels)

TPU Pods
~~~~~~~~

**Multi-TPU training:**

.. code-block:: python

   # TPU pods provide multiple TPU cores
   devices = jax.devices('tpu')
   print(f"TPU cores: {len(devices)}")

   # Use pmap for data parallelism
   @pmap
   def parallel_step(inputs):
       return net(inputs)

   # Split across TPU cores
   inputs_per_core = jnp.reshape(inputs, (len(devices), -1, 1000))
   outputs = parallel_step(inputs_per_core)

Performance Benchmarking
------------------------

Measure Speedup
~~~~~~~~~~~~~~~

.. code-block:: python

   import time
   import jax

   def benchmark_device(device_type, n_neurons=10000, n_steps=1000):
       """Benchmark simulation on specific device."""

       # Select device
       if device_type == 'cpu':
           device = jax.devices('cpu')[0]
       elif device_type == 'gpu':
           device = jax.devices('gpu')[0]
       else:
           device = jax.devices('tpu')[0]

       with jax.default_device(device):
           # Create network
           net = bp.LIF(n_neurons, V_rest=-65*u.mV, V_th=-50*u.mV, tau=10*u.ms)
           brainstate.nn.init_all_states(net)

           @brainstate.compile.jit
           def step(net, inp):
               return net(inp)

           # Warmup
           inp = brainstate.random.rand(n_neurons) * 2.0 * u.nA
           _ = step(net, inp)

           # Benchmark
           start = time.time()
           for _ in range(n_steps):
               inp = brainstate.random.rand(n_neurons) * 2.0 * u.nA
               output = step(net, inp)
           elapsed = time.time() - start

       return elapsed

   # Compare devices
   cpu_time = benchmark_device('cpu', n_neurons=10000, n_steps=1000)
   gpu_time = benchmark_device('gpu', n_neurons=10000, n_steps=1000)

   print(f"CPU time: {cpu_time:.2f}s")
   print(f"GPU time: {gpu_time:.2f}s")
   print(f"Speedup: {cpu_time/gpu_time:.1f}×")

Profile GPU Usage
~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Monitor GPU memory
   import jax

   # Get memory info (NVIDIA GPUs)
   try:
       from jax.lib import xla_bridge
       print("GPU memory allocated:", xla_bridge.get_backend().platform_memory_stats())
   except:
       print("Memory stats not available")

   # Profile with TensorBoard (advanced)
   with jax.profiler.trace("/tmp/tensorboard"):
       for _ in range(100):
           output = net(input_data)

   # View with: tensorboard --logdir=/tmp/tensorboard

Memory Management
-----------------

Check GPU Memory
~~~~~~~~~~~~~~~~

.. code-block:: python

   import jax

   # Check total memory
   for device in jax.devices('gpu'):
       try:
           # This may not work on all systems
           print(f"Device: {device}")
           print(f"Memory: {device.memory_stats()}")
       except:
           print("Memory stats not available")

Estimate Memory Requirements
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   def estimate_memory_mb(n_neurons, n_synapses, batch_size=1, dtype_bytes=4):
       """Estimate GPU memory needed.

       Args:
           n_neurons: Number of neurons
           n_synapses: Number of synapses
           batch_size: Batch size
           dtype_bytes: 4 for float32, 2 for float16
       """
       # Neuron states (V, spike, etc.) × batch
       neuron_memory = n_neurons * 3 * batch_size * dtype_bytes

       # Synapse states (g, x, etc.)
       synapse_memory = n_synapses * 2 * dtype_bytes

       # Weights
       weight_memory = n_synapses * dtype_bytes

       total_bytes = neuron_memory + synapse_memory + weight_memory
       total_mb = total_bytes / (1024 * 1024)

       return total_mb

   # Example
   mem_mb = estimate_memory_mb(
       n_neurons=100000,
       n_synapses=100000 * 100000 * 0.01,  # 1% connectivity
       batch_size=32
   )
   print(f"Estimated memory: {mem_mb:.1f} MB ({mem_mb/1024:.2f} GB)")

Clear GPU Memory
~~~~~~~~~~~~~~~~

.. code-block:: python

   import jax

   # JAX manages memory automatically
   # But you can force garbage collection

   import gc

   # Delete large arrays
   del large_array
   del network

   # Force garbage collection
   gc.collect()

   # Clear JAX compilation cache (if needed)
   jax.clear_caches()

Common Issues and Solutions
----------------------------

Issue: Out of Memory
~~~~~~~~~~~~~~~~~~~~

**Symptom:** `RESOURCE_EXHAUSTED: Out of memory`

**Solutions:**

1. **Reduce batch size:**

   .. code-block:: python

      # Try smaller batch
      brainstate.nn.init_all_states(net, batch_size=16)  # Instead of 64

2. **Use sparse connectivity:**

   .. code-block:: python

      # Reduce connectivity
      comm = brainstate.nn.EventFixedProb(..., prob=0.01)  # Instead of 0.1

3. **Use float16:**

   .. code-block:: python

      # Lower precision (experimental)
      jax.config.update('jax_default_dtype_bits', '32')  # Default
      # Note: BrainPy primarily uses float32

4. **Process in chunks:**

   .. code-block:: python

      # Split large population
      for i in range(0, n_neurons, chunk_size):
          chunk_output = process_chunk(neurons[i:i+chunk_size])

Issue: Slow First Run
~~~~~~~~~~~~~~~~~~~~~

**Symptom:** First iteration very slow

**Explanation:** JIT compilation happens on first call

**Solution:** Warm up before timing

.. code-block:: python

   @brainstate.compile.jit
   def step(net, inp):
       return net(inp)

   # Warmup (compile)
   _ = step(net, dummy_input)

   # Now fast
   for real_input in data:
       output = step(net, real_input)

Issue: GPU Not Being Used
~~~~~~~~~~~~~~~~~~~~~~~~~~

**Symptom:** Computation on CPU despite GPU available

**Check:**

.. code-block:: python

   import jax
   print("Devices:", jax.devices())
   print("Default backend:", jax.default_backend())

   # Should show GPU

**Solutions:**

1. Check installation: `pip list | grep jax`
2. Reinstall with GPU support: `pip install brainpy[cuda12]`
3. Check CUDA installation: `nvidia-smi`

Issue: Version Mismatch
~~~~~~~~~~~~~~~~~~~~~~~~

**Symptom:** `RuntimeError: CUDA error`

**Check versions:**

.. code-block:: bash

   # Check CUDA version
   nvcc --version

   # Check JAX version
   python -c "import jax; print(jax.__version__)"

**Solution:** Match JAX CUDA version with system CUDA

.. code-block:: bash

   # For CUDA 12.x
   pip install brainpy[cuda12]

   # For CUDA 11.x
   pip install brainpy[cuda11]

Best Practices
--------------

✅ **Use JIT compilation** - Essential for GPU performance

✅ **Batch operations** - Process multiple trials in parallel

✅ **Keep data on device** - Avoid CPU-GPU transfers

✅ **Use sparse connectivity** - For biological-scale networks

✅ **Profile before optimizing** - Identify real bottlenecks

✅ **Warm up JIT** - Compile before timing

✅ **Monitor memory** - Estimate before running large models

✅ **Static shapes** - Avoid dynamic shapes (causes recompilation)

❌ **Don't use GPU for small problems** - Overhead dominates

❌ **Don't transfer data unnecessarily** - Keep on GPU

❌ **Don't use dense connectivity for large networks** - Memory explosion

Example: Complete GPU Workflow
-------------------------------

.. code-block:: python

   import brainpy as bp
   import brainstate
   import brainunit as u
   import braintools
   import jax
   import time

   # 1. Check GPU availability
   print("Devices:", jax.devices())
   assert jax.default_backend() == 'gpu', "GPU not available!"

   # 2. Create large network
   class LargeNetwork(brainstate.nn.Module):
       def __init__(self, n_exc=8000, n_inh=2000):
           super().__init__()

           self.E = bp.LIF(n_exc, V_rest=-65*u.mV, V_th=-50*u.mV, tau=15*u.ms)
           self.I = bp.LIF(n_inh, V_rest=-65*u.mV, V_th=-50*u.mV, tau=10*u.ms)

           # Sparse connectivity (GPU efficient)
           self.E2E = bp.AlignPostProj(
               comm=brainstate.nn.EventFixedProb(n_exc, n_exc, prob=0.02, weight=0.5*u.mS),
               syn=bp.Expon.desc(n_exc, tau=5*u.ms),
               out=bp.CUBA.desc(),
               post=self.E
           )
           # ... more projections

       def update(self, inp_e, inp_i):
           spk_e = self.E.get_spike()
           spk_i = self.I.get_spike()

           self.E2E(spk_e)
           # ... update all projections

           self.E(inp_e)
           self.I(inp_i)

           return spk_e, spk_i

   # 3. Initialize with large batch
   net = LargeNetwork()
   batch_size = 64  # Process 64 trials in parallel
   brainstate.nn.init_all_states(net, batch_size=batch_size)

   # 4. JIT compile
   @brainstate.compile.jit
   def simulate_step(net, inp_e, inp_i):
       return net(inp_e, inp_i)

   # 5. Warmup (compilation)
   print("Compiling...")
   inp_e = brainstate.random.rand(batch_size, 8000) * 1.0 * u.nA
   inp_i = brainstate.random.rand(batch_size, 2000) * 1.0 * u.nA
   _ = simulate_step(net, inp_e, inp_i)
   print("✅ Compilation complete")

   # 6. Run simulation
   print("Running simulation...")
   n_steps = 1000

   start = time.time()
   for _ in range(n_steps):
       inp_e = brainstate.random.rand(batch_size, 8000) * 1.0 * u.nA
       inp_i = brainstate.random.rand(batch_size, 2000) * 1.0 * u.nA
       spk_e, spk_i = simulate_step(net, inp_e, inp_i)

   elapsed = time.time() - start

   print(f"✅ Simulation complete")
   print(f"   Time: {elapsed:.2f}s")
   print(f"   Throughput: {n_steps/elapsed:.1f} steps/s")
   print(f"   Speed: {batch_size * n_steps / elapsed:.1f} trials/s")

Summary
-------

**Key Points:**

- BrainPy automatically uses GPU/TPU when available
- JIT compilation is essential for GPU performance
- Batch operations maximize GPU utilization
- Keep data on device to avoid transfer overhead
- Use sparse connectivity for large networks
- GPU beneficial for networks > 1,000 neurons

**Quick Reference:**

.. code-block:: python

   # Check device
   import jax
   print(jax.devices())

   # JIT for GPU
   @brainstate.compile.jit
   def step(net, inp):
       return net(inp)

   # Batch for GPU
   brainstate.nn.init_all_states(net, batch_size=64)

   # Sparse for memory
   comm = brainstate.nn.EventFixedProb(..., prob=0.02)

See Also
--------

- :doc:`../tutorials/advanced/07-large-scale-simulations` - Optimization techniques
- :doc:`performance-optimization` - General performance tips
- JAX documentation: https://jax.readthedocs.io/
