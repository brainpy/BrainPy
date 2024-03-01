Installation
============

.. contents::
    :local:
    :depth: 2


``BrainPy`` is designed to run cross platforms, including Windows,
Linux, and MacOS. It only relies on Python libraries.


Minimum requirements
--------------------

To install brainpy with minimum requirements (only depends on ``jax``), you can use:

.. code-block:: bash

    pip install brainpy[cpu_mini] # for CPU

    # or

    pip install brainpy[cuda_mini] -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html  # for GPU (Linux only)



CPU with all dependencies
-------------------------

To install a CPU-only version of BrainPy, which might be useful for doing local development on a laptop, you can run

.. code-block:: bash

    pip install brainpy[cpu]



GPU with all dependencies
-------------------------

BrainPy supports NVIDIA GPUs that have SM version 5.2 (Maxwell) or newer.
To install a GPU-only version of BrainPy, you can run

.. code-block:: bash

    pip install brainpy[cuda12] -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html  # for CUDA 12.0
    pip install brainpy[cuda11] -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html  # for CUDA 11.0



``brainpylib``
--------------


``brainpylib`` defines a set of useful operators for building and simulating spiking neural networks.


To install the ``brainpylib`` package on CPU devices, you can run

.. code-block:: bash

    pip install brainpylib


To install the ``brainpylib`` package on CUDA 11, you can run


.. code-block:: bash

    pip install brainpylib-cu11x


To install the ``brainpylib`` package on CUDA 12, you can run


.. code-block:: bash

    pip install brainpylib-cu12x

