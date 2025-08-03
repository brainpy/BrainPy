Installation
============

.. contents::
    :local:
    :depth: 2


``BrainPy`` is designed to run cross platforms, including Windows,
Linux, and MacOS. It only relies on Python libraries.


Without dependencies
--------------------

To install brainpy with minimum requirements (has installed ``jax`` and ``jaxlib`` before), you can use:

.. code-block:: bash

    pip install brainpy


CPU with all dependencies
-------------------------

To install a CPU-only version of BrainPy, which might be useful for doing local development on a laptop, you can run

.. code-block:: bash

    pip install brainpy[cpu]

    pip install BrainX[cpu]  # for whole BrainX ecosystem




GPU with all dependencies
-------------------------

BrainPy supports NVIDIA GPUs that have SM version 5.2 (Maxwell) or newer.
To install a GPU-only version of BrainPy, you can run

.. code-block:: bash

    pip install brainpy[cuda12] # for CUDA 12.0

    pip install BrainX[cuda12]  # for whole BrainX ecosystem




TPU with all dependencies
-------------------------

BrainPy supports Google Cloud TPU. To install BrainPy along with appropriate versions of jax,
you can run the following in your cloud TPU VM:

.. code-block:: bash

    pip install brainpy[tpu]  # for google TPU

    pip install BrainX[tpu]  # for whole BrainX ecosystem


