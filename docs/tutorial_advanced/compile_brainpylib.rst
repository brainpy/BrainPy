Compile GPU operators in brainpylib
===================================

``brainpylib`` is designed to provide dedicated operators for sparse
and event-based synaptic computation.
We have already published CPU version of ``brainpylib`` on Pypi and users can install the CPU version by following instructions:

.. code-block:: bash

    pip install brainpylib

This section aims to introduce how to build up and install the GPU version. We currently did not provide GPU wheel on Pypi
and users need to build ``brainpylib`` from source. There are some prerequisites first:

- Linux platform.
- Nvidia GPU series required.
- CUDA and cuDNN have installed.

We have tested whole building process on Nvidia RTX A6000 GPU with CUDA 11.6 version.

Building ``brainpylib`` GPU version
------------------------

First, obtain the BrainPy source code:

.. code-block:: bash

    git clone https://github.com/PKU-NIP-Lab/BrainPy.git
    cd BrainPy/extensions

In ``extensions`` directory, users can compile GPU wheel:

.. code-block:: bash

    python setup_cuda.py bdist_wheel

After compilation, it's convenient for users to install the package through following instructions:

.. code-block:: bash

    pip install dist/brainpylib-*.whl

``brainpylib-*.whl`` is the generated file from compilation, which is located in ``dist`` folder.

Now users have successfully install GPU version of ``brainpylib``, and we recommend users to check if ``brainpylib`` can
be imported in the Python script.

