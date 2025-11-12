Installation Guide
==================

``brainpy.state`` is a flexible, efficient, and extensible framework for computational neuroscience and
brain-inspired computation. This guide will help you install BrainPy on your system.

Requirements
------------

- Python 3.10 or later
- pip package manager
- Supported platforms: Linux (Ubuntu 16.04+), macOS (10.12+), Windows

Basic Installation
------------------

Install the latest version of BrainPy:

.. code-block:: bash

   pip install brainpy -U

This will install BrainPy with CPU support by default.

Hardware-Specific Installation
-------------------------------

Depending on your hardware, you can install BrainPy with optimized support:

CPU Only
~~~~~~~~

For CPU-only installations:

.. code-block:: bash

   pip install brainpy[cpu] -U

This is suitable for development, testing, and small-scale simulations.

GPU Support (CUDA)
~~~~~~~~~~~~~~~~~~

For NVIDIA GPU acceleration:

**CUDA 12.x:**

.. code-block:: bash

   pip install brainpy[cuda12] -U

**CUDA 13.x:**

.. code-block:: bash

   pip install brainpy[cuda13] -U

.. note::
   Make sure you have the appropriate CUDA toolkit installed on your system before installing the GPU version.

TPU Support
~~~~~~~~~~~

For Google Cloud TPU support:

.. code-block:: bash

   pip install brainpy[tpu] -U

This is typically used when running on Google Cloud Platform or Colab with TPU runtime.

Ecosystem Installation
----------------------

To install BrainPy along with the entire ecosystem of tools:

.. code-block:: bash

   pip install BrainX -U

This includes:

- ``brainpy``: Main framework
- ``brainstate``: State management and compilation backend
- ``brainunit``: Physical units system
- ``braintools``: Utilities and tools
- Additional ecosystem packages

Verifying Installation
----------------------

To verify that BrainPy is installed correctly:

.. code-block:: python

   import brainpy
   import brainstate
   import brainunit as u

   print(f"BrainPy version: {brainpy.__version__}")
   print(f"BrainState version: {brainstate.__version__}")

   # Test basic functionality
   neuron = brainpy.LIF(10)
   print("Installation successful!")

Development Installation
------------------------

If you want to install BrainPy from source for development:

.. code-block:: bash

   git clone https://github.com/brainpy/BrainPy.git
   cd BrainPy
   pip install -e .

This creates an editable installation that reflects your local changes.

Troubleshooting
---------------

Common Issues
~~~~~~~~~~~~~

**ImportError: No module named 'brainpy'**

Make sure you've activated the correct Python environment and that the installation completed successfully.

**CUDA not found**

If you installed the GPU version but get CUDA errors, ensure that:

1. Your NVIDIA drivers are up to date
2. CUDA toolkit is installed and matches the version (12.x or 13.x)
3. Your GPU is CUDA-capable

**Version Conflicts**

If you're upgrading from BrainPy 2.x, you might need to uninstall the old version first:

.. code-block:: bash

   pip uninstall brainpy
   pip install brainpy -U

Getting Help
~~~~~~~~~~~~

If you encounter issues:

- Check the `GitHub Issues <https://github.com/brainpy/BrainPy/issues>`_
- Read the documentation at `https://brainpy-state.readthedocs.io/ <https://brainpy-state.readthedocs.io/>`_
- Join our community discussions

Next Steps
----------

Now that you have BrainPy installed, you can:

- Follow the :doc:`5-minute tutorial <5min-tutorial>` for a quick introduction
- Read about :doc:`core concepts <core-concepts/index>` to understand BrainPy's architecture
- Explore the :doc:`tutorials <../tutorials/index>` for detailed guides

Using BrainPy with Binder
--------------------------

If you want to try BrainPy without installing it locally, you can use our Binder environment:

.. image:: https://mybinder.org/badge_logo.svg
   :target: https://mybinder.org/v2/gh/brainpy/BrainPy-binder/main
   :alt: Binder

This provides a pre-configured Jupyter notebook environment in your browser.
