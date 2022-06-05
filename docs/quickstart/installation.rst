Installation
============

.. contents::
    :local:
    :depth: 2


``BrainPy`` is designed to run cross platforms, including Windows,
GNU/Linux, and OSX. It only relies on Python libraries.


Installation with pip
---------------------

You can install ``BrainPy`` from the `pypi <https://pypi.org/project/brain-py/>`_.
To do so, use:

.. code-block:: bash

    pip install brainpy

To update the BrainPy version, you can use

.. code-block:: bash

    pip install -U brainpy


If you want to install the pre-release version (the latest development version)
of BrainPy, you can use:

.. code-block:: bash

   pip install --pre brainpy


Installation from source
------------------------

If you decide not to use ``conda`` or ``pip``, you can install ``BrainPy`` from
`GitHub <https://github.com/PKU-NIP-Lab/BrainPy>`_,
or `OpenI <https://git.openi.org.cn/OpenI/BrainPy>`_.

To do so, use:

.. code-block:: bash

    pip install git+https://github.com/PKU-NIP-Lab/BrainPy

    # or

    pip install git+https://git.openi.org.cn/OpenI/BrainPy


Dependency 1: NumPy
--------------------------------

In order to make BrainPy work normally, users should install
several dependent Python packages.

The basic function of ``BrainPy`` only relies on `NumPy`_, which is very
easy to install through ``pip`` or ``conda``:

.. code-block:: bash

    pip install numpy

    # or

    conda install numpy

Dependency 2: JAX
-----------------

BrainPy relies on `JAX`_. JAX is a high-performance JIT compiler which enables
users to run Python code on CPU, GPU, and TPU devices. Core functionalities of
BrainPy (>=2.0.0) have been migrated to the JAX backend.

Linux & MacOS
^^^^^^^^^^^^^

Currently, JAX supports **Linux** (Ubuntu 16.04 or later) and **macOS** (10.12 or
later) platforms. The provided binary releases of JAX for Linux and macOS
systems are available at https://storage.googleapis.com/jax-releases/jax_releases.html .

To install a CPU-only version of JAX, you can run

.. code-block:: bash

    pip install --upgrade "jax[cpu]"

If you want to install JAX with both CPU and NVidia GPU support, you must first install
`CUDA`_ and `CuDNN`_, if they have not already been installed. Next, run

.. code-block:: bash

    pip install --upgrade "jax[cuda]" -f https://storage.googleapis.com/jax-releases/jax_releases.html

Alternatively, you can download the preferred release ".whl" file for jaxlib, and install it via ``pip``:

.. code-block:: bash

    pip install xxxx.whl

    pip install jax


.. warning::
    For m1 macOS users, you should run your python environment on ``Apple`` silicon instead of ``intel``
    silicon since ``rosetta2`` cannot translate ``jaxlib``. One suggestion is uninstall miniconda3 and install
    miniforge3 for managing your python environment.

Windows
^^^^^^^

For **Windows** users, JAX can be installed by the following methods:

- **Method 1**: There are several community supported Windows build for jax, please refer
  to the github link for more details: https://github.com/cloudhan/jax-windows-builder .
  Simply speaking, the provided binary releases of JAX for Windows
  are available at https://whls.blob.core.windows.net/unstable/index.html .

  You can download the preferred release ".whl" file, and install it via ``pip``:

.. code-block:: bash

    pip install xxxx.whl

    pip install jax

- **Method 2**: For Windows 10+ system, you can use `Windows Subsystem for Linux (WSL)`_.
  The installation guide can be found in `WSL Installation Guide for Windows 10`_.
  Then, you can install JAX in WSL just like the installation step in Linux/MacOs.


- **Method 3**: You can also `build JAX from source`_.


Other Dependency
----------------

In order to get full supports of BrainPy, we recommend you install the following
packages:

- `Numba`_: needed in some NumPy-based computations

.. code-block:: bash

    pip install numba

    # or

    conda install numba

- brainpylib: needed in dedicated operators

.. code-block:: bash

    pip install brainpylib
    
 
- `matplotlib`_: required in some visualization functions, but now it is recommended that users explicitly import matplotlib for visualization

.. code-block:: bash

    pip install matplotlib

    # or

    conda install matplotlib

- `NetworkX`_: needed in the visualization of network training

.. code-block:: bash

    pip install networkx

    # or

    conda install networkx

.. _NumPy: https://numpy.org/
.. _Matplotlib: https://matplotlib.org/
.. _JAX: https://github.com/google/jax
.. _Windows Subsystem for Linux (WSL): https://docs.microsoft.com/en-us/windows/wsl/about
.. _WSL Installation Guide for Windows 10: https://docs.microsoft.com/en-us/windows/wsl/install-win10
.. _build JAX from source: https://jax.readthedocs.io/en/latest/developer.html
.. _SymPy: https://github.com/sympy/sympy
.. _Numba: https://numba.pydata.org/
.. _CUDA: https://developer.nvidia.com/cuda-downloads
.. _CuDNN: https://developer.nvidia.com/CUDNN
.. _NetworkX: https://networkx.org/
