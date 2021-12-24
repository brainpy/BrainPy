Installation
============

.. contents::
    :local:
    :depth: 2


``BrainPy`` is designed to run on across-platforms, including Windows,
GNU/Linux and OSX. It only relies on Python libraries.


Installation with pip
---------------------

You can install ``BrainPy`` from the `pypi <https://pypi.org/project/brain-py/>`_.
To do so, use:

.. code-block:: bash

    pip install brain-py

If you try to update the BrainPy version, you can use

.. code-block:: bash

    pip install -U brain-py


If you want to install the pre-release version (the latest development version)
of BrainPy, you can use:

.. code-block:: bash

   pip install --pre brain-py


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


Dependency 1: NumPy & Matplotlib
--------------------------------

In order to make BrainPy work normally, users should install
several dependent Python packages.

The basic function of ``BrainPy`` only relies on `NumPy`_
and `Matplotlib`_. Install these two packages is very
easy, just using ``pip`` or ``conda``:

.. code-block:: bash

    pip install numpy matplotlib

    # or

    conda install numpy matplotlib

Dependency 2: JAX
-----------------

We highly recommend you to install `JAX`_.
JAX is a high-performance JIT compiler which enables users run
Python code on CPU, GPU, or TPU devices. Most functionalities of BrainPy
is based on JAX.

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

Alternatively, you can download the preferred release ".whl" file, and install it via ``pip``:

.. code-block:: bash

    pip install xxxx.whl

Windows
^^^^^^^

For **Windows** users, JAX can be installed by the following methods:

Method 1: For Windows 10+ system, you can `Windows Subsystem for Linux (WSL)`_.
The installation guide can be found in `WSL Installation Guide for Windows 10`_.
Then, you can install JAX in WSL just like the installation step in Linux.

Method 2: There are several community supported Windows build for jax, please refer
to the github link for more details: https://github.com/cloudhan/jax-windows-builder .
Simply speaking, you can run:

.. code-block:: bash

    # for only CPU
    pip install jaxlib -f https://whls.blob.core.windows.net/unstable/index.html

    # for GPU support
    pip install <downloaded jaxlib>

Method 3: You can also `build JAX from source`_.


Other Dependency
----------------

In order to get full supports of BrainPy, we recommend you install the following
packages:

- `SymPy`_: needed in one of Exponential Euler methods

.. code-block:: bash

    pip install sympy

    # or

    conda install sympy


- `Numba`_: needed in some NumPy-based computations

.. code-block:: bash

    pip install numba

    # or

    conda install numba

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
