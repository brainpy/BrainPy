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


To install ``brainpylib`` (needed in dedicated operators), you can use:

.. code-block:: bash

    pip install brainpylib



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
later) platforms. The provided binary releases of `jax` and `jaxlib` for Linux and macOS
systems are available at

- for CPU: https://storage.googleapis.com/jax-releases/jax_releases.html
- for GPU: https://storage.googleapis.com/jax-releases/jax_cuda_releases.html


If you want to install a CPU-only version of `jax` and `jaxlib`, you can run

.. code-block:: bash

    pip install --upgrade "jax[cpu]" -f https://storage.googleapis.com/jax-releases/jax_releases.html

If you want to install JAX with both CPU and NVidia GPU support, you must first install
`CUDA`_ and `CuDNN`_, if they have not already been installed. Next, run

.. code-block:: bash

    pip install --upgrade "jax[cuda]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html


Alternatively, you can download the preferred release ".whl" file for jaxlib
from the above release links, and install it via ``pip``:

.. code-block:: bash

    pip install xxx-0.3.14-xxx.whl

    pip install jax==0.3.14

.. note::

   Note that the versions of `jaxlib` and `jax` should be consistent.

   For example, if you are using `jax==0.3.14`, you would better install `jax==0.3.14`.



Windows
^^^^^^^

For **Windows** users, `jax` and `jaxlib` can be installed from the community supports.
Specifically, you can install `jax` and `jaxlib` through:

.. code-block:: bash

   pip install "jax[cpu]" -f https://whls.blob.core.windows.net/unstable/index.html

If you are using GPU, you can install GPU-versioned wheels through:

.. code-block:: bash

   pip install "jax[cuda111]" -f https://whls.blob.core.windows.net/unstable/index.html

Alternatively, you can manually install you favourite version of `jax` and `jaxlib` by
downloading binary releases of JAX for Windows from https://whls.blob.core.windows.net/unstable/index.html .
Then install it via ``pip``:

.. code-block:: bash

    pip install xxx-0.3.14-xxx.whl

    pip install jax==0.3.14

WSL
^^^

Moreover, for Windows 10+ system, we recommend using `Windows Subsystem for Linux (WSL)`_.
The installation guide can be found in
`WSL Installation Guide for Windows 10/11 <https://docs.microsoft.com/en-us/windows/wsl/install-win10>`_.
Then, you can install JAX in WSL just like the installation step in Linux/MacOs.


Dependency 3: brainpylib
------------------------

Many customized operators in BrainPy are implemented in ``brainpylib``.
``brainpylib`` can also be installed through `pypi <https://pypi.org/project/brainpylib/>`_.

.. code-block:: bash

   pip install brainpylib

For GPU operators, you should compile ``brainpylib`` from source.
The details please see
`Compile GPU operators in brainpylib <../tutorial_advanced/compile_brainpylib.html>`_.


Other Dependency
----------------

In order to get full supports of BrainPy, we recommend you install the following
packages:

- `Numba`_: needed in some NumPy-based computations

.. code-block:: bash

    pip install numba

    # or

    conda install numba


- `matplotlib`_: required in some visualization functions, but now it is recommended that users explicitly import matplotlib for visualization

.. code-block:: bash

    pip install matplotlib

    # or

    conda install matplotlib


.. _NumPy: https://numpy.org/
.. _Matplotlib: https://matplotlib.org/
.. _JAX: https://github.com/google/jax
.. _Windows Subsystem for Linux (WSL): https://docs.microsoft.com/en-us/windows/wsl/about
.. _build JAX from source: https://jax.readthedocs.io/en/latest/developer.html
.. _SymPy: https://github.com/sympy/sympy
.. _Numba: https://numba.pydata.org/
.. _CUDA: https://developer.nvidia.com/cuda-downloads
.. _CuDNN: https://developer.nvidia.com/CUDNN
