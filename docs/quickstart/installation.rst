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


Package Dependency
------------------

In order to make BrainPy work normally, users should install
several dependent Python packages.


NumPy & Matplotlib
^^^^^^^^^^^^^^^^^^

The basic function of ``BrainPy`` only relies on `NumPy`_
and `Matplotlib`_. Install these two packages is very
easy, just using ``pip`` or ``conda``:

.. code-block:: bash

    pip install numpy matplotlib
    # or
    conda install numpy matplotlib

JAX
^^^

We highly recommend you to install `JAX`_.
JAX is a high-performance JIT compiler which enables users run
Python code on CPU, GPU, or TPU devices. Most functionalities of BrainPy
is based on JAX.

Currently, JAX supports **Linux** (Ubuntu 16.04 or later) and **macOS** (10.12 or
later) platforms. The provided binary releases of JAX for Linux and macOS
systems are available at https://storage.googleapis.com/jax-releases/jax_releases.html .
Users can download the preferred release ".whl" file, and install it via ``pip``:

.. code-block:: bash

    pip install xxxx.whl

For **Windows** users, JAX can be installed by the following methods:

- For Windows 10+ system, you can `Windows Subsystem for Linux (WSL)`_.
  The installation guide can be found in `WSL Installation Guide for Windows 10`_.
  Then, you can install JAX in WSL just like the installation step in Linux.
- There are several precompiled Windows wheels, like `jaxlib_0.1.68_Windows_wheels`_ and `jaxlib_0.1.61_Windows_wheels`_.
- Finally, you can also `build JAX from source`_.

More details of JAX installation can be found in https://github.com/google/jax#installation .


Numba
^^^^^

`Numba <https://numba.pydata.org/>`_ is also an excellent JIT compiler,
which can accelerate your Python codes to approach the speeds of C or FORTRAN.
Numba works best with NumPy. Many BrainPy modules rely on Numba for speed
acceleration, such like connectivity, simulation, analysis, measurements, etc.
Numba is also a suitable framework for the computation of sparse synaptic
connections commonly used in the computational neuroscience project.

Numba is a cross-platform package which can be installed on Windows, Linux, and macOS.
Install Numba is a piece of cake. You just need type the following commands in you terminal:

.. code-block:: bash

    pip install numba
    # or
    conda install numba


SymPy
^^^^^

In BrainPy, several modules need the symbolic inference by `SymPy`_. For example,
`Exponential Euler numerical solver`_ needs SymPy to compute the linear part of
your defined Python codes, phase plane and bifurcation analysis in
`dynamics analysis module`_ needs symbolic computation from SymPy.
Therefore, we highly recommend you to install sympy, just typing

.. code-block:: bash

    pip install sympy
    # or
    conda install sympy


.. _NumPy: https://numpy.org/
.. _Matplotlib: https://matplotlib.org/
.. _JAX: https://github.com/google/jax
.. _Windows Subsystem for Linux (WSL): https://docs.microsoft.com/en-us/windows/wsl/about
.. _WSL Installation Guide for Windows 10: https://docs.microsoft.com/en-us/windows/wsl/install-win10
.. _jaxlib_0.1.68_Windows_wheels: https://github.com/erwincoumans/jax/releases/tag/jax-v0.1.68_windows
.. _jaxlib_0.1.61_Windows_wheels: https://github.com/erwincoumans/jax/releases/tag/winwhl-0.1.61
.. _build JAX from source: https://jax.readthedocs.io/en/latest/developer.html
.. _SymPy: https://github.com/sympy/sympy
.. _Exponential Euler numerical solver: https://brainpy.readthedocs.io/en/latest/tutorials_advanced/ode_numerical_solvers.html#Exponential-Euler-methods
.. _dynamics analysis module: https://brainpy.readthedocs.io/en/latest/apis/analysis.html
