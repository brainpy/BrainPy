Installation
============

.. contents::
    :local:
    :depth: 2


``BrainPy`` is designed to run cross platforms, including Windows,
Linux, and MacOS. It only relies on Python libraries.


Installation with pip
---------------------

You can install ``BrainPy`` from the `pypi <https://pypi.org/project/brainpy/>`_.
To do so, use:

.. code-block:: bash

    pip install brainpy

To update the latest BrainPy, you can use

.. code-block:: bash

    pip install -U brainpy


If you want to install the pre-release version (the latest development version)
of BrainPy, you can use:

.. code-block:: bash

   pip install --pre brainpy



Installation from source
------------------------

If you decide not to use ``pip``, you can install ``BrainPy`` from
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

Linux
^^^^^

Currently, JAX supports **Linux** (Ubuntu 16.04 or later) and **macOS** (10.12 or
later) platforms. The provided binary releases of `jax` and `jaxlib` for Linux and macOS
systems are available at

- for CPU: https://storage.googleapis.com/jax-releases/jax_releases.html
- for GPU: https://storage.googleapis.com/jax-releases/jax_cuda_releases.html


If you want to install a CPU-only version of `jax` and `jaxlib`, you can run

.. code-block:: bash

    pip install --upgrade "jax[cpu]"

If you want to install JAX with both CPU and NVidia GPU support, you must first install
`CUDA`_ and `CuDNN`_, if they have already been installed. Next, run

.. code-block:: bash

    # CUDA 12 installation
    # Note: wheels only available on linux.
    pip install --upgrade "jax[cuda12_local]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

    # CUDA 11 installation
    # Note: wheels only available on linux.
    pip install --upgrade "jax[cuda11_local]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

In the event of a version mismatch error with JAX, such as encountering an error message like:

.. code-block:: text

    CUDA backend failed to initialize: Found CUDA version 12000, but JAX was built against version 12020, which is newer. The copy of CUDA that is installed must be at least as new as the version against which JAX was built. (Set TF_CPP_MIN_LOG_LEVEL=0 and rerun for more info.)

You will need to employ an alternative installation method that aligns with your environment's CUDA version. This can be achieved using the following commands:

.. code-block:: bash

    # CUDA 12 installation
    pip install --upgrade "jax[cuda12_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

    # CUDA 11 installation
    pip install --upgrade "jax[cuda11_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html


Alternatively, you can download the preferred release ".whl" file for jaxlib
from the above release links, and install it via ``pip``:

.. code-block:: bash

    pip install xxx-0.4.15-xxx.whl

    pip install jax==0.4.15

.. note::

    Note that the versions of jaxlib and jax should be consistent.

    For example, if you are using jax==0.4.15, you would better install jax==0.4.15.


MacOS
^^^^^

If you are using macOS Intel, we recommend you first to install the Miniconda Intel installer:

1. Download the package in the link https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-x86_64.pkg
2. Then click the downloaded package and install it.


If you are using the latest M1 macOS version, you'd better to install the Miniconda M1 installer:


1. Download the package in the link https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-arm64.pkg
2. Then click the downloaded package and install it.


Finally, you can install `jax` and `jaxlib` as the same as the Linux platform.

.. code-block:: bash

   pip install --upgrade "jax[cpu]"



Windows
^^^^^^^

For **Windows** users with Python >= 3.9, `jax` and `jaxlib` can be installed
directly from the PyPi channel.

.. code-block:: bash

   pip install jax jaxlib


For **Windows** users with Python <= 3.8, `jax` and `jaxlib` can be installed
from the community supports. Specifically, you can install `jax` and `jaxlib` through:

.. code-block:: bash

   pip install "jax[cpu]" -f https://whls.blob.core.windows.net/unstable/index.html

If you are using GPU, you can install GPU-versioned wheels through:

.. code-block:: bash

   pip install "jax[cuda111]" -f https://whls.blob.core.windows.net/unstable/index.html

Alternatively, you can manually install you favourite version of `jax` and `jaxlib` by
downloading binary releases of JAX for Windows from
https://whls.blob.core.windows.net/unstable/index.html .
Then install it via ``pip``:

.. code-block:: bash

    pip install xxx-0.4.15-xxx.whl

    pip install jax==0.4.15

WSL
^^^

Moreover, for Windows 10+ system, we recommend using `Windows Subsystem for Linux (WSL)`_.
The installation guide can be found in
`WSL Installation Guide for Windows 10/11 <https://docs.microsoft.com/en-us/windows/wsl/install-win10>`_.
Then, you can install JAX in WSL just like the installation step in Linux/MacOs.


Dependency 3: brainpylib
------------------------

Many customized operators in BrainPy are implemented in ``brainpylib``.
``brainpylib`` can also be installed from pypi according to your devices.
For windows, Linux and MacOS users, ``brainpylib`` supports CPU operators.
You can install CPU-version `brainpylib` by:

.. code-block:: bash

    # CPU installation
    pip install --upgrade brainpylib

For Nvidia GPU users, ``brainpylib`` only support Linux system and WSL2 subsystem. You can install the CUDA-version by using:

.. code-block:: bash

    # CUDA 12 installation
    pip install --upgrade brainpylib-cu12x

.. code-block:: bash

    # CUDA 11 installation
    pip install --upgrade brainpylib-cu11x

Dependency 4: taichi
------------------------
Now BrainPy supports customized operators implemented in `taichi`_. You can install the latest version of `taichi`_ by:

.. code-block:: bash

    pip install -i https://pypi.taichi.graphics/simple/ taichi-nightly

.. _taichi: https://www.taichi-lang.org

And you can try it in the `operator custom with taichi <../tutorial_advanced/operator_custom_with_taichi.html>`_ tutorial page
Attention: customized operators is still in the experimental stage. If you meet any problems, please contact us through the issue page.

Running BrainPy with docker
------------------------

If you want to use BrainPy in docker, you can use the following command to pull the docker image:

.. code:: bash

   docker pull brainpy/brainpy:latest

You can then run the docker image by:

.. code:: bash

   docker run -it --platform linux/amd64 brainpy/brainpy:latest

Please notice that BrainPy docker image is based on the `ubuntu22.04` image, so it only support CPU version of BrainPy.


Running BrainPy online with binder
----------------------------------

Click on the following link to launch the Binder environment with the
BrainPy repository:

|image1|

Wait for the Binder environment to build. This might take a few moments.

Once the environment is ready, you'll be redirected to a Jupyter
notebook interface within your web browser.

.. |image1| image:: https://camo.githubusercontent.com/581c077bdbc6ca6899c86d0acc6145ae85e9d80e6f805a1071793dbe48917982/68747470733a2f2f6d7962696e6465722e6f72672f62616467655f6c6f676f2e737667
   :target: https://mybinder.org/v2/gh/brainpy/BrainPy-binder/main


.. _NumPy: https://numpy.org/
.. _Matplotlib: https://matplotlib.org/
.. _JAX: https://github.com/google/jax
.. _Windows Subsystem for Linux (WSL): https://docs.microsoft.com/en-us/windows/wsl/about
.. _build JAX from source: https://jax.readthedocs.io/en/latest/developer.html
.. _SymPy: https://github.com/sympy/sympy
.. _Numba: https://numba.pydata.org/
.. _CUDA: https://developer.nvidia.com/cuda-downloads
.. _CuDNN: https://developer.nvidia.com/CUDNN

