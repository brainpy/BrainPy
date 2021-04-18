Installation
============

.. contents::
    :local:
    :depth: 1


``BrainPy`` is designed to run on across-platforms, including Windows,
GNU/Linux and OSX. It only relies on Python libraries.


Installation with pip
---------------------

You can install ``BrainPy`` from the `pypi <https://pypi.org/project/brainpy-simulator/>`_.
To do so, use:

.. code-block:: bash

    pip install -U brainpy-simulator


Installation with Anaconda
--------------------------

You can install ``BrainPy`` from the `anaconda cloud <https://anaconda.org/brainpy/brainpy-simulator>`_. To do so, use:

.. code-block:: bash

    conda install brainpy-simulator -c brainpy


Installation from source
------------------------

If you decide not to use ``conda`` or ``pip``, you can install ``BrainPy`` from
`GitHub <https://github.com/PKU-NIP-Lab/BrainPy>`_,
or `OpenI <https://git.openi.org.cn/OpenI/BrainPy>`_.

To do so, use:

.. code-block:: bash

    pip install git+https://github.com/PKU-NIP-Lab/BrainPy

Or

.. code-block:: bash

    pip install git+https://git.openi.org.cn/OpenI/BrainPy


To install the specific version of ``BrainPy``, your can use

.. code-block:: bash

    pip install -e git://github.com/PKU-NIP-Lab/BrainPy.git@V1.0.0



Package Dependency
------------------

The normal functions of ``BrainPy`` for dynamics simulation only relies on
`NumPy <https://numpy.org/>`_ and `Matplotlib <https://matplotlib.org/>`_.
You can install these two packages through

.. code-block:: bash

    pip install numpy matplotlib

    # or

    conda install numpy matplotlib


Some numerical solvers (such like `exponential euler` methods) and the
`dynamics analysis module <https://brainpy.readthedocs.io/en/latest/apis/analysis.html>`_
heavily rely on symbolic mathematics library `SymPy <https://docs.sympy.org/latest/index.html>`_.
Therefore, we highly recommend you to install sympy via

.. code-block:: bash

    pip install sympy

    # or

    conda install sympy

If you use ``BrainPy`` for your computational neuroscience project, we recommend you
to install `Numba <https://numba.pydata.org/>`_. This is because BrainPy heavily rely
on Numba for speed acceleration in almost its every module, such like connectivity,
simulation, analysis, and measurements. Numba is also a suitable framework for the
computation of sparse synaptic connections commonly used in the computational
neuroscience project. Install Numba is a piece of cake. You just need type the
following commands in you terminal:

.. code-block:: bash

    pip install numba

    # or

    conda install numba

As we stated later, ``BrainPy`` is a backend-independent neural simulator. You can
define and run your models on nearly any computation backends you prefer. These
packages can be installed by your project's need.
