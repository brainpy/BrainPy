
.. image:: docs/_static/logo.png
    :target: https://github.com/PKU-NIP-Lab/BrainPy/blob/master/docs/_static/logo.png


.. image:: https://img.shields.io/github/license/PKU-NIP-Lab/BrainPy
    :target: https://github.com/PKU-NIP-Lab/BrainPy/blob/master/LICENSE
    :alt: licence

.. image:: https://readthedocs.org/projects/brainpy/badge/?version=latest
    :target: https://brainpy.readthedocs.io/en/latest/?badge=latest
    :alt: documentation

.. image:: https://badge.fury.io/py/brain-py.svg
    :target: https://badge.fury.io/py/brain-py
    :alt: pypi

.. image:: https://travis-ci.com/PKU-NIP-Lab/BrainPy.svg?branch=master
    :target: https://travis-ci.com/PKU-NIP-Lab/BrainPy
    :alt: ci



Why to use BrainPy
==================

``BrainPy`` is an integrative framework for computational neuroscience and brain-inspired computation based on Just-In-Time (JIT) compilation (built on the top of `JAX`_ and `Numba`_). Core functions provided in BrainPy includes

- **JIT compilation** for class objects.
- **Numerical solvers** for ODEs, SDEs, DDEs, FDEs, and others.
- **Dynamics simulation tools** for various brain objects, like neurons, synapses, networks, soma, dendrites, channels, and even more.
- **Dynamics analysis tools** for differential equations, including phase plane analysis and bifurcation analysis, continuation analysis and sensitive analysis.
- **Seamless integration with deep learning models**, and has the speed benefit on JIT compilation.
- And more ......

`BrainPy` is designed to effectively satisfy your basic requirements:

- *Easy to learn and use*: BrainPy is only based on Python language and has little dependency requirements.
- *Flexible and transparent*: BrainPy endows the users with the fully data/logic flow control. Users can code any logic they want with BrainPy.
- *Extensible*: BrainPy allow users to extend new functionality just based on Python coding. For example, we extend the numerical integration with the ability to do numerical analysis. In such a way, the same code in BrainPy can not only be used for simulation, but also for dynamics analysis.
- *Efficient running speed*: All codes in BrainPy can be just-in-time compiled (based on `JAX`_ and `Numba`_) to run on CPU or GPU devices, thus guaranteeing its running efficiency.

.. _JAX: https://github.com/google/jax
.. _Numba: https://github.com/numba/


How to use BrainPy
==================

Step 1: installation
--------------------

``BrainPy`` is based on Python (>=3.6), and the following packages are required to be installed to use ``BrainPy``:

- NumPy >= 1.15
- Matplotlib >= 3.3

*The installation details please see documentation:*
`Quickstart/Installation <https://brainpy.readthedocs.io/en/latest/quickstart/installation.html>`_

**Method 1**: install ``BrainPy`` by using ``pip``:

.. code-block:: bash

    pip install -U brain-py

**Method 2**: install ``BrainPy`` from source:

.. code-block:: bash

    pip install git+https://github.com/PKU-NIP-Lab/BrainPy

    # or
    pip install git+https://git.openi.org.cn/OpenI/BrainPy

    # or
    pip install -e git://github.com/PKU-NIP-Lab/BrainPy.git@V1.0.0


**Other dependencies**: you want to get the full supports by BrainPy, please install the following packages:

- ``JAX >= 0.2.10``,  needed for "jax" backend and "dnn" module
- ``Numba >= 0.52``,  needed for JIT compilation on "numpy" backend
- ``SymPy >= 1.4``, needed for dynamics "analysis" module and Exponential Euler method



Step 2: useful links
--------------------

- **Documentation:** https://brainpy.readthedocs.io/
- **Source code:** https://github.com/PKU-NIP-Lab/BrainPy   or   https://git.openi.org.cn/OpenI/BrainPy
- **Bug reports:** https://github.com/PKU-NIP-Lab/BrainPy/issues   or   Email to adaduo@outlook.com
- **Examples from papers**: https://brainmodels.readthedocs.io/en/latest/from_papers.html



Step 3: comprehensive examples
------------------------------

Here list several examples of BrainPy. More detailed examples and tutorials please see `BrainModels <https://brainmodels.readthedocs.io>`_.

Neuron models
~~~~~~~~~~~~~

- `Hodgkin–Huxley neuron model`_


.. _Hodgkin–Huxley neuron model:
    https://github.com/PKU-NIP-Lab/BrainModels/blob/main/brainmodels/tensor_backend/neurons/HodgkinHuxley_model.py


Synapse models
~~~~~~~~~~~~~~

- `AMPA synapse model`_

.. _AMPA synapse model:
    https://github.com/PKU-NIP-Lab/BrainModels/blob/main/brainmodels/tensor_backend/synapses/AMPA_synapse.py


Network models
~~~~~~~~~~~~~~

- `Gamma oscillation network model`_
- `E/I balanced network model`_
- `Continuous attractor network model`_

.. _Gamma oscillation network model:
    https://brainmodels.readthedocs.io/en/latest/from_papers/Wang_1996_gamma_oscillation.html
.. _E/I balanced network model:
    https://brainmodels.readthedocs.io/en/latest/from_papers/Vreeswijk_1996_EI_net.html
.. _Continuous attractor network model:
    https://brainmodels.readthedocs.io/en/latest/from_papers/Wu_2008_CANN.html



Low-dimension dynamics analysis
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- `Phase plane analysis of the INap-IK model`_
- `Codimension 1 bifurcation analysis of FitzHugh Nagumo model`_
- `Codimension 2 bifurcation analysis of FitzHugh Nagumo model`_

.. _Phase plane analysis of the INap-IK model:
    https://brainmodels.readthedocs.io/en/latest/tutorials/dynamics_analysis/NaK_model_analysis.html
.. _Codimension 1 bifurcation analysis of FitzHugh Nagumo model:
    https://brainmodels.readthedocs.io/en/latest/tutorials/dynamics_analysis/FitzHugh_Nagumo_analysis.html
.. _Codimension 2 bifurcation analysis of FitzHugh Nagumo model:
    https://brainmodels.readthedocs.io/en/latest/tutorials/dynamics_analysis/FitzHugh_Nagumo_analysis.html#Codimension-2-bifurcation-analysis


Learning through back-propagation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~





