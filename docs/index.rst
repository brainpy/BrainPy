NumpyBrain documentation
========================

``NumpyBrain`` is a microkernel framework for SNN (spiking neural network) simulation
purely based on **native** python. It only relies on `NumPy <https://numpy.org/>`_.
However, if you want to get faster performance,you can additionally
install `Numba <http://numba.pydata.org/>`_. With `Numba`, the speed of C or FORTRAN can
be obtained in the simulation.


.. note::

    NumpyBrain is a project under development.
    More features are coming soon. Contributions are welcome.
    https://github.com/chaoming0625/NumpyBrain


.. toctree::
   :maxdepth: 1
   :caption: Introduction

   intro/installation
   intro/motivations
   intro/quick_start

.. toctree::
   :maxdepth: 2
   :caption: User guides

   guides/how_it_works
   guides/neurons
   guides/synapses
   guides/numerical_integrators

.. toctree::
   :maxdepth: 2
   :caption: API references

   apis/core
   apis/neurons
   apis/synapses
   apis/utils
   apis/changelog


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
