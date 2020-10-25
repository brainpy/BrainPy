BrainPy documentation
========================

``BrainPy`` is a microkernel framework for neuronal dynamics simulation
purely based on **native** python. It only relies on `NumPy <https://numpy.org/>`_.
However, if you want to get faster performance,you can additionally
install `Numba <http://numba.pydata.org/>`_. With `Numba`, the speed of C or FORTRAN can
be obtained in the simulation.


.. note::

    BrainPy is a project under development.
    More features are coming soon. Contributions are welcome.
    https://github.com/PKU-NIP-Lab/BrainPy


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


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
