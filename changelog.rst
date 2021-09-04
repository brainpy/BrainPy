Release notes
=============

Version 1.1.0
-------------

This package releases a new version of BrainPy.

Highlights of core changes:

``math`` module
~~~~~~~~~~~~~~~

- support JAX backend
- support numpy backend
- support JAX ``jit``, ``vmap``, ``pmap``, ``grad`` and ``value_and_grad`` on class objects
- support numba ``jit`` on class objects
- unified numpy-like API 

``base`` module
~~~~~~~~~~~~~~~

- ``Base`` for whole Version ecosystem
- ``Function`` to wrap functions
- ``Collector`` and ``ArrayCollector`` to collect variables, integrators, nodes and others

``integrators`` module
~~~~~~~~~~~~~~~~~~~~~~

- detailed documentation for ODE numerical methods

``simulation`` module
~~~~~~~~~~~~~~~~~~~~~

- support modular and composable programming
- support multi-scale modeling
- support large-scale modeling
- support simulation on GPUs

``dnn`` module
~~~~~~~~~~~~~~

- support define deep neural networks
- loss functions
- optimizers
- activation functions
- initializations
- common used layers

``measure`` module
~~~~~~~~~~~~~~~~~~

- fix bugs on ``firing_rate()``



Version 1.0.3
-------------

Fix bugs on

- firing rate measurement
- stability analysis


Version 1.0.2
-------------

This release continues to improve the user-friendliness.

Highlights of core changes:

* Remove support for Numba-CUDA backend
* Super initialization `super(XXX, self).__init__()` can be done at anywhere
  (not required to add at the bottom of the `__init__()` function).
* Add the output message of the step function running error.
* More powerful support for Monitoring
* More powerful support for running order scheduling
* Remove `unsqueeze()` and `squeeze()` operations in ``brainpy.ops``
* Add `reshape()` operation in ``brainpy.ops``
* Improve docs for numerical solvers
* Improve tests for numerical solvers
* Add keywords checking in ODE numerical solvers
* Add more unified operations in brainpy.ops
* Support "@every" in steps and monitor functions
* Fix ODE solver bugs for class bounded function
* Add build phase in Monitor


Version 1.0.1
-------------

- Fix bugs


Version 1.0.0
-------------

- **NEW VERSION OF BRAINPY**
- Change the coding style into the object-oriented programming
- Systematically improve the documentation


Version 0.3.5
-------------

- Add 'timeout' in sympy solver in neuron dynamics analysis
- Reconstruct and generalize phase plane analysis
- Generalize the repeat mode of ``Network`` to different running duration between two runs
- Update benchmarks
- Update detailed documentation


Version 0.3.1
-------------

- Add a more flexible way for NeuState/SynState initialization
- Fix bugs of "is_multi_return"
- Add "hand_overs", "requires" and "satisfies".
- Update documentation
- Auto-transform `range` to `numba.prange`
- Support `_obj_i`, `_pre_i`, `_post_i` for more flexible operation in scalar-based models



Version 0.3.0
-------------

Computation API
~~~~~~~~~~~~~~~

- Rename "brainpy.numpy" to "brainpy.backend"
- Delete "pytorch", "tensorflow" backends
- Add "numba" requirement
- Add GPU support

Profile setting
~~~~~~~~~~~~~~~

- Delete "backend" profile setting, add "jit"

Core systems
~~~~~~~~~~~~

- Delete "autopepe8" requirement
- Delete the format code prefix
- Change keywords "_t_, _dt_, _i_" to "_t, _dt, _i"
- Change the "ST" declaration out of "requires"
- Add "repeat" mode run in Network
- Change "vector-based" to "mode" in NeuType and SynType definition

Package installation
~~~~~~~~~~~~~~~~~~~~

- Remove "pypi" installation, installation now only rely on "conda"



Version 0.2.4
-------------

API changes
~~~~~~~~~~~

- Fix bugs


Version 0.2.3
-------------

API changes
~~~~~~~~~~~

- Add "animate_1D" in ``visualization`` module
- Add "PoissonInput", "SpikeTimeInput" and "FreqInput" in ``inputs`` module
- Update phase_portrait_analyzer.py


Models and examples
~~~~~~~~~~~~~~~~~~~

- Add CANN examples


Version 0.2.2
-------------

API changes
~~~~~~~~~~~

- Redesign visualization
- Redesign connectivity
- Update docs


Version 0.2.1
-------------

API changes
~~~~~~~~~~~

- Fix bugs in `numba import`
- Fix bugs in `numpy` mode with `scalar` model


Version 0.2.0
-------------

API changes
~~~~~~~~~~~

- For computation: ``numpy``, ``numba``
- For model definition: ``NeuType``, ``SynConn``
- For model running: ``Network``, ``NeuGroup``, ``SynConn``, ``Runner``
- For numerical integration: ``integrate``, ``Integrator``, ``DiffEquation``
- For connectivity: ``One2One``, ``All2All``, ``GridFour``, ``grid_four``,
  ``GridEight``, ``grid_eight``, ``GridN``, ``FixedPostNum``, ``FixedPreNum``,
  ``FixedProb``, ``GaussianProb``, ``GaussianWeight``, ``DOG``
- For visualization: ``plot_value``, ``plot_potential``, ``plot_raster``,
  ``animation_potential``
- For measurement: ``cross_correlation``, ``voltage_fluctuation``,
  ``raster_plot``, ``firing_rate``
- For inputs: ``constant_current``, ``spike_current``, ``ramp_current``.


Models and examples
~~~~~~~~~~~~~~~~~~~

- Neuron models: ``HH model``, ``LIF model``, ``Izhikevich model``
- Synapse models: ``AMPA``, ``GABA``, ``NMDA``, ``STP``, ``GapJunction``
- Network models: ``gamma oscillation``

