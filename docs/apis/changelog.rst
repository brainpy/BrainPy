Release notes
=============

BrainPy 0.3.5
-------------

- Add 'timeout' in sympy solver in neuron dynamics analysis
- Reconstruct and generalize phase plane analysis
- Generalize the repeat mode of ``Network`` to different running duration between two runs
- Remove ugly "heter_params_replace"
- Update benchmarks


BrainPy 0.3.1
-------------

- Add a more flexible way for NeuState/SynState initialization
- Fix bugs of "is_multi_return"
- Add "hand_overs", "requires" and "satisfies".
- Update documentation
- Auto-transform `range` to `numba.prange`
- Support `_obj_i`, `_pre_i`, `_post_i` for more flexible operation in scalar-based models



BrainPy 0.3.0
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



BrainPy 0.2.4
-------------

API changes
~~~~~~~~~~~

- Fix bugs


BrainPy 0.2.3
-------------

API changes
~~~~~~~~~~~

- Add "animate_1D" in ``visualization`` module
- Add "PoissonInput", "SpikeTimeInput" and "FreqInput" in ``inputs`` module
- Update phase_portrait_analyzer.py


Models and examples
~~~~~~~~~~~~~~~~~~~

- Add CANN examples


BrainPy 0.2.2
-------------

API changes
~~~~~~~~~~~

- Redesign visualization
- Redesign connectivity
- Update docs


BrainPy 0.2.1
-------------

API changes
~~~~~~~~~~~

- Fix bugs in `numba import`
- Fix bugs in `numpy` mode with `scalar` model


BrainPy 0.2.0
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

