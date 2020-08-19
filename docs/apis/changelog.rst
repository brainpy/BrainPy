Release notes
=============


NumpyBrain 0.2.8
----------------

API changes
~~~~~~~~~~~

- Add exponential euler method for ODE and SDE.
- Change Neurons and Synapse "update_state" function.
- Code generation for network running.
- Format signature completion and profile setting.
- Format step() and input() functions.
- Support "multi_return" in @integrate function.
- Add methods "pre2post", "post2pre", "pre2syn" and "post2syn" in utils/connect.py


NumpyBrain 0.2.7
----------------

API changes
~~~~~~~~~~~

* remove 'collect_spike()' in synapse
* change the way of synapse conductance index updating
* support to pre-define numba signatures
* fix bugs in `utils.Dict()`
* change profile setting

Models and examples
~~~~~~~~~~~~~~~~~~~

* add refractory consideration in `AMPA`, `GABA`, `STP`, `GJ` synapses
* add variable choice to link in `VoltageJumpSynapse`


NumpyBrain 0.2.6
----------------

API changes
~~~~~~~~~~~

* rename `conn.py` to `connect.py`
* rename `vis.py` to `visualize.py`

Models and examples
~~~~~~~~~~~~~~~~~~~

* add `multi_process` examples in `examples/networks`
* modify `benckmark` examples


NumpyBrain 0.2.5
----------------

API changes
~~~~~~~~~~~

* add ``@integrate`` decorator.
* move `profile.py` into `npbrain.utils` module.
* fix the ugly of `profile` setting.
* reformat neurons and synapse models.

Others
~~~~~~

* add more documents.
* add detailed framework introduction.




NumpyBrain 0.2.4
----------------

API changes
~~~~~~~~~~~
* format `synapses` codes.
* change ``synapse`` and ``neuron`` `var2index` API:
  organize `var_index` by numpy.ndarray.
* format `autojit` codes in the package.
* add `repeat=True or False` in the network `run()`.

Models and examples
~~~~~~~~~~~~~~~~~~~
* add `measure.py` in ``utils`` module.
* add `inputs.py` in ``neurons`` module.
* add `freq_inputs` in ``neurons`` module.
* add `time_inputs` in ``neurons`` module.
* add `run.py` in ``utils`` module.
* add `repeat_run` example.
* add `animation_potential()` in `vis.py`.




NumpyBrain 0.2.1
----------------

API changes
~~~~~~~~~~~
* Change synapse and neuron `var2index` API:
  organized `var_index` by `numpy.ndarray`.
* Add `raster_plot()` in `synapse.py`.
* Modify neurons API: change `state[-5]` to 'not refractory';
* Modify synapses API: change the rotation mechanism of `delay values`.

Models and examples
~~~~~~~~~~~~~~~~~~~
* Add short-term depression synapse.

Others
~~~~~~

* Add synapse structure comparison.
* Add `Brian2` error for LIF gap junction.
* Add `ANNarchy` error for HH model.
* Add benchmark codes for comparison with `Brian2` and `ANNarchy`




NumpyBrain 0.2.0
----------------

API changes
~~~~~~~~~~~

* Change `Synapses` API, making it more computationally efficient.
* Reformat connection methods.
* Change the fixed running order for "Neurons - Synapse - Monitors" to
  user defined orders in the function of `run()` in `Network`.
* remove "output_spike()" in "Neurons", add "collect_spike()" in "Synapses".
* add "variables" to Neurons and Synapse, change monitor corresponding API

Models and examples
~~~~~~~~~~~~~~~~~~~

* Add more `Neuron` examples, like Izhikevich model, HH model.
* Add AMPA synapses.
* Add GABAa and GABAb synapses.
* Add gap junction synapse.
* Add NMDA synapses.
* Add short-term plasticity synapses.




NumpyBrain 0.1.0
----------------

This is the first release of NumpyBrain. Original NumpyBrain is a lightweight
SNN library only based on pure `NumPy <https://numpy.org/>`_. It is highly
highly highly flexible. However, for large-scale networks, this framework seems
slow. Recently, we changed the API to accommodate the
`Numba <http://numba.pydata.org/>`_ backend. Thus, when encountering large-scale
spiking neural network, the model can get the C or FORTRAN-like simulation speed.


