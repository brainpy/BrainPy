Release notes (brainpy)
#######################


brainpy 2.2.x
*************

BrainPy 2.2.x is a complete re-design of the framework,
tackling the shortcomings of brainpy 2.1.x generation,
effectively bringing it to research needs and standards.


Version 2.2.0 (2022.08.12)
==========================



This release has provided important improvements for BrainPy, including usability, speed, functions, and others.

Backwards Incompatible changes
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


1. ``brainpy.nn`` module is no longer supported and has been removed since version 2.2.0. Instead, users should use ``brainpy.train`` module for the training of BP algorithms, online learning, or offline learning algorithms, and ``brainpy.algorithms`` module for online / offline training algorithms.
2. The ``update()`` function for the model definition has been changed:

.. code-block::

   >>> # 2.1.x
   >>>
   >>> import brainpy as bp
   >>>
   >>> class SomeModel(bp.dyn.DynamicalSystem):
   >>>      def __init__(self, ):
   >>>            ......
   >>>      def update(self, t, dt):
   >>>           pass
   >>> # 2.2.x
   >>>
   >>> import brainpy as bp
   >>>
   >>> class SomeModel(bp.dyn.DynamicalSystem):
   >>>      def __init__(self, ):
   >>>            ......
   >>>      def update(self, tdi):
   >>>           t, dt = tdi.t, tdi.dt
   >>>           pass

where ``tdi`` can be defined with other names, like ``sha``\ , to represent the shared argument across modules.

Deprecations
~~~~~~~~~~~~~~~~~~~~


#. ``brainpy.dyn.xxx (neurons)`` and ``brainpy.dyn.xxx (synapse)`` are no longer supported. Please use ``brainpy.neurons``\ , ``brainpy.synapses`` modules.
#. ``brainpy.running.monitor`` has been removed.
#. ``brainpy.nn`` module has been removed.

New features
~~~~~~~~~~~~~~~~~~~~


1. ``brainpy.math.Variable`` receives a ``batch_axis`` setting to represent the batch axis of the data.

.. code-block::

   >>> import brainpy.math as bm
   >>> a = bm.Variable(bm.zeros((1, 4, 5)), batch_axis=0)
   >>> a.value = bm.zeros((2, 4, 5))  # success
   >>> a.value = bm.zeros((1, 2, 5))  # failed
   MathError: The shape of the original data is (2, 4, 5), while we got (1, 2, 5) with batch_axis=0.


2. ``brainpy.train`` provides ``brainpy.train.BPTT`` for back-propagation algorithms, ``brainpy.train.Onlinetrainer`` for online training algorithms, ``brainpy.train.OfflineTrainer`` for offline training algorithms.
3. ``brainpy.Base`` class supports ``_excluded_vars`` setting to ignore variables when retrieving variables by using ``Base.vars()`` method.

.. code-block::

   >>> class OurModel(bp.Base):
   >>>     _excluded_vars = ('a', 'b')
   >>>     def __init__(self):
   >>>         super(OurModel, self).__init__()
   >>>         self.a = bm.Variable(bm.zeros(10))
   >>>         self.b = bm.Variable(bm.ones(20))
   >>>         self.c = bm.Variable(bm.random.random(10))
   >>>
   >>> model = OurModel()
   >>> model.vars().keys()
   dict_keys(['OurModel0.c'])


4. ``brainpy.analysis.SlowPointFinder`` supports directly analyzing an instance of ``brainpy.dyn.DynamicalSystem``.

.. code-block::

   >>> hh = bp.neurons.HH(1)
   >>> finder = bp.analysis.SlowPointFinder(hh, target_vars={'V': hh.V, 'm': hh.m, 'h': hh.h, 'n': hh.n})


5. ``brainpy.datasets`` supports MNIST, FashionMNIST, and other datasets.
6. Supports defining conductance-based neuron models``.

.. code-block::

   >>> class HH(bp.dyn.CondNeuGroup):
   >>>   def __init__(self, size):
   >>>     super(HH, self).__init__(size)
   >>>
   >>>     self.INa = channels.INa_HH1952(size, )
   >>>     self.IK = channels.IK_HH1952(size, )
   >>>     self.IL = channels.IL(size, E=-54.387, g_max=0.03)


7. ``brainpy.layers`` module provides commonly used models for DNN and reservoir computing.
8. Support composable definition of synaptic models by using ``TwoEndConn``\ , ``SynOut``\ , ``SynSTP`` and ``SynLTP``.

.. code-block::

   >>> bp.synapses.Exponential(self.E, self.E, bp.conn.FixedProb(prob),
   >>>                      g_max=0.03 / scale, tau=5,
   >>>                      output=bp.synouts.COBA(E=0.),
   >>>                      stp=bp.synplast.STD())


9. Provide commonly used surrogate gradient function for spiking generation, including

   * ``brainpy.math.spike_with_sigmoid_grad``
   * ``brainpy.math.spike_with_linear_grad``
   * ``brainpy.math.spike_with_gaussian_grad``
   * ``brainpy.math.spike_with_mg_grad``

10. Provide shortcuts for GPU memory management via ``brainpy.math.disable_gpu_memory_preallocation()``\ , and ``brainpy.math.clear_buffer_memory()``.

What's Changed
~~~~~~~~~~~~~~~~~~~~


* fix `#207 <https://github.com/PKU-NIP-Lab/BrainPy/issues/207>`_\ : synapses update first, then neurons, finally delay variables by `@chaoming0625 <https://github.com/chaoming0625>`_ in `#219 <https://github.com/PKU-NIP-Lab/BrainPy/pull/219>`_
* docs: add logos by `@ztqakita <https://github.com/ztqakita>`_ in `#218 <https://github.com/PKU-NIP-Lab/BrainPy/pull/218>`_
* Add the biological NMDA model by `@c-xy17 <https://github.com/c-xy17>`_ in `#221 <https://github.com/PKU-NIP-Lab/BrainPy/pull/221>`_
* docs: fix mathjax problem by `@ztqakita <https://github.com/ztqakita>`_ in `#222 <https://github.com/PKU-NIP-Lab/BrainPy/pull/222>`_
* Add the parameter R to the LIF model by `@c-xy17 <https://github.com/c-xy17>`_ in `#224 <https://github.com/PKU-NIP-Lab/BrainPy/pull/224>`_
* new version of brainpy: V2.2.0-rc1 by `@chaoming0625 <https://github.com/chaoming0625>`_ in `#226 <https://github.com/PKU-NIP-Lab/BrainPy/pull/226>`_
* update training apis by `@chaoming0625 <https://github.com/chaoming0625>`_ in `#227 <https://github.com/PKU-NIP-Lab/BrainPy/pull/227>`_
* Update quickstart and the analysis module by `@c-xy17 <https://github.com/c-xy17>`_ in `#229 <https://github.com/PKU-NIP-Lab/BrainPy/pull/229>`_
* Eseential updates for montors, analysis, losses, and examples by `@chaoming0625 <https://github.com/chaoming0625>`_ in `#230 <https://github.com/PKU-NIP-Lab/BrainPy/pull/230>`_
* add numpy op tests by `@ztqakita <https://github.com/ztqakita>`_ in `#231 <https://github.com/PKU-NIP-Lab/BrainPy/pull/231>`_
* Integrated simulation, simulaton and analysis by `@chaoming0625 <https://github.com/chaoming0625>`_ in `#232 <https://github.com/PKU-NIP-Lab/BrainPy/pull/232>`_
* update docs by `@chaoming0625 <https://github.com/chaoming0625>`_ in `#233 <https://github.com/PKU-NIP-Lab/BrainPy/pull/233>`_
* unify ``brainpy.layers`` with other modules in ``brainpy.dyn`` by `@chaoming0625 <https://github.com/chaoming0625>`_ in `#234 <https://github.com/PKU-NIP-Lab/BrainPy/pull/234>`_
* fix bugs by `@chaoming0625 <https://github.com/chaoming0625>`_ in `#235 <https://github.com/PKU-NIP-Lab/BrainPy/pull/235>`_
* update apis, docs, examples and others by `@chaoming0625 <https://github.com/chaoming0625>`_ in `#236 <https://github.com/PKU-NIP-Lab/BrainPy/pull/236>`_
* fixes by `@chaoming0625 <https://github.com/chaoming0625>`_ in `#237 <https://github.com/PKU-NIP-Lab/BrainPy/pull/237>`_
* fix: add dtype promotion = standard by `@ztqakita <https://github.com/ztqakita>`_ in `#239 <https://github.com/PKU-NIP-Lab/BrainPy/pull/239>`_
* updates by `@chaoming0625 <https://github.com/chaoming0625>`_ in `#240 <https://github.com/PKU-NIP-Lab/BrainPy/pull/240>`_
* update training docs by `@chaoming0625 <https://github.com/chaoming0625>`_ in `#241 <https://github.com/PKU-NIP-Lab/BrainPy/pull/241>`_
* change doc path/organization by `@chaoming0625 <https://github.com/chaoming0625>`_ in `#242 <https://github.com/PKU-NIP-Lab/BrainPy/pull/242>`_
* Update advanced docs by `@chaoming0625 <https://github.com/chaoming0625>`_ in `#243 <https://github.com/PKU-NIP-Lab/BrainPy/pull/243>`_
* update quickstart docs & enable jit error checking by `@chaoming0625 <https://github.com/chaoming0625>`_ in `#244 <https://github.com/PKU-NIP-Lab/BrainPy/pull/244>`_
* update apis and examples by `@chaoming0625 <https://github.com/chaoming0625>`_ in `#245 <https://github.com/PKU-NIP-Lab/BrainPy/pull/245>`_
* update apis and tests by `@chaoming0625 <https://github.com/chaoming0625>`_ in `#246 <https://github.com/PKU-NIP-Lab/BrainPy/pull/246>`_
* Docs update and bugs fixed by `@ztqakita <https://github.com/ztqakita>`_ in `#247 <https://github.com/PKU-NIP-Lab/BrainPy/pull/247>`_
* version 2.2.0 by `@chaoming0625 <https://github.com/chaoming0625>`_ in `#248 <https://github.com/PKU-NIP-Lab/BrainPy/pull/248>`_
* add norm and pooling & fix bugs in operators by `@ztqakita <https://github.com/ztqakita>`_ in `#249 <https://github.com/PKU-NIP-Lab/BrainPy/pull/249>`_

**Full Changelog**: `V2.1.12...V2.2.0 <https://github.com/PKU-NIP-Lab/BrainPy/compare/V2.1.12...V2.2.0>`_




brainpy 2.1.x
*************



Version 2.1.12 (2022.05.17)
===========================


Highlights
~~~~~~~~~~

This release is excellent. We have made important improvements.

1. We provide dozens of random sampling in NumPy which are not
   supportted in JAX, such as ``brainpy.math.random.bernoulli``,
   ``brainpy.math.random.lognormal``, ``brainpy.math.random.binomial``,
   ``brainpy.math.random.chisquare``, ``brainpy.math.random.dirichlet``,
   ``brainpy.math.random.geometric``, ``brainpy.math.random.f``,
   ``brainpy.math.random.hypergeometric``,
   ``brainpy.math.random.logseries``,
   ``brainpy.math.random.multinomial``,
   ``brainpy.math.random.multivariate_normal``,
   ``brainpy.math.random.negative_binomial``,
   ``brainpy.math.random.noncentral_chisquare``,
   ``brainpy.math.random.noncentral_f``, ``brainpy.math.random.power``,
   ``brainpy.math.random.rayleigh``, ``brainpy.math.random.triangular``,
   ``brainpy.math.random.vonmises``, ``brainpy.math.random.wald``,
   ``brainpy.math.random.weibull``
2. make efficient checking on numerical values. Instead of direct
   ``id_tap()`` checking which has large overhead, currently
   ``brainpy.tools.check_erro_in_jit()`` is highly efficient.
3. Fix ``JaxArray`` operator errors on ``None``
4. improve oo-to-function transformation speeds
5. ``io`` works: ``.save_states()`` and ``.load_states()``

What’s Changed
~~~~~~~~~~~~~~

-  support dtype setting in array interchange functions by
   [@chaoming0625](https://github.com/chaoming0625) in
   `#209 <https://github.com/PKU-NIP-Lab/BrainPy/pull/209>`__
-  fix `#144 <https://github.com/PKU-NIP-Lab/BrainPy/issues/144>`__:
   operations on None raise errors by
   [@chaoming0625](https://github.com/chaoming0625) in
   `#210 <https://github.com/PKU-NIP-Lab/BrainPy/pull/210>`__
-  add tests and new functions for random sampling by
   [@c-xy17](https://github.com/c-xy17) in
   `#213 <https://github.com/PKU-NIP-Lab/BrainPy/pull/213>`__
-  feat: fix ``io`` for brainpy.Base by
   [@chaoming0625](https://github.com/chaoming0625) in
   `#211 <https://github.com/PKU-NIP-Lab/BrainPy/pull/211>`__
-  update advanced tutorial documentation by
   [@chaoming0625](https://github.com/chaoming0625) in
   `#212 <https://github.com/PKU-NIP-Lab/BrainPy/pull/212>`__
-  fix `#149 <https://github.com/PKU-NIP-Lab/BrainPy/issues/149>`__
   (dozens of random samplings in NumPy) and fix JaxArray op errors by
   [@chaoming0625](https://github.com/chaoming0625) in
   `#216 <https://github.com/PKU-NIP-Lab/BrainPy/pull/216>`__
-  feat: efficient checking on numerical values by
   [@chaoming0625](https://github.com/chaoming0625) in
   `#217 <https://github.com/PKU-NIP-Lab/BrainPy/pull/217>`__

**Full Changelog**:
`V2.1.11...V2.1.12 <https://github.com/PKU-NIP-Lab/BrainPy/compare/V2.1.11...V2.1.12>`__



Version 2.1.11 (2022.05.15)
===========================


What's Changed
~~~~~~~~~~~~~~

* fix: cross-correlation bug by `@ztqakita <https://github.com/ztqakita>`_ in `#201 <https://github.com/PKU-NIP-Lab/BrainPy/pull/201>`_
* update apis, test and docs of numpy ops by `@chaoming0625 <https://github.com/chaoming0625>`_ in `#202 <https://github.com/PKU-NIP-Lab/BrainPy/pull/202>`_
* docs: add sphinx_book_theme by `@ztqakita <https://github.com/ztqakita>`_ in `#203 <https://github.com/PKU-NIP-Lab/BrainPy/pull/203>`_
* fix: add requirements-doc.txt by `@ztqakita <https://github.com/ztqakita>`_ in `#204 <https://github.com/PKU-NIP-Lab/BrainPy/pull/204>`_
* update control flow, integrators, operators, and docs by `@chaoming0625 <https://github.com/chaoming0625>`_ in `#205 <https://github.com/PKU-NIP-Lab/BrainPy/pull/205>`_
* improve oo-to-function transformation speed by `@chaoming0625 <https://github.com/chaoming0625>`_ in `#208 <https://github.com/PKU-NIP-Lab/BrainPy/pull/208>`_

**Full Changelog**\ : `V2.1.10...V2.1.11 <https://github.com/PKU-NIP-Lab/BrainPy/compare/V2.1.10...V2.1.11>`_



Version 2.1.10 (2022.05.05)
===========================


What's Changed
~~~~~~~~~~~~~~

* update control flow APIs and Docs by `@chaoming0625 <https://github.com/chaoming0625>`_ in `#192 <https://github.com/PKU-NIP-Lab/BrainPy/pull/192>`_
* doc: update docs of dynamics simulation by `@chaoming0625 <https://github.com/chaoming0625>`_ in `#193 <https://github.com/PKU-NIP-Lab/BrainPy/pull/193>`_
* fix `#125 <https://github.com/PKU-NIP-Lab/BrainPy/issues/125>`_: add channel models and two-compartment Pinsky-Rinzel model by `@chaoming0625 <https://github.com/chaoming0625>`_ in `#194 <https://github.com/PKU-NIP-Lab/BrainPy/pull/194>`_
* JIT errors do not change Variable values by `@chaoming0625 <https://github.com/chaoming0625>`_ in `#195 <https://github.com/PKU-NIP-Lab/BrainPy/pull/195>`_
* fix a bug in math.activations.py by `@c-xy17 <https://github.com/c-xy17>`_ in `#196 <https://github.com/PKU-NIP-Lab/BrainPy/pull/196>`_
* Functionalinaty improvements by `@chaoming0625 <https://github.com/chaoming0625>`_ in `#197 <https://github.com/PKU-NIP-Lab/BrainPy/pull/197>`_
* update rate docs by `@chaoming0625 <https://github.com/chaoming0625>`_ in `#198 <https://github.com/PKU-NIP-Lab/BrainPy/pull/198>`_
* update brainpy.dyn doc by `@chaoming0625 <https://github.com/chaoming0625>`_ in `#199 <https://github.com/PKU-NIP-Lab/BrainPy/pull/199>`_

**Full Changelog**\ : `V2.1.8...V2.1.10 <https://github.com/PKU-NIP-Lab/BrainPy/compare/V2.1.8...V2.1.10>`_



Version 2.1.8 (2022.04.26)
==========================


What's Changed
~~~~~~~~~~~~~~

* Fix `#120 <https://github.com/PKU-NIP-Lab/BrainPy/issues/120>`_ by `@chaoming0625 <https://github.com/chaoming0625>`_ in `#178 <https://github.com/PKU-NIP-Lab/BrainPy/pull/178>`_
* feat: brainpy.Collector supports addition and subtraction by `@chaoming0625 <https://github.com/chaoming0625>`_ in `#179 <https://github.com/PKU-NIP-Lab/BrainPy/pull/179>`_
* feat: delay variables support "indices" and "reset()" function by `@chaoming0625 <https://github.com/chaoming0625>`_ in `#180 <https://github.com/PKU-NIP-Lab/BrainPy/pull/180>`_
* Support reset functions in neuron and synapse models by `@chaoming0625 <https://github.com/chaoming0625>`_ in `#181 <https://github.com/PKU-NIP-Lab/BrainPy/pull/181>`_
* ``update()`` function on longer need ``_t`` and ``_dt`` by `@chaoming0625 <https://github.com/chaoming0625>`_ in `#183 <https://github.com/PKU-NIP-Lab/BrainPy/pull/183>`_
* small updates by `@chaoming0625 <https://github.com/chaoming0625>`_ in `#188 <https://github.com/PKU-NIP-Lab/BrainPy/pull/188>`_
* feat: easier control flows with ``brainpy.math.ifelse`` by `@chaoming0625 <https://github.com/chaoming0625>`_ in `#189 <https://github.com/PKU-NIP-Lab/BrainPy/pull/189>`_
* feat: update delay couplings of ``DiffusiveCoupling`` and ``AdditiveCouping`` by `@chaoming0625 <https://github.com/chaoming0625>`_ in `#190 <https://github.com/PKU-NIP-Lab/BrainPy/pull/190>`_
* update version and changelog by `@chaoming0625 <https://github.com/chaoming0625>`_ in `#191 <https://github.com/PKU-NIP-Lab/BrainPy/pull/191>`_

**Full Changelog**\ : `V2.1.7...V2.1.8 <https://github.com/PKU-NIP-Lab/BrainPy/compare/V2.1.7...V2.1.8>`_



Version 2.1.7 (2022.04.22)
==========================


What's Changed
~~~~~~~~~~~~~~

* synapse models support heterogeneuos weights by `@chaoming0625 <https://github.com/chaoming0625>`_ in `#170 <https://github.com/PKU-NIP-Lab/BrainPy/pull/170>`_
* more efficient synapse implementation by `@chaoming0625 <https://github.com/chaoming0625>`_ in `#171 <https://github.com/PKU-NIP-Lab/BrainPy/pull/171>`_
* fix input models in brainpy.dyn by `@chaoming0625 <https://github.com/chaoming0625>`_ in `#172 <https://github.com/PKU-NIP-Lab/BrainPy/pull/172>`_
* fix: np array astype by `@ztqakita <https://github.com/ztqakita>`_ in `#173 <https://github.com/PKU-NIP-Lab/BrainPy/pull/173>`_
* update README: 'brain-py' to 'brainpy' by `@chaoming0625 <https://github.com/chaoming0625>`_ in `#174 <https://github.com/PKU-NIP-Lab/BrainPy/pull/174>`_
* fix: fix the updating rules in the STP model by `@c-xy17 <https://github.com/c-xy17>`_ in `#176 <https://github.com/PKU-NIP-Lab/BrainPy/pull/176>`_
* Updates and fixes by `@chaoming0625 <https://github.com/chaoming0625>`_ in `#177 <https://github.com/PKU-NIP-Lab/BrainPy/pull/177>`_

**Full Changelog**\ : `V2.1.5...V2.1.7 <https://github.com/PKU-NIP-Lab/BrainPy/compare/V2.1.5...V2.1.7>`_


Version 2.1.5 (2022.04.18)
==========================


What's Changed
~~~~~~~~~~~~~~

* ``brainpy.math.random.shuffle`` is numpy like by `@chaoming0625 <https://github.com/chaoming0625>`_ in `#153 <https://github.com/PKU-NIP-Lab/BrainPy/pull/153>`_
* update LICENSE by `@chaoming0625 <https://github.com/chaoming0625>`_ in `#155 <https://github.com/PKU-NIP-Lab/BrainPy/pull/155>`_
* docs: add m1 warning by `@ztqakita <https://github.com/ztqakita>`_ in `#154 <https://github.com/PKU-NIP-Lab/BrainPy/pull/154>`_
* compatible apis of 'brainpy.math' with those of 'jax.numpy' in most modules by `@chaoming0625 <https://github.com/chaoming0625>`_ in `#156 <https://github.com/PKU-NIP-Lab/BrainPy/pull/156>`_
* Important updates by `@chaoming0625 <https://github.com/chaoming0625>`_ in `#157 <https://github.com/PKU-NIP-Lab/BrainPy/pull/157>`_
* Updates by `@chaoming0625 <https://github.com/chaoming0625>`_ in `#159 <https://github.com/PKU-NIP-Lab/BrainPy/pull/159>`_
* Add LayerNorm, GroupNorm, and InstanceNorm as nn_nodes in normalization.py by `@c-xy17 <https://github.com/c-xy17>`_ in `#162 <https://github.com/PKU-NIP-Lab/BrainPy/pull/162>`_
* feat: add conv & pooling nodes by `@ztqakita <https://github.com/ztqakita>`_ in `#161 <https://github.com/PKU-NIP-Lab/BrainPy/pull/161>`_
* fix: update setup.py by `@ztqakita <https://github.com/ztqakita>`_ in `#163 <https://github.com/PKU-NIP-Lab/BrainPy/pull/163>`_
* update setup.py by `@chaoming0625 <https://github.com/chaoming0625>`_ in `#165 <https://github.com/PKU-NIP-Lab/BrainPy/pull/165>`_
* fix: change trigger condition by `@ztqakita <https://github.com/ztqakita>`_ in `#166 <https://github.com/PKU-NIP-Lab/BrainPy/pull/166>`_
* fix: add build_conn() function by `@ztqakita <https://github.com/ztqakita>`_ in `#164 <https://github.com/PKU-NIP-Lab/BrainPy/pull/164>`_
* update synapses by `@chaoming0625 <https://github.com/chaoming0625>`_ in `#167 <https://github.com/PKU-NIP-Lab/BrainPy/pull/167>`_
* get the deserved name: brainpy by `@chaoming0625 <https://github.com/chaoming0625>`_ in `#168 <https://github.com/PKU-NIP-Lab/BrainPy/pull/168>`_
* update tests by `@chaoming0625 <https://github.com/chaoming0625>`_ in `#169 <https://github.com/PKU-NIP-Lab/BrainPy/pull/169>`_

**Full Changelog**\ : `V2.1.4...V2.1.5 <https://github.com/PKU-NIP-Lab/BrainPy/compare/V2.1.4...V2.1.5>`_



Version 2.1.4 (2022.04.04)
==========================


What's Changed
~~~~~~~~~~~~~~

* fix doc parsing bug by `@chaoming0625 <https://github.com/chaoming0625>`_ in `#127 <https://github.com/PKU-NIP-Lab/BrainPy/pull/127>`_
* Update overview_of_dynamic_model.ipynb by `@c-xy17 <https://github.com/c-xy17>`_ in `#129 <https://github.com/PKU-NIP-Lab/BrainPy/pull/129>`_
* Reorganization of ``brainpylib.custom_op`` and adding interface in ``brainpy.math`` by `@ztqakita <https://github.com/ztqakita>`_ in `#128 <https://github.com/PKU-NIP-Lab/BrainPy/pull/128>`_
* Fix: modify ``register_op`` and brainpy.math interface by `@ztqakita <https://github.com/ztqakita>`_ in `#130 <https://github.com/PKU-NIP-Lab/BrainPy/pull/130>`_
* new features about RNN training and delay differential equations by `@chaoming0625 <https://github.com/chaoming0625>`_ in `#132 <https://github.com/PKU-NIP-Lab/BrainPy/pull/132>`_
* Fix `#123 <https://github.com/PKU-NIP-Lab/BrainPy/issues/123>`_\ : Add low-level operators docs and modify register_op by `@ztqakita <https://github.com/ztqakita>`_ in `#134 <https://github.com/PKU-NIP-Lab/BrainPy/pull/134>`_
* feat: add generate_changelog by `@ztqakita <https://github.com/ztqakita>`_ in `#135 <https://github.com/PKU-NIP-Lab/BrainPy/pull/135>`_
* fix `#133 <https://github.com/PKU-NIP-Lab/BrainPy/issues/133>`_\ , support batch size training with offline algorithms by `@chaoming0625 <https://github.com/chaoming0625>`_ in `#136 <https://github.com/PKU-NIP-Lab/BrainPy/pull/136>`_
* fix `#84 <https://github.com/PKU-NIP-Lab/BrainPy/issues/84>`_\ : support online training algorithms by `@chaoming0625 <https://github.com/chaoming0625>`_ in `#137 <https://github.com/PKU-NIP-Lab/BrainPy/pull/137>`_
* feat: add the batch normalization node by `@c-xy17 <https://github.com/c-xy17>`_ in `#138 <https://github.com/PKU-NIP-Lab/BrainPy/pull/138>`_
* fix: fix shape checking error by `@chaoming0625 <https://github.com/chaoming0625>`_ in `#139 <https://github.com/PKU-NIP-Lab/BrainPy/pull/139>`_
* solve `#131 <https://github.com/PKU-NIP-Lab/BrainPy/issues/131>`_\ , support efficient synaptic computation for special connection types by `@chaoming0625 <https://github.com/chaoming0625>`_ in `#140 <https://github.com/PKU-NIP-Lab/BrainPy/pull/140>`_
* feat: update the API and test for batch normalization by `@c-xy17 <https://github.com/c-xy17>`_ in `#142 <https://github.com/PKU-NIP-Lab/BrainPy/pull/142>`_
* Node is default trainable by `@chaoming0625 <https://github.com/chaoming0625>`_ in `#143 <https://github.com/PKU-NIP-Lab/BrainPy/pull/143>`_
* Updates training apis and docs by `@chaoming0625 <https://github.com/chaoming0625>`_ in `#145 <https://github.com/PKU-NIP-Lab/BrainPy/pull/145>`_
* fix: add dependencies and update version by `@ztqakita <https://github.com/ztqakita>`_ in `#147 <https://github.com/PKU-NIP-Lab/BrainPy/pull/147>`_
* update requirements by `@chaoming0625 <https://github.com/chaoming0625>`_ in `#146 <https://github.com/PKU-NIP-Lab/BrainPy/pull/146>`_
* data pass of the Node is default SingleData by `@chaoming0625 <https://github.com/chaoming0625>`_ in `#148 <https://github.com/PKU-NIP-Lab/BrainPy/pull/148>`_

**Full Changelog**\ : `V2.1.3...V2.1.4 <https://github.com/PKU-NIP-Lab/BrainPy/compare/V2.1.3...V2.1.4>`_



Version 2.1.3 (2022.03.27)
==========================

This release improves the functionality and usability of BrainPy. Core changes include

* support customization of low-level operators by using Numba
* fix bugs

What's Changed
~~~~~~~~~~~~~~

* Provide custom operators written in numba for jax jit by `@ztqakita <https://github.com/ztqakita>`_ in `#122 <https://github.com/PKU-NIP-Lab/BrainPy/pull/122>`_
* fix DOGDecay bugs, add more features by `@chaoming0625 <https://github.com/chaoming0625>`_ in `#124 <https://github.com/PKU-NIP-Lab/BrainPy/pull/124>`_
* fix bugs by `@chaoming0625 <https://github.com/chaoming0625>`_ in `#126 <https://github.com/PKU-NIP-Lab/BrainPy/pull/126>`_

**Full Changelog** : `V2.1.2...V2.1.3 <https://github.com/PKU-NIP-Lab/BrainPy/compare/V2.1.2...V2.1.3>`_




Version 2.1.2 (2022.03.23)
==========================

This release improves the functionality and usability of BrainPy. Core changes include

- support rate-based whole-brain modeling
- add more neuron models, including rate neurons/synapses
- support Python 3.10
- improve delays etc. APIs


What's Changed
~~~~~~~~~~~~~~

* fix matplotlib dependency on "brainpy.analysis" module by `@chaoming0625 <https://github.com/chaoming0625>`_ in `#110 <https://github.com/PKU-NIP-Lab/BrainPy/pull/110>`_
* Sync master to brainpy-2.x branch by `@ztqakita <https://github.com/ztqakita>`_ in `#111 <https://github.com/PKU-NIP-Lab/BrainPy/pull/111>`_
* add py3.6 test & delete multiple macos env by `@ztqakita <https://github.com/ztqakita>`_ in `#112 <https://github.com/PKU-NIP-Lab/BrainPy/pull/112>`_
* Modify ci by `@ztqakita <https://github.com/ztqakita>`_ in `#113 <https://github.com/PKU-NIP-Lab/BrainPy/pull/113>`_
* Add py3.10 test by `@ztqakita <https://github.com/ztqakita>`_ in `#115 <https://github.com/PKU-NIP-Lab/BrainPy/pull/115>`_
* update python version by `@chaoming0625 <https://github.com/chaoming0625>`_ in `#114 <https://github.com/PKU-NIP-Lab/BrainPy/pull/114>`_
* add brainpylib mac py3.10 by `@ztqakita <https://github.com/ztqakita>`_ in `#116 <https://github.com/PKU-NIP-Lab/BrainPy/pull/116>`_
* Enhance measure/input/brainpylib by `@chaoming0625 <https://github.com/chaoming0625>`_ in `#117 <https://github.com/PKU-NIP-Lab/BrainPy/pull/117>`_
* fix `#105 <https://github.com/PKU-NIP-Lab/BrainPy/issues/105>`_\ : Add customize connections docs by `@ztqakita <https://github.com/ztqakita>`_ in `#118 <https://github.com/PKU-NIP-Lab/BrainPy/pull/118>`_
* fix bugs by `@chaoming0625 <https://github.com/chaoming0625>`_ in `#119 <https://github.com/PKU-NIP-Lab/BrainPy/pull/119>`_
* Whole brain modeling by `@chaoming0625 <https://github.com/chaoming0625>`_ in `#121 <https://github.com/PKU-NIP-Lab/BrainPy/pull/121>`_

**Full Changelog**: `V2.1.1...V2.1.2 <https://github.com/PKU-NIP-Lab/BrainPy/compare/V2.1.1...V2.1.2>`_


Version 2.1.1 (2022.03.18)
==========================

This release continues to update the functionality of BrainPy. Core changes include

- numerical solvers for fractional differential equations
- more standard ``brainpy.nn`` interfaces


New Features
~~~~~~~~~~~~

- Numerical solvers for fractional differential equations
    - ``brainpy.fde.CaputoEuler``
    - ``brainpy.fde.CaputoL1Schema``
    - ``brainpy.fde.GLShortMemory``
- Fractional neuron models
    - ``brainpy.dyn.FractionalFHR``
    - ``brainpy.dyn.FractionalIzhikevich``
- support ``shared_kwargs`` in `RNNTrainer` and `RNNRunner`


Version 2.1.0 (2022.03.14)
==========================


Highlights
~~~~~~~~~~

We are excited to announce the release of BrainPy 2.1.0. This release is composed of nearly
270 commits since 2.0.2, made by `Chaoming Wang <https://github.com/chaoming0625>`_,
`Xiaoyu Chen <mailto:c-xy17@tsinghua.org.cn>`_, and `Tianqiu Zhang <mailto:tianqiuakita@gmail.com>`_ .

BrainPy 2.1.0 updates are focused on improving usability, functionality, and stability of BrainPy.
Highlights of version 2.1.0 include:

- New module ``brainpy.dyn`` for dynamics building and simulation. It is composed of many
  neuron models, synapse models, and others.
- New module ``brainpy.nn`` for neural network building and training. It supports to
  define reservoir models, artificial neural networks, ridge regression training,
  and back-propagation through time training.
- New module ``brainpy.datasets`` for convenient dataset construction and initialization.
- New module ``brainpy.integrators.dde`` for numerical integration of delay differential equations.
- Add more numpy-like operators in ``brainpy.math`` module.
- Add automatic continuous integration on Linux, Windows, and MacOS platforms.
- Fully update brainpy documentation.
- Fix bugs on ``brainpy.analysis`` and ``brainpy.math.autograd``


Incompatible changes
~~~~~~~~~~~~~~~~~~~~

- Remove ``brainpy.math.numpy`` module.
- Remove numba requirements
- Remove matplotlib requirements
- Remove `steps` in ``brainpy.dyn.DynamicalSystem``
- Remove travis CI


New Features
~~~~~~~~~~~~

- ``brainpy.ddeint`` for numerical integration of delay differential equations,
  the supported methods include:
    - Euler
    - MidPoint
    - Heun2
    - Ralston2
    - RK2
    - RK3
    - Heun3
    - Ralston3
    - SSPRK3
    - RK4
    - Ralston4
    - RK4Rule38
- set default int/float/complex types
    - ``brainpy.math.set_dfloat()``
    - ``brainpy.math.set_dint()``
    - ``brainpy.math.set_dcomplex()``
- Delay variables
    - ``brainpy.math.FixedLenDelay``
    - ``brainpy.math.NeutralDelay``
- Dedicated operators
    - ``brainpy.math.sparse_matmul()``
- More numpy-like operators
- Neural network building ``brainpy.nn``
- Dynamics model building and simulation ``brainpy.dyn``


Version 2.0.2 (2022.02.11)
==========================

There are important updates by `Chaoming Wang <https://github.com/chaoming0625>`_
in BrainPy 2.0.2.

- provide ``pre2post_event_prod`` operator
- support array creation from a list/tuple of JaxArray in ``brainpy.math.asarray`` and ``brainpy.math.array``
- update ``brainpy.ConstantDelay``, add ``.latest`` and ``.oldest`` attributes
- add ``brainpy.IntegratorRunner`` support for efficient simulation of brainpy integrators
- support auto finding of RandomState when JIT SDE integrators
- fix bugs in SDE ``exponential_euler`` method
- move ``parallel`` running APIs into ``brainpy.simulation``
- add ``brainpy.math.syn2post_mean``, ``brainpy.math.syn2post_softmax``,
  ``brainpy.math.pre2post_mean`` and ``brainpy.math.pre2post_softmax`` operators



Version 2.0.1 (2022.01.31)
==========================

Today we release BrainPy 2.0.1. This release is composed of over
70 commits since 2.0.0, made by `Chaoming Wang <https://github.com/chaoming0625>`_,
`Xiaoyu Chen <mailto:c-xy17@tsinghua.org.cn>`_, and
`Tianqiu Zhang <mailto:tianqiuakita@gmail.com>`_ .

BrainPy 2.0.0 updates are focused on improving documentation and operators.
Core changes include:

- Improve ``brainpylib`` operators
- Complete documentation for programming system
- Add more numpy APIs
- Add ``jaxfwd`` in autograd module
- And other changes


Version 2.0.0.1 (2022.01.05)
============================

- Add progress bar in ``brainpy.StructRunner``


Version 2.0.0 (2021.12.31)
==========================

Start a new version of BrainPy.

Highlight
~~~~~~~~~

We are excited to announce the release of BrainPy 2.0.0. This release is composed of over
260 commits since 1.1.7, made by `Chaoming Wang <https://github.com/chaoming0625>`_,
`Xiaoyu Chen <mailto:c-xy17@tsinghua.org.cn>`_, and `Tianqiu Zhang <mailto:tianqiuakita@gmail.com>`_ .

BrainPy 2.0.0 updates are focused on improving performance, usability and consistence of BrainPy.
All the computations are migrated into JAX. Model ``building``, ``simulation``, ``training``
and ``analysis`` are all based on JAX. Highlights of version 2.0.0 include:

- `brainpylib <https://pypi.org/project/brainpylib/>`_ are provided to dedicated operators for
  brain dynamics programming
- Connection APIs in ``brainpy.conn`` module are more efficient.
- Update analysis tools for low-dimensional and high-dimensional systems in ``brainpy.analysis`` module.
- Support more general Exponential Euler methods based on automatic differentiation.
- Improve the usability and consistence of ``brainpy.math`` module.
- Remove JIT compilation based on Numba.
- Separate brain building with brain simulation.


Incompatible changes
~~~~~~~~~~~~~~~~~~~~

- remove ``brainpy.math.use_backend()``
- remove ``brainpy.math.numpy`` module
- no longer support ``.run()`` in ``brainpy.DynamicalSystem`` (see New Features)
- remove ``brainpy.analysis.PhasePlane`` (see New Features)
- remove ``brainpy.analysis.Bifurcation`` (see New Features)
- remove ``brainpy.analysis.FastSlowBifurcation`` (see New Features)


New Features
~~~~~~~~~~~~

- Exponential Euler method based on automatic differentiation
    - ``brainpy.ode.ExpEulerAuto``
- Numerical optimization based low-dimensional analyzers:
    - ``brainpy.analysis.PhasePlane1D``
    - ``brainpy.analysis.PhasePlane2D``
    - ``brainpy.analysis.Bifurcation1D``
    - ``brainpy.analysis.Bifurcation2D``
    - ``brainpy.analysis.FastSlow1D``
    - ``brainpy.analysis.FastSlow2D``
- Numerical optimization based high-dimensional analyzer:
    - ``brainpy.analysis.SlowPointFinder``
- Dedicated operators in ``brainpy.math`` module:
    - ``brainpy.math.pre2post_event_sum``
    - ``brainpy.math.pre2post_sum``
    - ``brainpy.math.pre2post_prod``
    - ``brainpy.math.pre2post_max``
    - ``brainpy.math.pre2post_min``
    - ``brainpy.math.pre2syn``
    - ``brainpy.math.syn2post``
    - ``brainpy.math.syn2post_prod``
    - ``brainpy.math.syn2post_max``
    - ``brainpy.math.syn2post_min``
- Conversion APIs in ``brainpy.math`` module:
    - ``brainpy.math.as_device_array()``
    - ``brainpy.math.as_variable()``
    - ``brainpy.math.as_jaxarray()``
- New autograd APIs in ``brainpy.math`` module:
    - ``brainpy.math.vector_grad()``
- Simulation runners:
    - ``brainpy.ReportRunner``
    - ``brainpy.StructRunner``
    - ``brainpy.NumpyRunner``
- Commonly used models in ``brainpy.models`` module
    - ``brainpy.models.LIF``
    - ``brainpy.models.Izhikevich``
    - ``brainpy.models.AdExIF``
    - ``brainpy.models.SpikeTimeInput``
    - ``brainpy.models.PoissonInput``
    - ``brainpy.models.DeltaSynapse``
    - ``brainpy.models.ExpCUBA``
    - ``brainpy.models.ExpCOBA``
    - ``brainpy.models.AMPA``
    - ``brainpy.models.GABAa``
- Naming cache clean: ``brainpy.clear_name_cache``
- add safe in-place operations of ``update()`` method and ``.value``  assignment for JaxArray


Documentation
~~~~~~~~~~~~~

- Complete tutorials for quickstart
- Complete tutorials for dynamics building
- Complete tutorials for dynamics simulation
- Complete tutorials for dynamics training
- Complete tutorials for dynamics analysis
- Complete tutorials for API documentation


brainpy 1.1.x
*************


If you are using ``brainpy==1.x``, you can find *documentation*, *examples*, and *models* through the following links:

- **Documentation:** https://brainpy.readthedocs.io/en/brainpy-1.x/
- **Examples from papers**: https://brainpy-examples.readthedocs.io/en/brainpy-1.x/
- **Canonical brain models**: https://brainmodels.readthedocs.io/en/brainpy-1.x/


Version 1.1.7 (2021.12.13)
==========================

- fix bugs on ``numpy_array()`` conversion in `brainpy.math.utils` module


Version 1.1.5 (2021.11.17)
==========================

**API changes:**

- fix bugs on ndarray import in `brainpy.base.function.py`
- convenient 'get_param' interface `brainpy.simulation.layers`
- add more weight initialization methods

**Doc changes:**

- add more examples in README


Version 1.1.4
=============

**API changes:**

- add ``.struct_run()`` in DynamicalSystem
- add ``numpy_array()`` conversion in `brainpy.math.utils` module
- add ``Adagrad``, ``Adadelta``, ``RMSProp`` optimizers
- remove `setting` methods in `brainpy.math.jax` module
- remove import jax in `brainpy.__init__.py` and enable jax setting, including

  - ``enable_x64()``
  - ``set_platform()``
  - ``set_host_device_count()``
- enable ``b=None`` as no bias in `brainpy.simulation.layers`
- set `int_` and `float_` as default 32 bits
- remove ``dtype`` setting in Initializer constructor

**Doc changes:**

- add ``optimizer`` in "Math Foundation"
- add ``dynamics training`` docs
- improve others


Version 1.1.3
=============

- fix bugs of JAX parallel API imports
- fix bugs of `post_slice` structure construction
- update docs


Version 1.1.2
=============

- add ``pre2syn`` and ``syn2post`` operators
- add `verbose` and `check` option to ``Base.load_states()``
- fix bugs on JIT DynamicalSystem (numpy backend)


Version 1.1.1
=============

- fix bugs on symbolic analysis: model trajectory
- change `absolute` access in the variable saving and loading to the `relative` access
- add UnexpectedTracerError hints in JAX transformation functions


Version 1.1.0 (2021.11.08)
==========================

This package releases a new version of BrainPy.

Highlights of core changes:

``math`` module
~~~~~~~~~~~~~~~

- support numpy backend
- support JAX backend
- support ``jit``, ``vmap`` and ``pmap`` on class objects on JAX backend
- support ``grad``, ``jacobian``, ``hessian`` on class objects on JAX backend
- support ``make_loop``, ``make_while``, and ``make_cond`` on JAX backend
- support ``jit`` (based on numba) on class objects on numpy backend
- unified numpy-like ndarray operation APIs
- numpy-like random sampling APIs
- FFT functions
- gradient descent optimizers
- activation functions
- loss function
- backend settings


``base`` module
~~~~~~~~~~~~~~~

- ``Base`` for whole Version ecosystem
- ``Function`` to wrap functions
- ``Collector`` and ``TensorCollector`` to collect variables, integrators, nodes and others


``integrators`` module
~~~~~~~~~~~~~~~~~~~~~~

- class integrators for ODE numerical methods
- class integrators for SDE numerical methods

``simulation`` module
~~~~~~~~~~~~~~~~~~~~~

- support modular and composable programming
- support multi-scale modeling
- support large-scale modeling
- support simulation on GPUs
- fix bugs on ``firing_rate()``
- remove ``_i`` in ``update()`` function, replace ``_i`` with ``_dt``,
  meaning the dynamic system has the canonic equation form
  of :math:`dx/dt = f(x, t, dt)`
- reimplement the ``input_step`` and ``monitor_step`` in a more intuitive way
- support to set `dt`  in the single object level (i.e., single instance of DynamicSystem)
- common used DNN layers
- weight initializations
- refine synaptic connections


brainpy 1.0.x
*************

Version 1.0.3 (2021.08.18)
==========================

Fix bugs on

- firing rate measurement
- stability analysis


Version 1.0.2
=============

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
=============

- Fix bugs


Version 1.0.0
=============

- **NEW VERSION OF BRAINPY**
- Change the coding style into the object-oriented programming
- Systematically improve the documentation


brainpy 0.x
***********

Version 0.3.5
=============

- Add 'timeout' in sympy solver in neuron dynamics analysis
- Reconstruct and generalize phase plane analysis
- Generalize the repeat mode of ``Network`` to different running duration between two runs
- Update benchmarks
- Update detailed documentation


Version 0.3.1
=============

- Add a more flexible way for NeuState/SynState initialization
- Fix bugs of "is_multi_return"
- Add "hand_overs", "requires" and "satisfies".
- Update documentation
- Auto-transform `range` to `numba.prange`
- Support `_obj_i`, `_pre_i`, `_post_i` for more flexible operation in scalar-based models



Version 0.3.0
=============

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
=============

API changes
~~~~~~~~~~~

- Fix bugs


Version 0.2.3
=============

API changes
~~~~~~~~~~~

- Add "animate_1D" in ``visualization`` module
- Add "PoissonInput", "SpikeTimeInput" and "FreqInput" in ``inputs`` module
- Update phase_portrait_analyzer.py


Models and examples
~~~~~~~~~~~~~~~~~~~

- Add CANN examples


Version 0.2.2
=============

API changes
~~~~~~~~~~~

- Redesign visualization
- Redesign connectivity
- Update docs


Version 0.2.1
=============

API changes
~~~~~~~~~~~

- Fix bugs in `numba import`
- Fix bugs in `numpy` mode with `scalar` model


Version 0.2.0
=============

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

