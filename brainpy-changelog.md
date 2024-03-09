# Release notes (``brainpy``)


## brainpy>2.3.x


### Version 2.5.0


This release contains many new features and fixes. It is the first release with a mature solution for Brain Dynamics Operator Customization on both CPU and GPU platforms. 


#### New Features

1. Add synapse projection with Delta synapse models through ``brainpy.dyn.HalfProjDelta`` and ``brainpy.dyn.FullProjDelta``. 
2. Add ``brainpy.math.exprel``, and change the code in the corresponding HH neuron models to improve numerical computation accuracy. These changes can significantly improve the numerical integration accuracy of HH-like models under x32 computation. 
3. Add ``brainpy.reset_level()`` decorator so that the state resetting order can be customized by users. 
4. Add ``brainpy.math.ein_rearrange``, ``brainpy.math.ein_reduce``, and ``brainpy.math.ein_repeat`` functions 
5. Add ``brainpy.math.scan`` transformation.
6. Rebase all customized operators using Taichi JIT compiler. On the CPU platform, the speed performance can be boosted ten to hundred times. On the GPU platforms, the flexibility can be greatly improved.
7. Many bug fixes. 
8. A new version of ``brainpylib>=0.2.4`` has been released, supporting operator customization through the Taichi compiler. The supported backends include Linux, Windows, MacOS Intel, and MacOS M1 platforms. Tutorials please see https://brainpy.readthedocs.io/en/latest/tutorial_advanced/operator_custom_with_taichi.html

#### What's Changed
* [docs] Add taichi customized operators tutorial by @Routhleck in https://github.com/brainpy/BrainPy/pull/545
* [docs] Optimize tutorial code in `operator_custom_with_taichi.ipynb` of documentations by @Routhleck in https://github.com/brainpy/BrainPy/pull/546
* [running] fix multiprocessing bugs by @chaoming0625 in https://github.com/brainpy/BrainPy/pull/547
* [docs] Fix typo in docs by @Routhleck in https://github.com/brainpy/BrainPy/pull/549
* :arrow_up: Bump conda-incubator/setup-miniconda from 2 to 3 by @dependabot in https://github.com/brainpy/BrainPy/pull/551
* updates  by @chaoming0625 in https://github.com/brainpy/BrainPy/pull/550
* ``brainpy.math.defjvp`` and ``brainpy.math.XLACustomOp.defjvp`` by @chaoming0625 in https://github.com/brainpy/BrainPy/pull/554
* :arrow_up: Bump actions/setup-python from 4 to 5 by @dependabot in https://github.com/brainpy/BrainPy/pull/555
* Fix ``brainpy.math.ifelse`` bugs by @chaoming0625 in https://github.com/brainpy/BrainPy/pull/556
* [math & dyn] add ``brainpy.math.exprel``, and change the code in the corresponding HH neuron models to improve numerical computation accuracy by @chaoming0625 in https://github.com/brainpy/BrainPy/pull/557
* Update README by @chaoming0625 in https://github.com/brainpy/BrainPy/pull/558
* [doc] add conductance neuron model tutorial by @chaoming0625 in https://github.com/brainpy/BrainPy/pull/559
* Doc by @chaoming0625 in https://github.com/brainpy/BrainPy/pull/560
* add `brainpy.math.functional_vector_grad` and `brainpy.reset_level()` decorator by @chaoming0625 in https://github.com/brainpy/BrainPy/pull/561
* [math] change the internal implementation of surrogate function by @chaoming0625 in https://github.com/brainpy/BrainPy/pull/562
* Math by @chaoming0625 in https://github.com/brainpy/BrainPy/pull/563
* [doc] update citations by @chaoming0625 in https://github.com/brainpy/BrainPy/pull/564
* add support for multi-class margin loss by @charlielam0615 in https://github.com/brainpy/BrainPy/pull/566
* Support for Delta synapse projections by @chaoming0625 in https://github.com/brainpy/BrainPy/pull/568
* [math] Add taichi customized operators(event csrmv, csrmv, jitconn event mv, jitconn mv) by @Routhleck in https://github.com/brainpy/BrainPy/pull/553
* fix doc by @chaoming0625 in https://github.com/brainpy/BrainPy/pull/571
* Fix default math parameter setting bug by @chaoming0625 in https://github.com/brainpy/BrainPy/pull/572
* fix bugs in `brainpy.math.random.truncated_normal` by @chaoming0625 in https://github.com/brainpy/BrainPy/pull/574
* [doc] fix doc by @chaoming0625 in https://github.com/brainpy/BrainPy/pull/576
* fix bugs in  truncated_normal; add TruncatedNormal init. by @charlielam0615 in https://github.com/brainpy/BrainPy/pull/575
* [Dyn] Fix alpha synapse bugs by @ztqakita in https://github.com/brainpy/BrainPy/pull/578
* fix `brainpy.math.softplus` and `brainpy.dnn.SoftPlus` by @chaoming0625 in https://github.com/brainpy/BrainPy/pull/581
* add `TruncatedNormal` to `initialize.py` by @charlielam0615 in https://github.com/brainpy/BrainPy/pull/583
* Fix `_format_shape` in `random_inits.py` by @charlielam0615 in https://github.com/brainpy/BrainPy/pull/584
* fix bugs in `truncated_normal` by @charlielam0615 in https://github.com/brainpy/BrainPy/pull/585
* [dyn] fix warning of reset_state by @chaoming0625 in https://github.com/brainpy/BrainPy/pull/587
* [math] upgrade variable retrival by @chaoming0625 in https://github.com/brainpy/BrainPy/pull/589
* [math & dnn] add `brainpy.math.unflatten` and `brainpy.dnn.Unflatten` by @chaoming0625 in https://github.com/brainpy/BrainPy/pull/588
* [math] add ``ein_rearrange``, ``ein_reduce``, and ``ein_repeat`` functions by @chaoming0625 in https://github.com/brainpy/BrainPy/pull/590
* [math] Support taichi customized op with metal cpu backend by @Routhleck in https://github.com/brainpy/BrainPy/pull/579
* Doc fix and standardize Dual Exponential model again by @chaoming0625 in https://github.com/brainpy/BrainPy/pull/591
* update doc, upgrade reset_state, update projection models by @chaoming0625 in https://github.com/brainpy/BrainPy/pull/592
* [taichi] Make taichi caches more transparent and Add clean caches function by @Routhleck in https://github.com/brainpy/BrainPy/pull/596
* [test] remove test skip on macos, since brainpylib supports taichi interface on macos by @chaoming0625 in https://github.com/brainpy/BrainPy/pull/597
* [dyn] add `clear_input` in the `step_run` function. by @chaoming0625 in https://github.com/brainpy/BrainPy/pull/601
* [math] Refactor taichi operators by @Routhleck in https://github.com/brainpy/BrainPy/pull/598
* [math] fix `brainpy.math.scan` by @chaoming0625 in https://github.com/brainpy/BrainPy/pull/604
* ``disable_ jit`` support in ``brainpy.math.scan`` by @chaoming0625 in https://github.com/brainpy/BrainPy/pull/606
* [math] Remove the logs that `taichi.init()` print by @Routhleck in https://github.com/brainpy/BrainPy/pull/609
* Version control in Publish.yml CI by @chaoming0625 in https://github.com/brainpy/BrainPy/pull/610

#### New Contributors
* @charlielam0615 made their first contribution in https://github.com/brainpy/BrainPy/pull/566

**Full Changelog**: https://github.com/brainpy/BrainPy/compare/V2.4.6...V2.5.0




### Version 2.4.6

This release contains more than 130 commit updates, and has provided several new features. 


#### New Features


##### 1. surrogate gradient functions are more transparent. 

New instances can be used to compute the surrogate gradients. For example:

```python
import brainpy.math as bm
fun = bm.surrogate.Sigmoid()

# forward function
spk = fun(membrane_potential)

# backward function
dV = fun.surrogate_grad(1., membrane_potential)

# surrogate forward function
surro_spk = fun.surrogate_fun(membrane_potential)
```

##### 2. Add ``brainpy.math.eval_shape`` for evaluating the all dynamical variables used in the target function. 

This function is similar to ``jax.eval_shape`` which has no FLOPs, while it can extract all variables used in the target function. For example:

```python
net = ...  # any dynamical system
inputs = ...  # inputs to the dynamical system
variables, outputs= bm.eval_shape(net, inputs)  
# "variables" are all variables used in the target "net"
```

In future, this function will be used everywhere to transform all jax transformations into brainpy's oo transformations. 

##### 3. Generalize tools and interfaces for state managements. 

For a single object:
- The ``.reset_state()`` defines the state resetting of all local variables in this node.
- The ``.load_state()`` defines the state loading from external disks (typically, a dict is passed into this ``.load_state()`` function).
- The ``.save_state()`` defines the state saving to external disks (typically, the ``.save_state()`` function generates a dict containing all variable values).

Here is an example to define a full class of ``brainpy.DynamicalSystem``.

```python
import brainpy as bp

class YouDynSys(bp.DynamicalSystem):
   def __init__(self, ):  # define parameters
      self.par1 = ....
      self.num = ...

  def reset_state(self, batch_or_mode=None):  # define variables
     self.a = bp.init.variable_(bm.zeros, (self.num,), batch_or_mode)

  def load_state(self, state_dict):  # load states from an external dict
     self.a.value = bm.as_jax(state_dict['a'])

  def save_state(self):  # save states as an external dict
     return {'a': self.a.value}
```


For a complex network model, brainpy provide unified state managment interface for initializing, saving, and loading states. 
- The ``brainpy.reset_state()`` defines the state resetting of all variables in this node and its children nodes.
- The ``brainpy.load_state()`` defines the state loading from external disks of all variables in the node and its children.
- The ``brainpy.save_state()`` defines the state saving to external disks of all variables in the node and its children.
- The ``brainpy.clear_input()`` defines the clearing of all input variables in the node and its children.




##### 4. Unified brain simulation and brain-inspired computing interface through automatic membrane scaling. 

The same model used in brain simulation can be easily transformed into the one used for brain-inspired computing for training. For example,


```python
class EINet(bp.DynSysGroup):
  def __init__(self):
    super().__init__()
    self.N = bp.dyn.LifRefLTC(4000, V_rest=-60., V_th=-50., V_reset=-60., tau=20., tau_ref=5.,
                              V_initializer=bp.init.Normal(-55., 2.))
    self.delay = bp.VarDelay(self.N.spike, entries={'I': None})
    self.E = bp.dyn.ProjAlignPost1(
      comm=bp.dnn.EventCSRLinear(bp.conn.FixedProb(0.02, pre=3200, post=4000), weight=bp.init.Normal(0.6, 0.01)),
      syn=bp.dyn.Expon(size=4000, tau=5.),
      out=bp.dyn.COBA(E=0.),
      post=self.N
    )
    self.I = bp.dyn.ProjAlignPost1(
      comm=bp.dnn.EventCSRLinear(bp.conn.FixedProb(0.02, pre=800, post=4000), weight=bp.init.Normal(6.7, 0.01)),
      syn=bp.dyn.Expon(size=4000, tau=10.),
      out=bp.dyn.COBA(E=-80.),
      post=self.N
    )

  def update(self, input):
    spk = self.delay.at('I')
    self.E(spk[:3200])
    self.I(spk[3200:])
    self.delay(self.N(input))
    return self.N.spike.value


# used for brain simulation
with bm.environment(mode=bm.nonbatching_mode):
  net = EINet()


# used for brain-inspired computing
# define the `membrane_scaling` parameter
with bm.environment(mode=bm.TrainingMode(128), membrane_scaling=bm.Scaling.transform([-60., -50.])):
  net = EINet()
```



##### 5. New apis for operator customization on CPU and GPU devices through ``brainpy.math.XLACustomOp``. 

Starting from this release, brainpy introduces [Taichi](https://github.com/taichi-dev/taichi) for operator customization. Now, users can write CPU and GPU operators through numba and taichi syntax on CPU device, and taichi syntax on GPu device. Particularly, to define an operator, user can use:

```python

import numba as nb
import taichi as ti
import numpy as np
import jax
import brainpy.math as bm


@nb.njit
def numba_cpu_fun(a, b, out_a, out_b):
  out_a[:] = a
  out_b[:] = b


@ti.kernel
def taichi_gpu_fun(a, b, out_a, out_b):
  for i in range(a.size):
    out_a[i] = a[i]
  for i in range(b.size):
    out_b[i] = b[i]


prim = bm.XLACustomOp(cpu_kernel=numba_cpu_fun, gpu_kernel=taichi_gpu_fun)
a2, b2 = prim(np.random.random(1000), np.random.random(1000),
              outs=[jax.ShapeDtypeStruct(1000, dtype=np.float32),
                    jax.ShapeDtypeStruct(1000, dtype=np.float32)])

```

##### 6. Generalized STDP models which are compatible with diverse synapse models.

See https://github.com/brainpy/BrainPy/blob/master/brainpy/_src/dyn/projections/tests/test_STDP.py


#### What's Changed
* [bug] fix compatible bug by @chaoming0625 in https://github.com/brainpy/BrainPy/pull/508
* [docs] add low-level op customization by @ztqakita in https://github.com/brainpy/BrainPy/pull/507
* Compatible with `jax==0.4.16` by @chaoming0625 in https://github.com/brainpy/BrainPy/pull/511
* updates for parallelization support by @chaoming0625 in https://github.com/brainpy/BrainPy/pull/514
* Upgrade surrogate gradient functions by @chaoming0625 in https://github.com/brainpy/BrainPy/pull/516
* [doc] update operator customization by @chaoming0625 in https://github.com/brainpy/BrainPy/pull/517
* Updates for OO transforma and surrogate functions by @chaoming0625 in https://github.com/brainpy/BrainPy/pull/519
* [dyn] add neuron scaling by @ztqakita in https://github.com/brainpy/BrainPy/pull/520
* State saving, loading, and resetting by @chaoming0625 in https://github.com/brainpy/BrainPy/pull/521
* [delay] rewrite previous delay APIs so that they are compatible with new brainpy version by @chaoming0625 in https://github.com/brainpy/BrainPy/pull/522
* [projection] upgrade projections so that APIs are reused across different models by @chaoming0625 in https://github.com/brainpy/BrainPy/pull/523
* [math] the interface for operator registration by @chaoming0625 in https://github.com/brainpy/BrainPy/pull/524
* FIx bug in Delay by @ztqakita in https://github.com/brainpy/BrainPy/pull/525
* Fix bugs in membrane scaling by @ztqakita in https://github.com/brainpy/BrainPy/pull/526
* [math] Implement taichi op register by @Routhleck in https://github.com/brainpy/BrainPy/pull/527
* Link libtaichi_c_api.so when import brainpylib by @Routhleck in https://github.com/brainpy/BrainPy/pull/528
* update taichi op customization by @chaoming0625 in https://github.com/brainpy/BrainPy/pull/529
* Fix error message by @HoshinoKoji in https://github.com/brainpy/BrainPy/pull/530
* [math] remove the hard requirement of `taichi` by @chaoming0625 in https://github.com/brainpy/BrainPy/pull/531
* [math] Resolve encoding of source kernel when ti.func is nested in ti… by @Routhleck in https://github.com/brainpy/BrainPy/pull/532
* [math] new abstract function for XLACustomOp, fix its bugs by @chaoming0625 in https://github.com/brainpy/BrainPy/pull/534
* [math] fix numpy array priority by @chaoming0625 in https://github.com/brainpy/BrainPy/pull/533
* [brainpy.share] add category shared info by @chaoming0625 in https://github.com/brainpy/BrainPy/pull/535
* [doc] update documentations by @chaoming0625 in https://github.com/brainpy/BrainPy/pull/536
* [doc] update doc by @chaoming0625 in https://github.com/brainpy/BrainPy/pull/537
* [dyn] add `brainpy.reset_state()` and `brainpy.clear_input()` for more consistent and flexible state managements by @chaoming0625 in https://github.com/brainpy/BrainPy/pull/538
* [math] simplify the taichi AOT operator customization interface by @chaoming0625 in https://github.com/brainpy/BrainPy/pull/540
* [dyn] add `save_state`, `load_state`, `reset_state`, and `clear_input` helpers by @chaoming0625 in https://github.com/brainpy/BrainPy/pull/542
* [dyn] update STDP APIs on CPUs and fix bugs by @chaoming0625 in https://github.com/brainpy/BrainPy/pull/543

#### New Contributors
* @HoshinoKoji made their first contribution in https://github.com/brainpy/BrainPy/pull/530

**Full Changelog**: https://github.com/brainpy/BrainPy/compare/V2.4.5...V2.4.6





### Version 2.4.5


#### New Features

- A new version of ``brainpylib==0.1.10`` has been released. In this release, we have fixed some bugs of brainpy dedicated GPU operators. Users can freely use them in any application.
- Correspondingly, dedicated operators in ``brainpy.math`` have been refined. 
- ``.tracing_variable()`` has been created to support tracing ``Variable``s during computations and compilations. Example usage please see #472
- Add a new random API for creating multiple random keys: ``brainpy.math.random.split_keys()``. 
- Fix bugs, including
  - ``brainpy.dnn.AllToAll`` module
  - RandomState.
  - ``brainpy.math.cond`` and ``brainpy.math.while_loop`` when variables are used in both branches

#### What's Changed
* Creat random key automatically when it is detected by @chaoming0625 in https://github.com/brainpy/BrainPy/pull/461
* [encoding] upgrade encoding methods by @chaoming0625 in https://github.com/brainpy/BrainPy/pull/464
* fix #466 by @chaoming0625 in https://github.com/brainpy/BrainPy/pull/467
* Update operators for compatible with ``brainpylib>=0.1.10`` by @chaoming0625 in https://github.com/brainpy/BrainPy/pull/468
* Support tracing ``Variable`` during computation and compilation by using ``tracing_variable()`` function by @chaoming0625 in https://github.com/brainpy/BrainPy/pull/472
* Add code of conduct and contributing guides by @chaoming0625 in https://github.com/brainpy/BrainPy/pull/473
* add Funding and Development roadmap by @chaoming0625 in https://github.com/brainpy/BrainPy/pull/475
* Create SECURITY.md by @chaoming0625 in https://github.com/brainpy/BrainPy/pull/474
* Create dependabot.yml by @chaoming0625 in https://github.com/brainpy/BrainPy/pull/476
* update maintainence info in README by @chaoming0625 in https://github.com/brainpy/BrainPy/pull/479
* :arrow_up: Bump actions/setup-python from 2 to 4 by @dependabot in https://github.com/brainpy/BrainPy/pull/477
* :arrow_up: Bump actions/checkout from 2 to 4 by @dependabot in https://github.com/brainpy/BrainPy/pull/478
* ad acknowledgment.md by @chaoming0625 in https://github.com/brainpy/BrainPy/pull/482
* update quickstart of `simulating a brain dynamics model` with new APIs by @chaoming0625 in https://github.com/brainpy/BrainPy/pull/483
* update advanced tutorials by @chaoming0625 in https://github.com/brainpy/BrainPy/pull/484
* [docs] Update installation.rst by @Routhleck in https://github.com/brainpy/BrainPy/pull/485
* update requirements by @chaoming0625 in https://github.com/brainpy/BrainPy/pull/486
* [doc] update docs by @chaoming0625 in https://github.com/brainpy/BrainPy/pull/487
* [doc] update docs by @chaoming0625 in https://github.com/brainpy/BrainPy/pull/488
* Decouple Online and Offline training algorithms as ``brainpy.mixin.SupportOnline`` and `brainpy.mixin.SupportOffline` by @chaoming0625 in https://github.com/brainpy/BrainPy/pull/489
* [dyn] add STDP_Song2000 LTP model by @ztqakita in https://github.com/brainpy/BrainPy/pull/481
* update STDP by @chaoming0625 in https://github.com/brainpy/BrainPy/pull/491
* [doc] update the API of `brainpy.dyn` module & add synaptic plasticity module by @chaoming0625 in https://github.com/brainpy/BrainPy/pull/492
* fix bug by @chaoming0625 in https://github.com/brainpy/BrainPy/pull/493
* [math] fix bugs in `cond` and `while_loop` when same variables are used in both branches by @chaoming0625 in https://github.com/brainpy/BrainPy/pull/494
* [docs] add BrainPy docker and docs by @ztqakita in https://github.com/brainpy/BrainPy/pull/496
* [docs] update README and installation by @ztqakita in https://github.com/brainpy/BrainPy/pull/499
* :arrow_up: Bump docker/build-push-action from 4 to 5 by @dependabot in https://github.com/brainpy/BrainPy/pull/498
* :arrow_up: Bump docker/login-action from 2 to 3 by @dependabot in https://github.com/brainpy/BrainPy/pull/497
* Add strings in bp._src.dyn.bio_models and abstract_models by @AkitsuFaye in https://github.com/brainpy/BrainPy/pull/500
* [reset] update logics of state reset in `DynamicalSystem` by @chaoming0625 in https://github.com/brainpy/BrainPy/pull/501
* [doc] upgrade docs with the latest APIs, fix #463 by @chaoming0625 in https://github.com/brainpy/BrainPy/pull/502
* [doc] add synapse model documentations by @chaoming0625 in https://github.com/brainpy/BrainPy/pull/503
* Changed the order of code blocks in the docs of hh models and lif models by @AkitsuFaye in https://github.com/brainpy/BrainPy/pull/505
* [mode] move recurrent models in brainpy.dnn model into `brainpy.dyn` module by @chaoming0625 in https://github.com/brainpy/BrainPy/pull/506

#### New Contributors
* @dependabot made their first contribution in https://github.com/brainpy/BrainPy/pull/477

**Full Changelog**: https://github.com/brainpy/BrainPy/compare/V2.4.4...V2.4.5






### Version 2.4.4



This release has fixed several bugs and updated the sustainable documentation.

#### What's Changed
* [mixin] abstract the behavior of supporting input projection by ``brainpy.mixin.ReceiveInputProj`` by @chaoming0625 in https://github.com/brainpy/BrainPy/pull/428
* Update delays, models, and projections by @chaoming0625 in https://github.com/brainpy/BrainPy/pull/429
* Compatible with `jax=0.4.14` by @chaoming0625 in https://github.com/brainpy/BrainPy/pull/431
* Add new tests by @yygf123 in https://github.com/brainpy/BrainPy/pull/430
* Add NonBatchingMode function by @yygf123 in https://github.com/brainpy/BrainPy/pull/433
* [connect] Complete `FixedTotalNum` class and fix bugs by @Routhleck in https://github.com/brainpy/BrainPy/pull/434
* Update the document "Concept 2: Dynamical System" by @yygf123 in https://github.com/brainpy/BrainPy/pull/435
* [docs] Update three part of tutorial toolbox by @Routhleck in https://github.com/brainpy/BrainPy/pull/436
* [docs] Update index.rst for surrogate gradient by @Routhleck in https://github.com/brainpy/BrainPy/pull/437
* Reconstruct BrainPy documentations by @ztqakita in https://github.com/brainpy/BrainPy/pull/438
* Renew doc requirements.txt by @ztqakita in https://github.com/brainpy/BrainPy/pull/441
* Compatibility updates by @chaoming0625 in https://github.com/brainpy/BrainPy/pull/442
* update docs by @chaoming0625 in https://github.com/brainpy/BrainPy/pull/443
* Update optimizer by @yygf123 in https://github.com/brainpy/BrainPy/pull/451
* [docs] Update custom saving and loading by @Routhleck in https://github.com/brainpy/BrainPy/pull/439
* [doc] add new strings in bp._src.dyn.hh.py and bp._src.dyn.lif.py by @AkitsuFaye in https://github.com/brainpy/BrainPy/pull/454
* Serveral updates by @chaoming0625 in https://github.com/brainpy/BrainPy/pull/452
* Update doc bug in index.rst by @chaoming0625 in https://github.com/brainpy/BrainPy/pull/458
* add `brainpy.dyn.Alpha` synapse model by @chaoming0625 in https://github.com/brainpy/BrainPy/pull/459
* [doc] update ODE doc by @chaoming0625 in https://github.com/brainpy/BrainPy/pull/460

#### New Contributors
* @AkitsuFaye made their first contribution in https://github.com/brainpy/BrainPy/pull/454

**Full Changelog**: https://github.com/brainpy/BrainPy/compare/V2.4.3...V2.4.4





### Version 2.4.3


This release has standardized the modeling of DNN and SNN models by two intercorrelated packages: ``brainpy.dnn`` and ``brainpy.dyn``.

Overall, the modeling of brain dynamics in this release has the following advantages:

- the automatic merging of the duplicate synapses, keeping the minimal device memory
- easy model and data parallelization across multiple devices 
- easy integration with artificial neural networks
- a new abstraction that decouples dynamics from communication
- the unified ``DynamicalSystem`` interface

#### New Features

1. Support to define ion channel models which rely on multiple ions. For example, 

```python

class HH(bp.dyn.CondNeuGroup):
   def __init__(self, size):
      super().__init__(size)
      self.k = bp.dyn.PotassiumFixed(size)
      self.ca = bp.dyn.CalciumFirstOrder(size)

      self.kca = bp.dyn.mix_ions(self.k, self.ca)  # Ion that mixing Potassium and Calcium
      self.kca.add_elem(ahp=bp.dyn.IAHP_De1994v2(size))  # channel that relies on both Potassium and Calcium

```

2. New style ``.update()`` function in ``brainpy.DynamicalSystem`` which resolves all compatible issues. Starting from this version, all ``update()`` no longer needs to receive a global shared argument such as ``tdi``. 

```python

class YourDynSys(bp.DynamicalSystem):
  def update(self, x):
    t = bp.share['t']
    dt = bp.share['dt']
    i = bp.share['i']
    ...

```

3. Optimize the connection-building process when using ``brainpy.conn.ScaleFreeBA``, ``brainpy.conn.ScaleFreeBADual``, ``brainpy.conn.PowerLaw`` 

4. New dual exponential model ``brainpy.dyn.DualExponV2`` can be aligned with post dimension. 

5. More synaptic projection abstractions, including
   - ``brainpy.dyn.VanillaProj``
   - ``brainpy.dyn.ProjAlignPostMg1``
   - ``brainpy.dyn.ProjAlignPostMg2``
   - ``brainpy.dyn.ProjAlignPost1``
   - ``brainpy.dyn.ProjAlignPost2``
   - ``brainpy.dyn.ProjAlignPreMg1``
   - ``brainpy.dyn.ProjAlignPreMg2``

5. Fix compatible issues, fix unexpected bugs, and improve the model tests.



#### What's Changed
* [connect] Optimize the connector about ScaleFreeBA, ScaleFreeBADual, PowerLaw by @Routhleck in https://github.com/brainpy/BrainPy/pull/412
* [fix] bug of `connect.base.py`'s `require` function by @Routhleck in https://github.com/brainpy/BrainPy/pull/413
* Many Updates by @chaoming0625 in https://github.com/brainpy/BrainPy/pull/414
* Update docs by @chaoming0625 in https://github.com/brainpy/BrainPy/pull/415
* fix conflict by @yygf123 in https://github.com/brainpy/BrainPy/pull/416
* add a new implementation of Dual Exponential Synapse model which can be aligned post. by @chaoming0625 in https://github.com/brainpy/BrainPy/pull/417
* Enable test when pull requests by @chaoming0625 in https://github.com/brainpy/BrainPy/pull/418
* Add random.seed() by @yygf123 in https://github.com/brainpy/BrainPy/pull/419
* Remove windows CI because it always generates strange errors by @chaoming0625 in https://github.com/brainpy/BrainPy/pull/420
* Recent updates by @chaoming0625 in https://github.com/brainpy/BrainPy/pull/421
* upgrade Runner and Trainer for new style of ``DynamicalSystem.update()`` function   by @chaoming0625 in https://github.com/brainpy/BrainPy/pull/422
* update docs by @chaoming0625 in https://github.com/brainpy/BrainPy/pull/424
* fix ``lif`` model bugs and support two kinds of spike reset: ``soft`` and ``hard`` by @chaoming0625 in https://github.com/brainpy/BrainPy/pull/423
* rewrite old synapses with decomposed components by @chaoming0625 in https://github.com/brainpy/BrainPy/pull/425
* fix autograd bugs by @chaoming0625 in https://github.com/brainpy/BrainPy/pull/426

#### New Contributors
* @yygf123 made their first contribution in https://github.com/brainpy/BrainPy/pull/416

**Full Changelog**: https://github.com/brainpy/BrainPy/compare/V2.4.2...V2.4.3








### Version 2.4.2



We are very excited to release this new version of BrainPy V2.4.2. In this new update, we cover several exciting features:
#### New Features
* Reorganize the model to decouple dynamics and communication.
* Add `brainpy.dyn` for dynamics models and `brainpy.dnn` for the ANN layer and connection structures. 
* Supplement many docs for dedicated operators and common bugs of BrainPy.
* Fix many bugs.

#### What's Changed 
* [ANN] add more activation functions by @chaoming0625 in https://github.com/brainpy/BrainPy/pull/379
* Optimize Gaussian Decay initializer by @Routhleck in https://github.com/brainpy/BrainPy/pull/381
* [update] new loss functions, surrograte base class, Array built-in functions by @chaoming0625 in https://github.com/brainpy/BrainPy/pull/383
* [parallelization] new module of ``brainpy.pnn`` for auto parallelization of brain models by @chaoming0625 in https://github.com/brainpy/BrainPy/pull/385
* [fix] fix the bug of loading states by @chaoming0625 in https://github.com/brainpy/BrainPy/pull/388
* [math] support `jax.disable_jit()` for debugging by @chaoming0625 in https://github.com/brainpy/BrainPy/pull/389
* [initialize] speed up ``brainpy.init.DOGDecay`` by @chaoming0625 in https://github.com/brainpy/BrainPy/pull/390
* [doc] fix doc build by @chaoming0625 in https://github.com/brainpy/BrainPy/pull/391
* Add deprecations for deprecated APIs or functions by @chaoming0625 in https://github.com/brainpy/BrainPy/pull/393
* [math] enable debugging for new style of transformations in BrainPy by @chaoming0625 in https://github.com/brainpy/BrainPy/pull/392
* [math] flow control updates by @chaoming0625 in https://github.com/brainpy/BrainPy/pull/396
* Test of rates by @shangyangli in https://github.com/brainpy/BrainPy/pull/386
* Add math docs: NumPy-like operations and Dedicated operators by @c-xy17 in https://github.com/brainpy/BrainPy/pull/395
* [doc] documentation about ``how to debug`` and ``common gotchas`` by @chaoming0625 in https://github.com/brainpy/BrainPy/pull/397
* Update requirements-doc.txt by @chaoming0625 in https://github.com/brainpy/BrainPy/pull/399
* debug (images not displayed) by @c-xy17 in https://github.com/brainpy/BrainPy/pull/400
* Decouple dynamics and comminucations by @chaoming0625 in https://github.com/brainpy/BrainPy/pull/401
* [fix] bugs of control flows by @chaoming0625 in https://github.com/brainpy/BrainPy/pull/404
* Test for channels, neurons and synapses. by @ztqakita in https://github.com/brainpy/BrainPy/pull/403
* Implement function to visualize connection matrix  by @Routhleck in https://github.com/brainpy/BrainPy/pull/405
* Optimize GaussianProb  by @Routhleck in https://github.com/brainpy/BrainPy/pull/406
* [dyn] add reduce models, HH-type models and channels by @ztqakita in https://github.com/brainpy/BrainPy/pull/408
* [dnn] add various linear layers by @chaoming0625 in https://github.com/brainpy/BrainPy/pull/407
* [delay] `VariableDelay` and `DataDelay` by @chaoming0625 in https://github.com/brainpy/BrainPy/pull/409
* [dyn] add COBA examples using the interface of new `brainpy.dyn` module by @chaoming0625 in https://github.com/brainpy/BrainPy/pull/410
* [dyn] Update dyn.neurons docs and fix several bugs by @ztqakita in https://github.com/brainpy/BrainPy/pull/411

#### New Contributors
* @shangyangli made their first contribution in https://github.com/brainpy/BrainPy/pull/386

**Full Changelog**: https://github.com/brainpy/BrainPy/compare/V2.4.1...V2.4.2




### Version 2.4.1


#### New Features

1. [math] Support the error report when modifying a `brainpy.math.Array` during compilation
2. [math] add `brainpy.math.event`, `brainpy.math.sparse` and `brainpy.math.jitconn` module, needs ``brainpylib >= 0.1.9``
3. [interoperation] add apis and docs for `brainpy.layers.FromFlax` and `brainpy.layer.ToFlaxRNNCell`
4. [fix] Bug fixes:
   - fix WilsonCowan bug
   - fix `brainpy.connect.FixedProb` bug
   - fix analysis jit bug



#### What's Changed
* Update structures by @chaoming0625 in https://github.com/brainpy/BrainPy/pull/364
* create blocksparse matrix matrix multiplication opearator by @Routhleck in https://github.com/brainpy/BrainPy/pull/365
* commit by @grysgreat in https://github.com/brainpy/BrainPy/pull/367
* Fix bugs by @chaoming0625 in https://github.com/brainpy/BrainPy/pull/368
* [math] update dedicated operators by @chaoming0625 in https://github.com/brainpy/BrainPy/pull/370
* fix bugs by @chaoming0625 in https://github.com/brainpy/BrainPy/pull/371
* [bug] fix merging bug by @chaoming0625 in https://github.com/brainpy/BrainPy/pull/372
* [structure] update package structure by @chaoming0625 in https://github.com/brainpy/BrainPy/pull/369
* [test] update csrmv tests by @chaoming0625 in https://github.com/brainpy/BrainPy/pull/373
* [interoperation] add apis and docs for `brainpy.layers.FromFlax` and `brainpy.layer.ToFlaxRNNCell` by @chaoming0625 in https://github.com/brainpy/BrainPy/pull/374
* [doc] update documentation by @chaoming0625 in https://github.com/brainpy/BrainPy/pull/375
* [bug] fix `brainpy.connect.FixedProb` bug by @chaoming0625 in https://github.com/brainpy/BrainPy/pull/376
* [bug] fix analysis jit bug by @chaoming0625 in https://github.com/brainpy/BrainPy/pull/377
* update brainpylib requirements by @chaoming0625 in https://github.com/brainpy/BrainPy/pull/378

#### New Contributors
* @Routhleck made their first contribution in https://github.com/brainpy/BrainPy/pull/365
* @grysgreat made their first contribution in https://github.com/brainpy/BrainPy/pull/367

**Full Changelog**: https://github.com/brainpy/BrainPy/compare/V2.4.0...V2.4.1





### Version 2.4.0

This branch of releases (``brainpy==2.4.x``) are going to support the large-scale modeling for brain dynamics. 

As the start, this release provides support for automatic object-oriented (OO) transformations. 


#### What's New


1. Automatic OO transformations on longer need to take ``dyn_vars`` or ``child_objs`` information.
   These transformations are capable of automatically inferring the underlying dynamical variables. 
   Specifically, they include:
   
   - ``brainpy.math.grad`` and other autograd functionalities
   - ``brainpy.math.jit``
   - ``brainpy.math.for_loop``
   - ``brainpy.math.while_loop``
   - ``brainpy.math.ifelse``
   - ``brainpy.math.cond``

2. Update documentation
3. Fix several bugs

#### What's Changed
* reorganize operators in `brainpy.math` by @chaoming0625 in https://github.com/brainpy/BrainPy/pull/357
* Automatic transformations any function/object using `brainpy.math.Variable` by @chaoming0625 in https://github.com/brainpy/BrainPy/pull/358
* New OO transforms support ``jax.disable_jit`` mode by @chaoming0625 in https://github.com/brainpy/BrainPy/pull/359
* [oo transform] Enable new style of jit transformation to support `static_argnums` and `static_argnames` by @chaoming0625 in https://github.com/brainpy/BrainPy/pull/360
* [documentation] update documentation to brainpy>=2.4.0 by @chaoming0625 in https://github.com/brainpy/BrainPy/pull/361


**Full Changelog**: https://github.com/brainpy/BrainPy/compare/V2.3.8...V2.4.0







### Version 2.3.8


This release continues to add support for improving the usability of BrainPy.


#### New Features


1. New data structures for object-oriented transformations. 
   - ``NodeList`` and ``NodeDict`` for a list/tuple/dict of ``BrainPyObject`` instances.
   - ``ListVar`` and ``DictVar`` for a list/tuple/dict of brainpy data.
2. `Clip` transformation for brainpy initializers.
3. All ``brainpylib`` operators are accessible in ``brainpy.math`` module. Especially there are some dedicated operators for scaling up the million-level neuron networks. For an example, see example in [Simulating 1-million-neuron networks with 1GB GPU memory](https://brainpy-examples.readthedocs.io/en/latest/large_scale_modeling/EI_net_with_1m_neurons.html)
5. Enable monitoring GPU models on CPU when setting ``DSRunner(..., memory_efficient=True)``. This setting can usually reduce so much memory usage.   
6. ``brainpylib`` wheels on the Linux platform support the GPU operators. Users can install GPU version of ``brainpylib`` (require ``brainpylib>=0.1.7``) directly by ``pip install brainpylib``. @ztqakita

#### What's Changed
* Fix bugs and add more variable structures: `ListVar` and `DictVar` by @chaoming0625 in https://github.com/brainpy/BrainPy/pull/345
* add CI for testing various models by @chaoming0625 in https://github.com/brainpy/BrainPy/pull/346
* Update docs and tests by @chaoming0625 in https://github.com/brainpy/BrainPy/pull/347
* Fix `Runner(jit=False)`` bug by @chaoming0625 in https://github.com/brainpy/BrainPy/pull/348
* Compatible with jax>=0.4.7 by @chaoming0625 in https://github.com/brainpy/BrainPy/pull/349
* Updates by @chaoming0625 in https://github.com/brainpy/BrainPy/pull/350
* reconstruct BrainPy by merging brainpylib by @ztqakita in https://github.com/brainpy/BrainPy/pull/351
* Intergate brainpylib operators into brainpy by @chaoming0625 in https://github.com/brainpy/BrainPy/pull/352
* fix `brainpylib` call bug by @chaoming0625 in https://github.com/brainpy/BrainPy/pull/354
* Enable memory-efficient ``DSRunner`` by @chaoming0625 in https://github.com/brainpy/BrainPy/pull/355
* fix `Array` transform bug by @chaoming0625 in https://github.com/brainpy/BrainPy/pull/356


**Full Changelog**: https://github.com/brainpy/BrainPy/compare/V2.3.7...V2.3.8



### Version 2.3.7

- Fix bugs on population models in ``brainpy.rate`` module
- Fix bug on ``brainpy.LoopOverTime`` 
- Add more synaptic models including DualExpoenetial model and Alpha model in ``brainpy.experimental`` module
- Support call a module through right shift, such as ``data >> module1 >> module2`` 


### Version 2.3.6

This release continues to add support for brain-inspired computation.


#### New Features

##### More flexible customization of surrogate gradient functions. 

- brainpy.math.surrogate.Sigmoid
- brainpy.math.surrogate.PiecewiseQuadratic
- brainpy.math.surrogate.PiecewiseExp
- brainpy.math.surrogate.SoftSign
- brainpy.math.surrogate.Arctan
- brainpy.math.surrogate.NonzeroSignLog
- brainpy.math.surrogate.ERF
- brainpy.math.surrogate.PiecewiseLeakyRelu
- brainpy.math.surrogate.SquarewaveFourierSeries
- brainpy.math.surrogate.S2NN
- brainpy.math.surrogate.QPseudoSpike
- brainpy.math.surrogate.LeakyRelu
- brainpy.math.surrogate.LogTailedRelu
- brainpy.math.surrogate.ReluGrad
- brainpy.math.surrogate.GaussianGrad
- brainpy.math.surrogate.InvSquareGrad
- brainpy.math.surrogate.MultiGaussianGrad
- brainpy.math.surrogate.SlayerGrad

##### Fix bugs

- ``brainpy.LoopOverTime``





### Version 2.3.5



This release continues to add support for brain-inspired computation.


#### New Features


##### 1. ``brainpy.share`` for sharing data across submodules

In this release, we abstract the shared data as a ``brainpy.share`` object. 

This object together with ``brainpy.Delay`` we will introduce below constitutes the support that enables us to define SNN models like ANN ones.


##### 2. ``brainpy.Delay`` for delay processing

``Delay`` is abstracted as a dynamical system, which can be updated/retrieved by users. 

```python
import brainpy as bp

class EINet(bp.DynamicalSystemNS):
  def __init__(self, scale=1.0, e_input=20., i_input=20., delay=None):
    super().__init__()

    self.bg_exc = e_input
    self.bg_inh = i_input

    # network size
    num_exc = int(3200 * scale)
    num_inh = int(800 * scale)

    # neurons
    pars = dict(V_rest=-60., V_th=-50., V_reset=-60., tau=20., tau_ref=5.,
                V_initializer=bp.init.Normal(-55., 2.), input_var=False)
    self.E = bp.neurons.LIF(num_exc, **pars)
    self.I = bp.neurons.LIF(num_inh, **pars)

    # synapses
    we = 0.6 / scale  # excitatory synaptic weight (voltage)
    wi = 6.7 / scale  # inhibitory synaptic weight
    self.E2E = bp.experimental.Exponential(
      bp.conn.FixedProb(0.02, pre=self.E.size, post=self.E.size),
      g_max=we, tau=5., out=bp.experimental.COBA(E=0.)
    )
    self.E2I = bp.experimental.Exponential(
      bp.conn.FixedProb(0.02, pre=self.E.size, post=self.I.size, ),
      g_max=we, tau=5., out=bp.experimental.COBA(E=0.)
    )
    self.I2E = bp.experimental.Exponential(
      bp.conn.FixedProb(0.02, pre=self.I.size, post=self.E.size),
      g_max=wi, tau=10., out=bp.experimental.COBA(E=-80.)
    )
    self.I2I = bp.experimental.Exponential(
      bp.conn.FixedProb(0.02, pre=self.I.size, post=self.I.size),
      g_max=wi, tau=10., out=bp.experimental.COBA(E=-80.)
    )
    self.delayE = bp.Delay(self.E.spike, entries={'E': delay})
    self.delayI = bp.Delay(self.I.spike, entries={'I': delay})

  def update(self):
    e_spike = self.delayE.at('E')
    i_spike = self.delayI.at('I')
    e_inp = self.E2E(e_spike, self.E.V) + self.I2E(i_spike, self.E.V) + self.bg_exc
    i_inp = self.I2I(i_spike, self.I.V) + self.E2I(e_spike, self.I.V) + self.bg_inh
    self.delayE(self.E(e_inp))
    self.delayI(self.I(i_inp))

```



##### 3.  ``brainpy.checkpoints.save_pytree`` and ``brainpy.checkpoints.load_pytree`` for saving/loading target from the filename

Now we can directly use ``brainpy.checkpoints.save_pytree`` to save a network state into the file path we specified. 

Similarly, we can use ``brainpy.checkpoints.load_pytree`` to load states from the given file path.


##### 4. More ANN layers


- brainpy.layers.ConvTranspose1d
- brainpy.layers.ConvTranspose2d
- brainpy.layers.ConvTranspose3d
- brainpy.layers.Conv1dLSTMCell
- brainpy.layers.Conv2dLSTMCell
- brainpy.layers.Conv3dLSTMCell


##### 5. More compatible dense operators

PyTorch operators:

- brainpy.math.Tensor
- brainpy.math.flatten
- brainpy.math.cat
- brainpy.math.abs
- brainpy.math.absolute
- brainpy.math.acos
- brainpy.math.arccos
- brainpy.math.acosh
- brainpy.math.arccosh
- brainpy.math.add
- brainpy.math.addcdiv
- brainpy.math.addcmul
- brainpy.math.angle
- brainpy.math.asin
- brainpy.math.arcsin
- brainpy.math.asinh
- brainpy.math.arcsin
- brainpy.math.atan
- brainpy.math.arctan
- brainpy.math.atan2
- brainpy.math.atanh


TensorFlow operators:

- brainpy.math.concat
- brainpy.math.reduce_sum
- brainpy.math.reduce_max
- brainpy.math.reduce_min
- brainpy.math.reduce_mean
- brainpy.math.reduce_all
- brainpy.math.reduce_any
- brainpy.math.reduce_logsumexp
- brainpy.math.reduce_prod
- brainpy.math.reduce_std
- brainpy.math.reduce_variance
- brainpy.math.reduce_euclidean_norm
- brainpy.math.unsorted_segment_sqrt_n
- brainpy.math.segment_mean
- brainpy.math.unsorted_segment_sum
- brainpy.math.unsorted_segment_prod
- brainpy.math.unsorted_segment_max
- brainpy.math.unsorted_segment_min
- brainpy.math.unsorted_segment_mean
- brainpy.math.segment_sum
- brainpy.math.segment_prod
- brainpy.math.segment_max
- brainpy.math.segment_min
- brainpy.math.clip_by_value
- brainpy.math.cast


##### Others

- Remove the hard requirements of ``brainpylib`` and ``numba``.




### Version 2.3.4


This release mainly focuses on the compatibility with other frameworks:

1. Fix Jax import error when `jax>=0.4.2`
2. Backward compatibility of `brainpy.dyn` module 
3. Start to implement and be compatible with operators in pytorch and tensorflow, so that user's pytorch/tensorflow models can be easily migrated to brainpy


**Full Changelog**: https://github.com/brainpy/BrainPy/compare/V2.3.3...V2.3.4




### Version 2.3.3


Improve backward compatibility:

- monitors and inputs in ``DSRunner``
- models in ``brainpy.dyn``
- constants and function in ``brainpy.analysis``


### Version 2.3.2

This release (under the branch of ``brainpy=2.3.x``) continues to add support for brain-inspired computation.


#### New Features


##### 1. New package structure for stable API release

Unstable APIs are all hosted in ``brainpy._src`` module. 
Other APIs are stable and will be maintained for a long time. 


##### 2. New schedulers

- `brainpy.optim.CosineAnnealingWarmRestarts`
- `brainpy.optim.CosineAnnealingLR`
- `brainpy.optim.ExponentialLR`
- `brainpy.optim.MultiStepLR`
- `brainpy.optim.StepLR`


##### 3. Others

- support `static_argnums` in `brainpy.math.jit`
- fix bugs of `reset_state()` and `clear_input()` in `brainpy.channels`
- fix jit error checking






### Version 2.3.1

This release (under the release branch of ``brainpy=2.3.x``) continues to add supports for brain-inspired computation.



```python
import brainpy as bp
import brainpy.math as bm
```



#### Backwards Incompatible Changes



###### 1. Error: module 'brainpy' has no attribute 'datasets'

``brainpy.datasets`` module is now published as an independent package ``brainpy_datasets``. 

Please change your dataset access from 

```python
bp.datasets.xxxxx
```

to 

```python
import brainpy_datasets as bp_data

bp_data.chaos.XXX
bp_data.vision.XXX
```

For a chaotic data series, 

```python
# old version
data = bp.datasets.double_scroll_series(t_warmup + t_train + t_test, dt=dt)
x_var = data['x']
y_var = data['y']
z_var = data['z']

# new version
data = bd.chaos.DoubleScrollEq(t_warmup + t_train + t_test, dt=dt)
x_var = data.xs
y_var = data.ys
z_var = data.zs
```

For a vision dataset,

```python
# old version
dataset = bp.datasets.FashionMNIST(root, train=True, download=True)

# new version
dataset = bd.vision.FashionMNIST(root, split='train', download=True)
```



###### 2. Error: DSTrainer must receive an instance with BatchingMode

This error will happen when using ``brainpy.OnlineTrainer`` , ``brainpy.OfflineTrainer``, ``brainpy.BPTT`` , ``brainpy.BPFF``.

From version 2.3.1, BrainPy explicitly consider the computing mode of each model. For trainers, all training target should be a model with ``BatchingMode`` or ``TrainingMode``. 

If you are training model with ``OnlineTrainer`` or ``OfflineTrainer``, 

```python
# old version
class NGRC(bp.DynamicalSystem):
  def __init__(self, num_in):
    super(NGRC, self).__init__()
    self.r = bp.layers.NVAR(num_in, delay=2, order=3)
    self.di = bp.layers.Dense(self.r.num_out, num_in)

  def update(self, sha, x):
    di = self.di(sha, self.r(sha, x))
    return x + di


# new version
bm.set_enviroment(mode=bm.batching_mode)

class NGRC(bp.DynamicalSystem):
  def __init__(self, num_in):
    super(NGRC, self).__init__()
    self.r = bp.layers.NVAR(num_in, delay=2, order=3)
    self.di = bp.layers.Dense(self.r.num_out, num_in, mode=bm.training_mode)

  def update(self, sha, x):
    di = self.di(sha, self.r(sha, x))
    return x + di
```

 If you are training models with ``BPTrainer``, adding the following line at the top of the script,

```python
bm.set_enviroment(mode=bm.training_mode)
```



###### 3. Error: inputs_are_batching is no longer supported. 

This is because if the training target is in ``batching`` mode, this has already indicated that the inputs should be batching. 

Simple remove the ``inputs_are_batching`` from your functional call of ``.predict()`` will solve the issue. 





#### New Features



##### 1. ``brainpy.math`` module upgrade

###### ``brainpy.math.surrogate`` module for surrogate gradient functions.

Currently, we support

- `brainpy.math.surrogate.arctan`
- `brainpy.math.surrogate.erf`
- `brainpy.math.surrogate.gaussian_grad`
- `brainpy.math.surrogate.inv_square_grad`
- `brainpy.math.surrogate.leaky_relu`
- `brainpy.math.surrogate.log_tailed_relu`
- `brainpy.math.surrogate.multi_gaussian_grad`
- `brainpy.math.surrogate.nonzero_sign_log`
- `brainpy.math.surrogate.one_input`
- `brainpy.math.surrogate.piecewise_exp`
- `brainpy.math.surrogate.piecewise_leaky_relu`
- `brainpy.math.surrogate.piecewise_quadratic`
- `brainpy.math.surrogate.q_pseudo_spike`
- `brainpy.math.surrogate.relu_grad`
- `brainpy.math.surrogate.s2nn`
- `brainpy.math.surrogate.sigmoid`
- `brainpy.math.surrogate.slayer_grad`
- `brainpy.math.surrogate.soft_sign`
- `brainpy.math.surrogate.squarewave_fourier_series`



###### New transformation function ``brainpy.math.to_dynsys``

New transformation function ``brainpy.math.to_dynsys`` supports to transform a pure Python function into a ``DynamicalSystem``. This will be useful when running a `DynamicalSystem` with arbitrary customized inputs.

```python
import brainpy.math as bm

hh = bp.neurons.HH(1)

@bm.to_dynsys(child_objs=hh)
def run_hh(tdi, x=None):
    if x is not None:
	    hh.input += x
    
runner = bp.DSRunner(run_hhh, monitors={'v': hh.V})
runner.run(inputs=bm.random.uniform(3, 6, 1000))
```



###### Default data types 

Default data types `brainpy.math.int_`, `brainpy.math.float_` and `brainpy.math.complex_` are initialized according to the default `x64` settings. Then, these data types can be set or get by `brainpy.math.set_*` or `brainpy.math.get_*` syntaxes.

Take default integer type ``int_`` as an example, 

```python
# set the default integer type
bm.set_int_(jax.numpy.int64)

# get the default integer type
a1 = bm.asarray([1], dtype=bm.int_)
a2 = bm.asarray([1], dtype=bm.get_int()) # equivalent
```

Default data types are changed according to the `x64` setting of JAX. For instance, 

```python
bm.enable_x64()
assert bm.int_ == jax.numpy.int64
bm.disable_x64()
assert bm.int_ == jax.numpy.int32
```

``brainpy.math.float_`` and ``brainpy.math.complex_`` behaves similarly with ``brainpy.math.int_``.



###### Environment context manager 

This release introduces a new concept  ``computing environment``  in BrainPy. Computing environment is a default setting for current computation jobs, including the default data type (``int_``, ``float_``, ``complex_``), the default numerical integration precision (``dt``), the default computing mode (``mode``). All models, arrays, and computations using the default setting will be carried out under the environment setting. 

Users can set a default environment through 

```python
brainpy.math.set_environment(mode, dt, x64)
```

However, ones can also construct models or perform computation through a temporal environment context manager, this can be implemented through:

```python
# constructing a HH model with dt=0.1 and x64 precision
with bm.environment(mode, dt=0.1, x64=True):
    hh1 = bp.neurons.HH(1)
    
# constructing a HH model with dt=0.05 and x32 precision
with bm.environment(mode, dt=0.05, x64=False):
    hh2 = bp.neuron.HH(1)
```

Usually, users construct models for either brain-inspired computing (``training mode``) or brain simulation (``nonbatching mode``), therefore, there are shortcut context manager for setting a training environment or batching environment:

```python
with bm.training_environment(dt, x64):
    pass

with bm.batching_environment(dt, x64):
    pass
```



##### 2. ``brainpy.dyn`` module



###### ``brainpy.dyn.transfom`` module for transforming a ``DynamicalSystem`` instance to a callable ``BrainPyObject``. 

Specifically, we provide

- `LoopOverTime` for unrolling a dynamical system over time.
- `NoSharedArg` for removing the dependency of shared arguments.





##### 3. Running supports in BrainPy



###### All ``brainpy.Runner`` now are subclasses of ``BrainPyObject``

This means that all ``brainpy.Runner`` can be used as a part of the high-level program or transformation. 



###### Enable the continuous running of a differential equation (ODE, SDE, FDE, DDE, etc.) with `IntegratorRunner`. 

For example,

```python
import brainpy as bp

# differential equation
a, b, tau = 0.7, 0.8, 12.5
dV = lambda V, t, w, Iext: V - V * V * V / 3 - w + Iext
dw = lambda w, t, V: (V + a - b * w) / tau
fhn = bp.odeint(bp.JointEq([dV, dw]), method='rk4', dt=0.1)

# differential integrator runner
runner = bp.IntegratorRunner(fhn, monitors=['V', 'w'], inits=[1., 1.])

# run 1
Iext, duration = bp.inputs.section_input([0., 1., 0.5], [200, 200, 200], return_length=True)
runner.run(duration, dyn_args=dict(Iext=Iext))
bp.visualize.line_plot(runner.mon.ts, runner.mon['V'], legend='V')

# run 2
Iext, duration = bp.inputs.section_input([0.5], [200], return_length=True)
runner.run(duration, dyn_args=dict(Iext=Iext))
bp.visualize.line_plot(runner.mon.ts, runner.mon['V'], legend='V-run2', show=True)

```



###### Enable call a customized function during fitting of ``brainpy.BPTrainer``.

This customized function (provided through ``fun_after_report``) will be useful to save a checkpoint during the training. For instance, 

```python
class CheckPoint:
    def __init__(self, path='path/to/directory/'):
        self.max_acc = 0.
        self.path = path
        
    def __call__(self, idx, metrics, phase):
        if phase == 'test' and metrics['acc'] > self.max_acc:
            self.max_acc = matrics['acc']
            bp.checkpoints.save(self.path, net.state_dict(), idx)

trainer = bp.BPTT()
trainer.fit(..., fun_after_report=CheckPoint())    
```



###### Enable data with ``data_first_axis`` format when predicting or fitting in a ``brainpy.DSRunner`` and ``brainpy.DSTrainer``. 

Previous version of BrainPy only supports data with the batch dimension at the first axis. Currently, ``brainpy.DSRunner`` and ``brainpy.DSTrainer`` can support the data with the time dimension at the first axis. This can be set through ``data_first_axis='T'`` when initializing a runner or trainer. 

```python
runner = bp.DSRunner(..., data_first_axis='T')
trainer = bp.DSTrainer(..., data_first_axis='T')
```



##### 4. Utility in BrainPy



###### ``brainpy.encoding`` module for encoding rate values into spike trains

 Currently, we support

- `brainpy.encoding.LatencyEncoder`
- `brainpy.encoding.PoissonEncoder`
- `brainpy.encoding.WeightedPhaseEncoder`



###### ``brainpy.checkpoints`` module for model state serialization. 

This version of BrainPy supports to save a checkpoint of the model into the physical disk. Inspired from the Flax API, we provide the following checkpoint APIs:

- ``brainpy.checkpoints.save()`` for saving a checkpoint of the model.
- ``brainpy.checkpoints.multiprocess_save()`` for saving a checkpoint of the model in multi-process environment.
- ``brainpy.checkpoints.load()`` for loading the last or best checkpoint from the given checkpoint path.
- ``brainpy.checkpoints.load_latest()`` for retrieval the path of the latest checkpoint in a directory.





#### Deprecations



##### 1. Deprecations in the running supports of BrainPy

###### ``func_monitors`` is no longer supported in all ``brainpy.Runner`` subclasses.

We will remove its supports since version 2.4.0. Instead, monitoring with a dict of callable functions can be set  in ``monitors``. For example, 


   ```python
   # old version
   
   runner = bp.DSRunner(model, 
                        monitors={'sps': model.spike, 'vs': model.V},
                        func_monitors={'sp10': model.spike[10]})
   ```

   ```python
   # new version
   runner = bp.DSRunner(model, 
                        monitors={'sps': model.spike, 
                                  'vs': model.V, 
                                  'sp10': model.spike[10]})
   ```



###### ``func_inputs`` is no longer supported in all ``brainpy.Runner`` subclasses.

 Instead, giving inputs with a callable function should be done with ``inputs``. 

```python
# old version

net = EINet()

def f_input(tdi):
    net.E.input += 10.

runner = bp.DSRunner(net, fun_inputs=f_input, inputs=('I.input', 10.))
```

```python
# new version

def f_input(tdi):
    net.E.input += 10.
    net.I.input += 10.
runner = bp.DSRunner(net, inputs=f_input)
```



###### ``inputs_are_batching`` is deprecated. 

``inputs_are_batching`` is deprecated in ``predict()``/``.run()`` of all ``brainpy.Runner`` subclasses. 



###### ``args`` and ``dyn_args`` are now  deprecated in ``IntegratorRunner``.

Instead, users should specify ``args`` and ``dyn_args`` when using ``IntegratorRunner.run()`` function.  

```python
dV = lambda V, t, w, I: V - V * V * V / 3 - w + I
dw = lambda w, t, V, a, b: (V + a - b * w) / 12.5
integral = bp.odeint(bp.JointEq([dV, dw]), method='exp_auto')

# old version
runner = bp.IntegratorRunner(
  integral,
  monitors=['V', 'w'], 
  inits={'V': bm.random.rand(10), 'w': bm.random.normal(size=10)},
  args={'a': 1., 'b': 1.},  # CHANGE
  dyn_args={'I': bp.inputs.ramp_input(0, 4, 100)},  # CHANGE
)
runner.run(100.,)

```

```python
# new version
runner = bp.IntegratorRunner(
  integral,
  monitors=['V', 'w'], 
  inits={'V': bm.random.rand(10), 'w': bm.random.normal(size=10)},
)
runner.run(100., 
           args={'a': 1., 'b': 1.},
           dyn_args={'I': bp.inputs.ramp_input(0, 4, 100)})
```



#####  2. Deprecations in ``brainpy.math`` module

###### `ditype()` and `dftype()` are deprecated.

`brainpy.math.ditype()` and `brainpy.math.dftype()` are deprecated. Using `brainpy.math.int_` and `brainpy.math.float()` instead. 



###### ``brainpy.modes`` module is  now moved into ``brainpy.math``

The correspondences are listed as the follows:

- ``brainpy.modes.Mode``  => ``brainpy.math.Mode``
- ``brainpy.modes.NormalMode `` => ``brainpy.math.NonBatchingMode`` 
- ``brainpy.modes.BatchingMode `` => ``brainpy.math.BatchingMode`` 
- ``brainpy.modes.TrainingMode `` => ``brainpy.math.TrainingMode`` 
- ``brainpy.modes.normal `` => ``brainpy.math.nonbatching_mode`` 
- ``brainpy.modes.batching `` => ``brainpy.math.batching_mode`` 
- ``brainpy.modes.training `` => ``brainpy.math.training_mode`` 






### Version 2.3.0

This branch of releases aims to provide a unified computing framework for brain simulation and brain-inspired computing.

#### New features

1. ``brainpy.BPTT`` supports `train_data` and `test_data` with general Python iterators. For instance, one can train a model with PyTorch dataloader or TensorFlow datasets.

```python
import torchvision
from torch.utils.data import DataLoader
data = torchvision.datasets.CIFAR10("./CIFAR10", train=False, transform=torchvision.transforms.ToTensor())
loader = DataLoader(dataset=data, batch_size=4, shuffle=True, num_workers=0, drop_last=False)

# any generator can be used for train_data or test_data
trainer = bp.BPTT()
trainer.fit(loader)
```

2. Consolidated object-oriented transformation in ``brainpy.math.object_transform`` module. All brainpy transformations generate a new ``BrainPyObject`` instance so that objects in brainpy can be composed hierarchically.  ``brainpy.math.to_object()`` transformation transforms a pure Python function into a ``BrainPyObject``.

3. New [documentation](https://brainpy.readthedocs.io/en/latest/tutorial_math/brainpy_transform_concept.html) is currently online for introducing the consolidated BrainPy concept of object-oriented transformation. 

4. Change ``brainpy.math.JaxArray`` to ``brainpy.math.Array``.




#### Deprecations

1. ``brainpy.datasets`` module is no longer supported. New APIs will be moved into [``brainpy-datasets`` package](https://github.com/brainpy/datasets). 
2. ``brainpy.train.BPTT`` no longer support to receive the train data `[X, Y]`. Instead, users should provide a data generator such like ``pytorch`` dataset or ``tensorflow`` dataset. 
4. The update function of ``brainpy.math.TimeDealy`` does not support receiving a `time` index. Instead, one can update the new data by directly using ``TimeDealy.update(data)`` instead of `TimeDealy.update(time, data)`.
5. Fix the monitoring error of delay differential equations with  ``brainpy.integrators.IntegratorRunner``. 

#### Bug Fixes

1. Fix the bug on ``One2One`` connection. 
2. Fix the bug in ``eprop`` example.
3. Fix `ij2csr` transformation error.
4. Fix test bugs

#### What's Changed
* fix eprop example error by @chaoming0625 in https://github.com/brainpy/BrainPy/pull/305
* minor updates on API and DOC by @chaoming0625 in https://github.com/brainpy/BrainPy/pull/306
* Add new optimizers by @chaoming0625 in https://github.com/brainpy/BrainPy/pull/307
* add documentation of for random number generation by @chaoming0625 in https://github.com/brainpy/BrainPy/pull/308
* consolidate the concept of OO transformation by @chaoming0625 in https://github.com/brainpy/BrainPy/pull/309
* Upgrade documetations by @chaoming0625 in https://github.com/brainpy/BrainPy/pull/310
* Ready for publish by @chaoming0625 in https://github.com/brainpy/BrainPy/pull/311


**Full Changelog**: https://github.com/brainpy/BrainPy/compare/V2.2.4.0...V2.3.0


## brainpy 2.2.x

BrainPy 2.2.x is a complete re-design of the framework, tackling the
shortcomings of brainpy 2.1.x generation, effectively bringing it to
research needs and standards.




### Version 2.2.4

This release has updated many functionalities and fixed several bugs in BrainPy.

#### New Features

1. More ANN layers, including ``brainpy.layers.Flatten`` and ``brainpy.layers.Activation``.
2. Optimized connection building for ``brainpy.connect`` module. 
3. cifar dataset. 
4. Enhanced API and Doc for parallel simulations via ``brainpy.running.cpu_ordered_parallel``, ``brainpy.running.cpu_unordered_parallel``, ``brainpy.running.jax_vectorize_map`` and ``brainpy.running.jax_parallelize_map``. 


#### What's Changed
* add Activation and Flatten class by @LuckyHFC in https://github.com/PKU-NIP-Lab/BrainPy/pull/291
* optimizes the connect time when using gpu by @MamieZhu in https://github.com/PKU-NIP-Lab/BrainPy/pull/293
* datasets::vision: add cifar dataset by @hbelove in https://github.com/PKU-NIP-Lab/BrainPy/pull/292
* fix #294: remove VariableView in `dyn_vars` of a runner by @chaoming0625 in https://github.com/PKU-NIP-Lab/BrainPy/pull/295
* update issue template by @chaoming0625 in https://github.com/PKU-NIP-Lab/BrainPy/pull/296
* add multiprocessing functions for batch running of BrainPy functions by @chaoming0625 in https://github.com/PKU-NIP-Lab/BrainPy/pull/298
* upgrade connection apis by @chaoming0625 in https://github.com/PKU-NIP-Lab/BrainPy/pull/299
* fix #300: update parallelization api documentation by @chaoming0625 in https://github.com/PKU-NIP-Lab/BrainPy/pull/302
* update doc by @chaoming0625 in https://github.com/PKU-NIP-Lab/BrainPy/pull/303

#### New Contributors
* @LuckyHFC made their first contribution in https://github.com/PKU-NIP-Lab/BrainPy/pull/291
* @MamieZhu made their first contribution in https://github.com/PKU-NIP-Lab/BrainPy/pull/293
* @hbelove made their first contribution in https://github.com/PKU-NIP-Lab/BrainPy/pull/292

**Full Changelog**: https://github.com/PKU-NIP-Lab/BrainPy/compare/V2.2.3.6...V2.2.4




### Version 2.2.1 (2022.09.09)

This release fixes bugs found in the codebase and improves the usability
and functions of BrainPy.

#### Bug fixes

1. Fix the bug of operator customization in `brainpy.math.XLACustomOp`
   and `brainpy.math.register_op`. Now, it supports operator
   customization by using NumPy and Numba interface. For instance,

``` python
import brainpy.math as bm

def abs_eval(events, indices, indptr, post_val, values):
      return post_val

def con_compute(outs, ins):
      post_val = outs
      events, indices, indptr, _, values = ins
      for i in range(events.size):
        if events[i]:
          for j in range(indptr[i], indptr[i + 1]):
            index = indices[j]
            old_value = post_val[index]
            post_val[index] = values + old_value

event_sum = bm.XLACustomOp(eval_shape=abs_eval, con_compute=con_compute)
```

1. Fix the bug of `brainpy.tools.DotDict`. Now, it is compatible with
   the transformations of JAX. For instance,

``` python
import brainpy as bp
from jax import vmap

@vmap
def multiple_run(I):
  hh = bp.neurons.HH(1)
  runner = bp.dyn.DSRunner(hh, inputs=('input', I), numpy_mon_after_run=False)
  runner.run(100.)
  return runner.mon

mon = multiple_run(bp.math.arange(2, 10, 2))
```

#### New features

1. Add numpy operators `brainpy.math.mat`, `brainpy.math.matrix`,
   `brainpy.math.asmatrix`.
2. Improve translation rules of brainpylib operators, improve its
   running speeds.
3. Support `DSView` of `DynamicalSystem` instance. Now, it supports
   defining models with a slice view of a DS instance. For example,

``` python
import brainpy as bp
import brainpy.math as bm


class EINet_V2(bp.dyn.Network):
  def __init__(self, scale=1.0, method='exp_auto'):
    super(EINet_V2, self).__init__()

    # network size
    num_exc = int(3200 * scale)
    num_inh = int(800 * scale)

    # neurons
    self.N = bp.neurons.LIF(num_exc + num_inh,
                            V_rest=-60., V_th=-50., V_reset=-60., tau=20., tau_ref=5.,
                            method=method, V_initializer=bp.initialize.Normal(-55., 2.))

    # synapses
    we = 0.6 / scale  # excitatory synaptic weight (voltage)
    wi = 6.7 / scale  # inhibitory synaptic weight
    self.Esyn = bp.synapses.Exponential(pre=self.N[:num_exc], post=self.N,
                                        conn=bp.connect.FixedProb(0.02),
                                        g_max=we, tau=5.,
                                        output=bp.synouts.COBA(E=0.),
                                        method=method)
    self.Isyn = bp.synapses.Exponential(pre=self.N[num_exc:], post=self.N,
                                        conn=bp.connect.FixedProb(0.02),
                                        g_max=wi, tau=10.,
                                        output=bp.synouts.COBA(E=-80.),
                                        method=method)

net = EINet_V2(scale=1., method='exp_auto')
# simulation
runner = bp.dyn.DSRunner(
    net,
    monitors={'spikes': net.N.spike},
    inputs=[(net.N.input, 20.)]
  )
runner.run(100.)

# visualization
bp.visualize.raster_plot(runner.mon.ts, runner.mon['spikes'], show=True)
```

### Version 2.2.0 (2022.08.12)

This release has provided important improvements for BrainPy, including
usability, speed, functions, and others.

#### Backwards Incompatible changes

1. `brainpy.nn` module is no longer supported and has been removed
   since version 2.2.0. Instead, users should use `brainpy.train`
   module for the training of BP algorithms, online learning, or
   offline learning algorithms, and `brainpy.algorithms` module for
   online / offline training algorithms.
2. The `update()` function for the model definition has been changed:

``` 
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
```

where `tdi` can be defined with other names, like `sha`, to represent
the shared argument across modules.

#### Deprecations

1. `brainpy.dyn.xxx (neurons)` and `brainpy.dyn.xxx (synapse)` are no
   longer supported. Please use `brainpy.neurons`, `brainpy.synapses`
   modules.
2. `brainpy.running.monitor` has been removed.
3. `brainpy.nn` module has been removed.

#### New features

1. `brainpy.math.Variable` receives a `batch_axis` setting to represent
   the batch axis of the data.

``` 
>>> import brainpy.math as bm
>>> a = bm.Variable(bm.zeros((1, 4, 5)), batch_axis=0)
>>> a.value = bm.zeros((2, 4, 5))  # success
>>> a.value = bm.zeros((1, 2, 5))  # failed
MathError: The shape of the original data is (2, 4, 5), while we got (1, 2, 5) with batch_axis=0.
```

2. `brainpy.train` provides `brainpy.train.BPTT` for back-propagation
   algorithms, `brainpy.train.Onlinetrainer` for online training
   algorithms, `brainpy.train.OfflineTrainer` for offline training
   algorithms.
3. `brainpy.Base` class supports `_excluded_vars` setting to ignore
   variables when retrieving variables by using `Base.vars()` method.

``` 
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
```

4. `brainpy.analysis.SlowPointFinder` supports directly analyzing an
   instance of `brainpy.dyn.DynamicalSystem`.

``` 
>>> hh = bp.neurons.HH(1)
>>> finder = bp.analysis.SlowPointFinder(hh, target_vars={'V': hh.V, 'm': hh.m, 'h': hh.h, 'n': hh.n})
```

5. `brainpy.datasets` supports MNIST, FashionMNIST, and other datasets.
6. Supports defining conductance-based neuron models\`\`.

``` 
>>> class HH(bp.dyn.CondNeuGroup):
>>>   def __init__(self, size):
>>>     super(HH, self).__init__(size)
>>>
>>>     self.INa = channels.INa_HH1952(size, )
>>>     self.IK = channels.IK_HH1952(size, )
>>>     self.IL = channels.IL(size, E=-54.387, g_max=0.03)
```

7. `brainpy.layers` module provides commonly used models for DNN and
   reservoir computing.
8. Support composable definition of synaptic models by using
   `TwoEndConn`, `SynOut`, `SynSTP` and `SynLTP`.

``` 
>>> bp.synapses.Exponential(self.E, self.E, bp.conn.FixedProb(prob),
>>>                      g_max=0.03 / scale, tau=5,
>>>                      output=bp.synouts.COBA(E=0.),
>>>                      stp=bp.synplast.STD())
```

9. Provide commonly used surrogate gradient function for spiking
   generation, including
    - `brainpy.math.spike_with_sigmoid_grad`
    - `brainpy.math.spike_with_linear_grad`
    - `brainpy.math.spike_with_gaussian_grad`
    - `brainpy.math.spike_with_mg_grad`
10. Provide shortcuts for GPU memory management via
    `brainpy.math.disable_gpu_memory_preallocation()`, and
    `brainpy.math.clear_buffer_memory()`.

#### What\'s Changed

- fix [#207](https://github.com/PKU-NIP-Lab/BrainPy/issues/207):
  synapses update first, then neurons, finally delay variables by
  [\@chaoming0625](https://github.com/chaoming0625) in
  [#219](https://github.com/PKU-NIP-Lab/BrainPy/pull/219)
- docs: add logos by [\@ztqakita](https://github.com/ztqakita) in
  [#218](https://github.com/PKU-NIP-Lab/BrainPy/pull/218)
- Add the biological NMDA model by
  [\@c-xy17](https://github.com/c-xy17) in
  [#221](https://github.com/PKU-NIP-Lab/BrainPy/pull/221)
- docs: fix mathjax problem by
  [\@ztqakita](https://github.com/ztqakita) in
  [#222](https://github.com/PKU-NIP-Lab/BrainPy/pull/222)
- Add the parameter R to the LIF model by
  [\@c-xy17](https://github.com/c-xy17) in
  [#224](https://github.com/PKU-NIP-Lab/BrainPy/pull/224)
- new version of brainpy: V2.2.0-rc1 by
  [\@chaoming0625](https://github.com/chaoming0625) in
  [#226](https://github.com/PKU-NIP-Lab/BrainPy/pull/226)
- update training apis by
  [\@chaoming0625](https://github.com/chaoming0625) in
  [#227](https://github.com/PKU-NIP-Lab/BrainPy/pull/227)
- Update quickstart and the analysis module by
  [\@c-xy17](https://github.com/c-xy17) in
  [#229](https://github.com/PKU-NIP-Lab/BrainPy/pull/229)
- Eseential updates for montors, analysis, losses, and examples by
  [\@chaoming0625](https://github.com/chaoming0625) in
  [#230](https://github.com/PKU-NIP-Lab/BrainPy/pull/230)
- add numpy op tests by [\@ztqakita](https://github.com/ztqakita) in
  [#231](https://github.com/PKU-NIP-Lab/BrainPy/pull/231)
- Integrated simulation, simulaton and analysis by
  [\@chaoming0625](https://github.com/chaoming0625) in
  [#232](https://github.com/PKU-NIP-Lab/BrainPy/pull/232)
- update docs by [\@chaoming0625](https://github.com/chaoming0625) in
  [#233](https://github.com/PKU-NIP-Lab/BrainPy/pull/233)
- unify `brainpy.layers` with other modules in `brainpy.dyn` by
  [\@chaoming0625](https://github.com/chaoming0625) in
  [#234](https://github.com/PKU-NIP-Lab/BrainPy/pull/234)
- fix bugs by [\@chaoming0625](https://github.com/chaoming0625) in
  [#235](https://github.com/PKU-NIP-Lab/BrainPy/pull/235)
- update apis, docs, examples and others by
  [\@chaoming0625](https://github.com/chaoming0625) in
  [#236](https://github.com/PKU-NIP-Lab/BrainPy/pull/236)
- fixes by [\@chaoming0625](https://github.com/chaoming0625) in
  [#237](https://github.com/PKU-NIP-Lab/BrainPy/pull/237)
- fix: add dtype promotion = standard by
  [\@ztqakita](https://github.com/ztqakita) in
  [#239](https://github.com/PKU-NIP-Lab/BrainPy/pull/239)
- updates by [\@chaoming0625](https://github.com/chaoming0625) in
  [#240](https://github.com/PKU-NIP-Lab/BrainPy/pull/240)
- update training docs by
  [\@chaoming0625](https://github.com/chaoming0625) in
  [#241](https://github.com/PKU-NIP-Lab/BrainPy/pull/241)
- change doc path/organization by
  [\@chaoming0625](https://github.com/chaoming0625) in
  [#242](https://github.com/PKU-NIP-Lab/BrainPy/pull/242)
- Update advanced docs by
  [\@chaoming0625](https://github.com/chaoming0625) in
  [#243](https://github.com/PKU-NIP-Lab/BrainPy/pull/243)
- update quickstart docs & enable jit error checking by
  [\@chaoming0625](https://github.com/chaoming0625) in
  [#244](https://github.com/PKU-NIP-Lab/BrainPy/pull/244)
- update apis and examples by
  [\@chaoming0625](https://github.com/chaoming0625) in
  [#245](https://github.com/PKU-NIP-Lab/BrainPy/pull/245)
- update apis and tests by
  [\@chaoming0625](https://github.com/chaoming0625) in
  [#246](https://github.com/PKU-NIP-Lab/BrainPy/pull/246)
- Docs update and bugs fixed by
  [\@ztqakita](https://github.com/ztqakita) in
  [#247](https://github.com/PKU-NIP-Lab/BrainPy/pull/247)
- version 2.2.0 by [\@chaoming0625](https://github.com/chaoming0625)
  in [#248](https://github.com/PKU-NIP-Lab/BrainPy/pull/248)
- add norm and pooling & fix bugs in operators by
  [\@ztqakita](https://github.com/ztqakita) in
  [#249](https://github.com/PKU-NIP-Lab/BrainPy/pull/249)

**Full Changelog**:
[V2.1.12\...V2.2.0](https://github.com/PKU-NIP-Lab/BrainPy/compare/V2.1.12...V2.2.0)

## brainpy 2.1.x

### Version 2.1.12 (2022.05.17)

#### Highlights

This release is excellent. We have made important improvements.

1. We provide dozens of random sampling in NumPy which are not
   supportted in JAX, such as `brainpy.math.random.bernoulli`,
   `brainpy.math.random.lognormal`, `brainpy.math.random.binomial`,
   `brainpy.math.random.chisquare`, `brainpy.math.random.dirichlet`,
   `brainpy.math.random.geometric`, `brainpy.math.random.f`,
   `brainpy.math.random.hypergeometric`,
   `brainpy.math.random.logseries`, `brainpy.math.random.multinomial`,
   `brainpy.math.random.multivariate_normal`,
   `brainpy.math.random.negative_binomial`,
   `brainpy.math.random.noncentral_chisquare`,
   `brainpy.math.random.noncentral_f`, `brainpy.math.random.power`,
   `brainpy.math.random.rayleigh`, `brainpy.math.random.triangular`,
   `brainpy.math.random.vonmises`, `brainpy.math.random.wald`,
   `brainpy.math.random.weibull`
2. make efficient checking on numerical values. Instead of direct
   `id_tap()` checking which has large overhead, currently
   `brainpy.tools.check_erro_in_jit()` is highly efficient.
3. Fix `JaxArray` operator errors on `None`
4. improve oo-to-function transformation speeds
5. `io` works: `.save_states()` and `.load_states()`

#### What's Changed

- support dtype setting in array interchange functions by
  \[@chaoming0625\](<https://github.com/chaoming0625>) in
  [#209](https://github.com/PKU-NIP-Lab/BrainPy/pull/209)
- fix [#144](https://github.com/PKU-NIP-Lab/BrainPy/issues/144):
  operations on None raise errors by
  \[@chaoming0625\](<https://github.com/chaoming0625>) in
  [#210](https://github.com/PKU-NIP-Lab/BrainPy/pull/210)
- add tests and new functions for random sampling by
  \[@c-xy17\](<https://github.com/c-xy17>) in
  [#213](https://github.com/PKU-NIP-Lab/BrainPy/pull/213)
- feat: fix `io` for brainpy.Base by
  \[@chaoming0625\](<https://github.com/chaoming0625>) in
  [#211](https://github.com/PKU-NIP-Lab/BrainPy/pull/211)
- update advanced tutorial documentation by
  \[@chaoming0625\](<https://github.com/chaoming0625>) in
  [#212](https://github.com/PKU-NIP-Lab/BrainPy/pull/212)
- fix [#149](https://github.com/PKU-NIP-Lab/BrainPy/issues/149)
  (dozens of random samplings in NumPy) and fix JaxArray op errors by
  \[@chaoming0625\](<https://github.com/chaoming0625>) in
  [#216](https://github.com/PKU-NIP-Lab/BrainPy/pull/216)
- feat: efficient checking on numerical values by
  \[@chaoming0625\](<https://github.com/chaoming0625>) in
  [#217](https://github.com/PKU-NIP-Lab/BrainPy/pull/217)

**Full Changelog**:
[V2.1.11\...V2.1.12](https://github.com/PKU-NIP-Lab/BrainPy/compare/V2.1.11...V2.1.12)

### Version 2.1.11 (2022.05.15)

#### What\'s Changed

- fix: cross-correlation bug by
  [\@ztqakita](https://github.com/ztqakita) in
  [#201](https://github.com/PKU-NIP-Lab/BrainPy/pull/201)
- update apis, test and docs of numpy ops by
  [\@chaoming0625](https://github.com/chaoming0625) in
  [#202](https://github.com/PKU-NIP-Lab/BrainPy/pull/202)
- docs: add sphinx_book_theme by
  [\@ztqakita](https://github.com/ztqakita) in
  [#203](https://github.com/PKU-NIP-Lab/BrainPy/pull/203)
- fix: add requirements-doc.txt by
  [\@ztqakita](https://github.com/ztqakita) in
  [#204](https://github.com/PKU-NIP-Lab/BrainPy/pull/204)
- update control flow, integrators, operators, and docs by
  [\@chaoming0625](https://github.com/chaoming0625) in
  [#205](https://github.com/PKU-NIP-Lab/BrainPy/pull/205)
- improve oo-to-function transformation speed by
  [\@chaoming0625](https://github.com/chaoming0625) in
  [#208](https://github.com/PKU-NIP-Lab/BrainPy/pull/208)

**Full Changelog**:
[V2.1.10\...V2.1.11](https://github.com/PKU-NIP-Lab/BrainPy/compare/V2.1.10...V2.1.11)

### Version 2.1.10 (2022.05.05)

#### What\'s Changed

- update control flow APIs and Docs by
  [\@chaoming0625](https://github.com/chaoming0625) in
  [#192](https://github.com/PKU-NIP-Lab/BrainPy/pull/192)
- doc: update docs of dynamics simulation by
  [\@chaoming0625](https://github.com/chaoming0625) in
  [#193](https://github.com/PKU-NIP-Lab/BrainPy/pull/193)
- fix [#125](https://github.com/PKU-NIP-Lab/BrainPy/issues/125): add
  channel models and two-compartment Pinsky-Rinzel model by
  [\@chaoming0625](https://github.com/chaoming0625) in
  [#194](https://github.com/PKU-NIP-Lab/BrainPy/pull/194)
- JIT errors do not change Variable values by
  [\@chaoming0625](https://github.com/chaoming0625) in
  [#195](https://github.com/PKU-NIP-Lab/BrainPy/pull/195)
- fix a bug in math.activations.py by
  [\@c-xy17](https://github.com/c-xy17) in
  [#196](https://github.com/PKU-NIP-Lab/BrainPy/pull/196)
- Functionalinaty improvements by
  [\@chaoming0625](https://github.com/chaoming0625) in
  [#197](https://github.com/PKU-NIP-Lab/BrainPy/pull/197)
- update rate docs by
  [\@chaoming0625](https://github.com/chaoming0625) in
  [#198](https://github.com/PKU-NIP-Lab/BrainPy/pull/198)
- update brainpy.dyn doc by
  [\@chaoming0625](https://github.com/chaoming0625) in
  [#199](https://github.com/PKU-NIP-Lab/BrainPy/pull/199)

**Full Changelog**:
[V2.1.8\...V2.1.10](https://github.com/PKU-NIP-Lab/BrainPy/compare/V2.1.8...V2.1.10)

### Version 2.1.8 (2022.04.26)

#### What\'s Changed

- Fix [#120](https://github.com/PKU-NIP-Lab/BrainPy/issues/120) by
  [\@chaoming0625](https://github.com/chaoming0625) in
  [#178](https://github.com/PKU-NIP-Lab/BrainPy/pull/178)
- feat: brainpy.Collector supports addition and subtraction by
  [\@chaoming0625](https://github.com/chaoming0625) in
  [#179](https://github.com/PKU-NIP-Lab/BrainPy/pull/179)
- feat: delay variables support \"indices\" and \"reset()\" function
  by [\@chaoming0625](https://github.com/chaoming0625) in
  [#180](https://github.com/PKU-NIP-Lab/BrainPy/pull/180)
- Support reset functions in neuron and synapse models by
  [\@chaoming0625](https://github.com/chaoming0625) in
  [#181](https://github.com/PKU-NIP-Lab/BrainPy/pull/181)
- `update()` function on longer need `_t` and `_dt` by
  [\@chaoming0625](https://github.com/chaoming0625) in
  [#183](https://github.com/PKU-NIP-Lab/BrainPy/pull/183)
- small updates by [\@chaoming0625](https://github.com/chaoming0625)
  in [#188](https://github.com/PKU-NIP-Lab/BrainPy/pull/188)
- feat: easier control flows with `brainpy.math.ifelse` by
  [\@chaoming0625](https://github.com/chaoming0625) in
  [#189](https://github.com/PKU-NIP-Lab/BrainPy/pull/189)
- feat: update delay couplings of `DiffusiveCoupling` and
  `AdditiveCouping` by
  [\@chaoming0625](https://github.com/chaoming0625) in
  [#190](https://github.com/PKU-NIP-Lab/BrainPy/pull/190)
- update version and changelog by
  [\@chaoming0625](https://github.com/chaoming0625) in
  [#191](https://github.com/PKU-NIP-Lab/BrainPy/pull/191)

**Full Changelog**:
[V2.1.7\...V2.1.8](https://github.com/PKU-NIP-Lab/BrainPy/compare/V2.1.7...V2.1.8)

### Version 2.1.7 (2022.04.22)

#### What\'s Changed

- synapse models support heterogeneuos weights by
  [\@chaoming0625](https://github.com/chaoming0625) in
  [#170](https://github.com/PKU-NIP-Lab/BrainPy/pull/170)
- more efficient synapse implementation by
  [\@chaoming0625](https://github.com/chaoming0625) in
  [#171](https://github.com/PKU-NIP-Lab/BrainPy/pull/171)
- fix input models in brainpy.dyn by
  [\@chaoming0625](https://github.com/chaoming0625) in
  [#172](https://github.com/PKU-NIP-Lab/BrainPy/pull/172)
- fix: np array astype by [\@ztqakita](https://github.com/ztqakita) in
  [#173](https://github.com/PKU-NIP-Lab/BrainPy/pull/173)
- update README: \'brain-py\' to \'brainpy\' by
  [\@chaoming0625](https://github.com/chaoming0625) in
  [#174](https://github.com/PKU-NIP-Lab/BrainPy/pull/174)
- fix: fix the updating rules in the STP model by
  [\@c-xy17](https://github.com/c-xy17) in
  [#176](https://github.com/PKU-NIP-Lab/BrainPy/pull/176)
- Updates and fixes by
  [\@chaoming0625](https://github.com/chaoming0625) in
  [#177](https://github.com/PKU-NIP-Lab/BrainPy/pull/177)

**Full Changelog**:
[V2.1.5\...V2.1.7](https://github.com/PKU-NIP-Lab/BrainPy/compare/V2.1.5...V2.1.7)

### Version 2.1.5 (2022.04.18)

#### What\'s Changed

- `brainpy.math.random.shuffle` is numpy like by
  [\@chaoming0625](https://github.com/chaoming0625) in
  [#153](https://github.com/PKU-NIP-Lab/BrainPy/pull/153)
- update LICENSE by [\@chaoming0625](https://github.com/chaoming0625)
  in [#155](https://github.com/PKU-NIP-Lab/BrainPy/pull/155)
- docs: add m1 warning by [\@ztqakita](https://github.com/ztqakita) in
  [#154](https://github.com/PKU-NIP-Lab/BrainPy/pull/154)
- compatible apis of \'brainpy.math\' with those of \'jax.numpy\' in
  most modules by [\@chaoming0625](https://github.com/chaoming0625) in
  [#156](https://github.com/PKU-NIP-Lab/BrainPy/pull/156)
- Important updates by
  [\@chaoming0625](https://github.com/chaoming0625) in
  [#157](https://github.com/PKU-NIP-Lab/BrainPy/pull/157)
- Updates by [\@chaoming0625](https://github.com/chaoming0625) in
  [#159](https://github.com/PKU-NIP-Lab/BrainPy/pull/159)
- Add LayerNorm, GroupNorm, and InstanceNorm as nn_nodes in
  normalization.py by [\@c-xy17](https://github.com/c-xy17) in
  [#162](https://github.com/PKU-NIP-Lab/BrainPy/pull/162)
- feat: add conv & pooling nodes by
  [\@ztqakita](https://github.com/ztqakita) in
  [#161](https://github.com/PKU-NIP-Lab/BrainPy/pull/161)
- fix: update setup.py by [\@ztqakita](https://github.com/ztqakita) in
  [#163](https://github.com/PKU-NIP-Lab/BrainPy/pull/163)
- update setup.py by [\@chaoming0625](https://github.com/chaoming0625)
  in [#165](https://github.com/PKU-NIP-Lab/BrainPy/pull/165)
- fix: change trigger condition by
  [\@ztqakita](https://github.com/ztqakita) in
  [#166](https://github.com/PKU-NIP-Lab/BrainPy/pull/166)
- fix: add build_conn() function by
  [\@ztqakita](https://github.com/ztqakita) in
  [#164](https://github.com/PKU-NIP-Lab/BrainPy/pull/164)
- update synapses by [\@chaoming0625](https://github.com/chaoming0625)
  in [#167](https://github.com/PKU-NIP-Lab/BrainPy/pull/167)
- get the deserved name: brainpy by
  [\@chaoming0625](https://github.com/chaoming0625) in
  [#168](https://github.com/PKU-NIP-Lab/BrainPy/pull/168)
- update tests by [\@chaoming0625](https://github.com/chaoming0625) in
  [#169](https://github.com/PKU-NIP-Lab/BrainPy/pull/169)

**Full Changelog**:
[V2.1.4\...V2.1.5](https://github.com/PKU-NIP-Lab/BrainPy/compare/V2.1.4...V2.1.5)

### Version 2.1.4 (2022.04.04)

#### What\'s Changed

- fix doc parsing bug by
  [\@chaoming0625](https://github.com/chaoming0625) in
  [#127](https://github.com/PKU-NIP-Lab/BrainPy/pull/127)
- Update overview_of_dynamic_model.ipynb by
  [\@c-xy17](https://github.com/c-xy17) in
  [#129](https://github.com/PKU-NIP-Lab/BrainPy/pull/129)
- Reorganization of `brainpylib.custom_op` and adding interface in
  `brainpy.math` by [\@ztqakita](https://github.com/ztqakita) in
  [#128](https://github.com/PKU-NIP-Lab/BrainPy/pull/128)
- Fix: modify `register_op` and brainpy.math interface by
  [\@ztqakita](https://github.com/ztqakita) in
  [#130](https://github.com/PKU-NIP-Lab/BrainPy/pull/130)
- new features about RNN training and delay differential equations by
  [\@chaoming0625](https://github.com/chaoming0625) in
  [#132](https://github.com/PKU-NIP-Lab/BrainPy/pull/132)
- Fix [#123](https://github.com/PKU-NIP-Lab/BrainPy/issues/123): Add
  low-level operators docs and modify register_op by
  [\@ztqakita](https://github.com/ztqakita) in
  [#134](https://github.com/PKU-NIP-Lab/BrainPy/pull/134)
- feat: add generate_changelog by
  [\@ztqakita](https://github.com/ztqakita) in
  [#135](https://github.com/PKU-NIP-Lab/BrainPy/pull/135)
- fix [#133](https://github.com/PKU-NIP-Lab/BrainPy/issues/133),
  support batch size training with offline algorithms by
  [\@chaoming0625](https://github.com/chaoming0625) in
  [#136](https://github.com/PKU-NIP-Lab/BrainPy/pull/136)
- fix [#84](https://github.com/PKU-NIP-Lab/BrainPy/issues/84): support
  online training algorithms by
  [\@chaoming0625](https://github.com/chaoming0625) in
  [#137](https://github.com/PKU-NIP-Lab/BrainPy/pull/137)
- feat: add the batch normalization node by
  [\@c-xy17](https://github.com/c-xy17) in
  [#138](https://github.com/PKU-NIP-Lab/BrainPy/pull/138)
- fix: fix shape checking error by
  [\@chaoming0625](https://github.com/chaoming0625) in
  [#139](https://github.com/PKU-NIP-Lab/BrainPy/pull/139)
- solve [#131](https://github.com/PKU-NIP-Lab/BrainPy/issues/131),
  support efficient synaptic computation for special connection types
  by [\@chaoming0625](https://github.com/chaoming0625) in
  [#140](https://github.com/PKU-NIP-Lab/BrainPy/pull/140)
- feat: update the API and test for batch normalization by
  [\@c-xy17](https://github.com/c-xy17) in
  [#142](https://github.com/PKU-NIP-Lab/BrainPy/pull/142)
- Node is default trainable by
  [\@chaoming0625](https://github.com/chaoming0625) in
  [#143](https://github.com/PKU-NIP-Lab/BrainPy/pull/143)
- Updates training apis and docs by
  [\@chaoming0625](https://github.com/chaoming0625) in
  [#145](https://github.com/PKU-NIP-Lab/BrainPy/pull/145)
- fix: add dependencies and update version by
  [\@ztqakita](https://github.com/ztqakita) in
  [#147](https://github.com/PKU-NIP-Lab/BrainPy/pull/147)
- update requirements by
  [\@chaoming0625](https://github.com/chaoming0625) in
  [#146](https://github.com/PKU-NIP-Lab/BrainPy/pull/146)
- data pass of the Node is default SingleData by
  [\@chaoming0625](https://github.com/chaoming0625) in
  [#148](https://github.com/PKU-NIP-Lab/BrainPy/pull/148)

**Full Changelog**:
[V2.1.3\...V2.1.4](https://github.com/PKU-NIP-Lab/BrainPy/compare/V2.1.3...V2.1.4)

### Version 2.1.3 (2022.03.27)

This release improves the functionality and usability of BrainPy. Core
changes include

- support customization of low-level operators by using Numba
- fix bugs

#### What\'s Changed

- Provide custom operators written in numba for jax jit by
  [\@ztqakita](https://github.com/ztqakita) in
  [#122](https://github.com/PKU-NIP-Lab/BrainPy/pull/122)
- fix DOGDecay bugs, add more features by
  [\@chaoming0625](https://github.com/chaoming0625) in
  [#124](https://github.com/PKU-NIP-Lab/BrainPy/pull/124)
- fix bugs by [\@chaoming0625](https://github.com/chaoming0625) in
  [#126](https://github.com/PKU-NIP-Lab/BrainPy/pull/126)

**Full Changelog** :
[V2.1.2\...V2.1.3](https://github.com/PKU-NIP-Lab/BrainPy/compare/V2.1.2...V2.1.3)

### Version 2.1.2 (2022.03.23)

This release improves the functionality and usability of BrainPy. Core
changes include

- support rate-based whole-brain modeling
- add more neuron models, including rate neurons/synapses
- support Python 3.10
- improve delays etc. APIs

#### What\'s Changed

- fix matplotlib dependency on \"brainpy.analysis\" module by
  [\@chaoming0625](https://github.com/chaoming0625) in
  [#110](https://github.com/PKU-NIP-Lab/BrainPy/pull/110)
- Sync master to brainpy-2.x branch by
  [\@ztqakita](https://github.com/ztqakita) in
  [#111](https://github.com/PKU-NIP-Lab/BrainPy/pull/111)
- add py3.6 test & delete multiple macos env by
  [\@ztqakita](https://github.com/ztqakita) in
  [#112](https://github.com/PKU-NIP-Lab/BrainPy/pull/112)
- Modify ci by [\@ztqakita](https://github.com/ztqakita) in
  [#113](https://github.com/PKU-NIP-Lab/BrainPy/pull/113)
- Add py3.10 test by [\@ztqakita](https://github.com/ztqakita) in
  [#115](https://github.com/PKU-NIP-Lab/BrainPy/pull/115)
- update python version by
  [\@chaoming0625](https://github.com/chaoming0625) in
  [#114](https://github.com/PKU-NIP-Lab/BrainPy/pull/114)
- add brainpylib mac py3.10 by
  [\@ztqakita](https://github.com/ztqakita) in
  [#116](https://github.com/PKU-NIP-Lab/BrainPy/pull/116)
- Enhance measure/input/brainpylib by
  [\@chaoming0625](https://github.com/chaoming0625) in
  [#117](https://github.com/PKU-NIP-Lab/BrainPy/pull/117)
- fix [#105](https://github.com/PKU-NIP-Lab/BrainPy/issues/105): Add
  customize connections docs by
  [\@ztqakita](https://github.com/ztqakita) in
  [#118](https://github.com/PKU-NIP-Lab/BrainPy/pull/118)
- fix bugs by [\@chaoming0625](https://github.com/chaoming0625) in
  [#119](https://github.com/PKU-NIP-Lab/BrainPy/pull/119)
- Whole brain modeling by
  [\@chaoming0625](https://github.com/chaoming0625) in
  [#121](https://github.com/PKU-NIP-Lab/BrainPy/pull/121)

**Full Changelog**:
[V2.1.1\...V2.1.2](https://github.com/PKU-NIP-Lab/BrainPy/compare/V2.1.1...V2.1.2)

### Version 2.1.1 (2022.03.18)

This release continues to update the functionality of BrainPy. Core
changes include

- numerical solvers for fractional differential equations
- more standard `brainpy.nn` interfaces

#### New Features

-

Numerical solvers for fractional differential equations

:   -   `brainpy.fde.CaputoEuler`
-   `brainpy.fde.CaputoL1Schema`
-   `brainpy.fde.GLShortMemory`

-

Fractional neuron models

:   -   `brainpy.dyn.FractionalFHR`
-   `brainpy.dyn.FractionalIzhikevich`

- support `shared_kwargs` in [RNNTrainer]{.title-ref} and
  [RNNRunner]{.title-ref}

### Version 2.1.0 (2022.03.14)

#### Highlights

We are excited to announce the release of BrainPy 2.1.0. This release is
composed of nearly 270 commits since 2.0.2, made by [Chaoming
Wang](https://github.com/chaoming0625), [Xiaoyu
Chen](mailto:c-xy17@tsinghua.org.cn), and [Tianqiu
Zhang](mailto:tianqiuakita@gmail.com) .

BrainPy 2.1.0 updates are focused on improving usability, functionality,
and stability of BrainPy. Highlights of version 2.1.0 include:

- New module `brainpy.dyn` for dynamics building and simulation. It is
  composed of many neuron models, synapse models, and others.
- New module `brainpy.nn` for neural network building and training. It
  supports to define reservoir models, artificial neural networks,
  ridge regression training, and back-propagation through time
  training.
- New module `brainpy.datasets` for convenient dataset construction
  and initialization.
- New module `brainpy.integrators.dde` for numerical integration of
  delay differential equations.
- Add more numpy-like operators in `brainpy.math` module.
- Add automatic continuous integration on Linux, Windows, and MacOS
  platforms.
- Fully update brainpy documentation.
- Fix bugs on `brainpy.analysis` and `brainpy.math.autograd`

#### Incompatible changes

- Remove `brainpy.math.numpy` module.
- Remove numba requirements
- Remove matplotlib requirements
- Remove [steps]{.title-ref} in `brainpy.dyn.DynamicalSystem`
- Remove travis CI

#### New Features

- `brainpy.ddeint` for numerical integration of delay differential
  equations, the supported methods include: - Euler - MidPoint -
  Heun2 - Ralston2 - RK2 - RK3 - Heun3 - Ralston3 - SSPRK3 - RK4 -
  Ralston4 - RK4Rule38

-

set default int/float/complex types

:   -   `brainpy.math.set_dfloat()`
-   `brainpy.math.set_dint()`
-   `brainpy.math.set_dcomplex()`

-

Delay variables

:   -   `brainpy.math.FixedLenDelay`
-   `brainpy.math.NeutralDelay`

-

Dedicated operators

:   -   `brainpy.math.sparse_matmul()`

- More numpy-like operators

- Neural network building `brainpy.nn`

- Dynamics model building and simulation `brainpy.dyn`

### Version 2.0.2 (2022.02.11)

There are important updates by [Chaoming
Wang](https://github.com/chaoming0625) in BrainPy 2.0.2.

- provide `pre2post_event_prod` operator
- support array creation from a list/tuple of JaxArray in
  `brainpy.math.asarray` and `brainpy.math.array`
- update `brainpy.ConstantDelay`, add `.latest` and `.oldest`
  attributes
- add `brainpy.IntegratorRunner` support for efficient simulation of
  brainpy integrators
- support auto finding of RandomState when JIT SDE integrators
- fix bugs in SDE `exponential_euler` method
- move `parallel` running APIs into `brainpy.simulation`
- add `brainpy.math.syn2post_mean`, `brainpy.math.syn2post_softmax`,
  `brainpy.math.pre2post_mean` and `brainpy.math.pre2post_softmax`
  operators

### Version 2.0.1 (2022.01.31)

Today we release BrainPy 2.0.1. This release is composed of over 70
commits since 2.0.0, made by [Chaoming
Wang](https://github.com/chaoming0625), [Xiaoyu
Chen](mailto:c-xy17@tsinghua.org.cn), and [Tianqiu
Zhang](mailto:tianqiuakita@gmail.com) .

BrainPy 2.0.0 updates are focused on improving documentation and
operators. Core changes include:

- Improve `brainpylib` operators
- Complete documentation for programming system
- Add more numpy APIs
- Add `jaxfwd` in autograd module
- And other changes

### Version 2.0.0.1 (2022.01.05)

- Add progress bar in `brainpy.StructRunner`

### Version 2.0.0 (2021.12.31)

Start a new version of BrainPy.

#### Highlight

We are excited to announce the release of BrainPy 2.0.0. This release is
composed of over 260 commits since 1.1.7, made by [Chaoming
Wang](https://github.com/chaoming0625), [Xiaoyu
Chen](mailto:c-xy17@tsinghua.org.cn), and [Tianqiu
Zhang](mailto:tianqiuakita@gmail.com) .

BrainPy 2.0.0 updates are focused on improving performance, usability
and consistence of BrainPy. All the computations are migrated into JAX.
Model `building`, `simulation`, `training` and `analysis` are all based
on JAX. Highlights of version 2.0.0 include:

- [brainpylib](https://pypi.org/project/brainpylib/) are provided to
  dedicated operators for brain dynamics programming
- Connection APIs in `brainpy.conn` module are more efficient.
- Update analysis tools for low-dimensional and high-dimensional
  systems in `brainpy.analysis` module.
- Support more general Exponential Euler methods based on automatic
  differentiation.
- Improve the usability and consistence of `brainpy.math` module.
- Remove JIT compilation based on Numba.
- Separate brain building with brain simulation.

#### Incompatible changes

- remove `brainpy.math.use_backend()`
- remove `brainpy.math.numpy` module
- no longer support `.run()` in `brainpy.DynamicalSystem` (see New
  Features)
- remove `brainpy.analysis.PhasePlane` (see New Features)
- remove `brainpy.analysis.Bifurcation` (see New Features)
- remove `brainpy.analysis.FastSlowBifurcation` (see New Features)

#### New Features

-

Exponential Euler method based on automatic differentiation

:   -   `brainpy.ode.ExpEulerAuto`

-

Numerical optimization based low-dimensional analyzers:

:   -   `brainpy.analysis.PhasePlane1D`
-   `brainpy.analysis.PhasePlane2D`
-   `brainpy.analysis.Bifurcation1D`
-   `brainpy.analysis.Bifurcation2D`
-   `brainpy.analysis.FastSlow1D`
-   `brainpy.analysis.FastSlow2D`

-

Numerical optimization based high-dimensional analyzer:

:   -   `brainpy.analysis.SlowPointFinder`

-

Dedicated operators in `brainpy.math` module:

:   -   `brainpy.math.pre2post_event_sum`
-   `brainpy.math.pre2post_sum`
-   `brainpy.math.pre2post_prod`
-   `brainpy.math.pre2post_max`
-   `brainpy.math.pre2post_min`
-   `brainpy.math.pre2syn`
-   `brainpy.math.syn2post`
-   `brainpy.math.syn2post_prod`
-   `brainpy.math.syn2post_max`
-   `brainpy.math.syn2post_min`

-

Conversion APIs in `brainpy.math` module:

:   -   `brainpy.math.as_device_array()`
-   `brainpy.math.as_variable()`
-   `brainpy.math.as_jaxarray()`

-

New autograd APIs in `brainpy.math` module:

:   -   `brainpy.math.vector_grad()`

-

Simulation runners:

:   -   `brainpy.ReportRunner`
-   `brainpy.StructRunner`
-   `brainpy.NumpyRunner`

-

Commonly used models in `brainpy.models` module

:   -   `brainpy.models.LIF`
-   `brainpy.models.Izhikevich`
-   `brainpy.models.AdExIF`
-   `brainpy.models.SpikeTimeInput`
-   `brainpy.models.PoissonInput`
-   `brainpy.models.DeltaSynapse`
-   `brainpy.models.ExpCUBA`
-   `brainpy.models.ExpCOBA`
-   `brainpy.models.AMPA`
-   `brainpy.models.GABAa`

- Naming cache clean: `brainpy.clear_name_cache`

- add safe in-place operations of `update()` method and `.value`
  assignment for JaxArray

#### Documentation

- Complete tutorials for quickstart
- Complete tutorials for dynamics building
- Complete tutorials for dynamics simulation
- Complete tutorials for dynamics training
- Complete tutorials for dynamics analysis
- Complete tutorials for API documentation

## brainpy 1.1.x

If you are using `brainpy==1.x`, you can find *documentation*,
*examples*, and *models* through the following links:

- **Documentation:** <https://brainpy.readthedocs.io/en/brainpy-1.x/>
- **Examples from papers**:
  <https://brainpy-examples.readthedocs.io/en/brainpy-1.x/>
- **Canonical brain models**:
  <https://brainmodels.readthedocs.io/en/brainpy-1.x/>

### Version 1.1.7 (2021.12.13)

- fix bugs on `numpy_array()` conversion in
  [brainpy.math.utils]{.title-ref} module

### Version 1.1.5 (2021.11.17)

**API changes:**

- fix bugs on ndarray import in [brainpy.base.function.py]{.title-ref}
- convenient \'get_param\' interface
  [brainpy.simulation.layers]{.title-ref}
- add more weight initialization methods

**Doc changes:**

- add more examples in README

### Version 1.1.4

**API changes:**

- add `.struct_run()` in DynamicalSystem
- add `numpy_array()` conversion in [brainpy.math.utils]{.title-ref}
  module
- add `Adagrad`, `Adadelta`, `RMSProp` optimizers
- remove [setting]{.title-ref} methods in
  [brainpy.math.jax]{.title-ref} module
- remove import jax in [brainpy.\_\_init\_\_.py]{.title-ref} and
  enable jax setting, including
    - `enable_x64()`
    - `set_platform()`
    - `set_host_device_count()`
- enable `b=None` as no bias in
  [brainpy.simulation.layers]{.title-ref}
- set [int\_]{.title-ref} and [float\_]{.title-ref} as default 32 bits
- remove `dtype` setting in Initializer constructor

**Doc changes:**

- add `optimizer` in \"Math Foundation\"
- add `dynamics training` docs
- improve others

### Version 1.1.3

- fix bugs of JAX parallel API imports
- fix bugs of [post_slice]{.title-ref} structure construction
- update docs

### Version 1.1.2

- add `pre2syn` and `syn2post` operators
- add [verbose]{.title-ref} and [check]{.title-ref} option to
  `Base.load_states()`
- fix bugs on JIT DynamicalSystem (numpy backend)

### Version 1.1.1

- fix bugs on symbolic analysis: model trajectory
- change [absolute]{.title-ref} access in the variable saving and
  loading to the [relative]{.title-ref} access
- add UnexpectedTracerError hints in JAX transformation functions

### Version 1.1.0 (2021.11.08)

This package releases a new version of BrainPy.

Highlights of core changes:

#### `math` module

- support numpy backend
- support JAX backend
- support `jit`, `vmap` and `pmap` on class objects on JAX backend
- support `grad`, `jacobian`, `hessian` on class objects on JAX
  backend
- support `make_loop`, `make_while`, and `make_cond` on JAX backend
- support `jit` (based on numba) on class objects on numpy backend
- unified numpy-like ndarray operation APIs
- numpy-like random sampling APIs
- FFT functions
- gradient descent optimizers
- activation functions
- loss function
- backend settings

#### `base` module

- `Base` for whole Version ecosystem
- `Function` to wrap functions
- `Collector` and `TensorCollector` to collect variables, integrators,
  nodes and others

#### `integrators` module

- class integrators for ODE numerical methods
- class integrators for SDE numerical methods

#### `simulation` module

- support modular and composable programming
- support multi-scale modeling
- support large-scale modeling
- support simulation on GPUs
- fix bugs on `firing_rate()`
- remove `_i` in `update()` function, replace `_i` with `_dt`, meaning
  the dynamic system has the canonic equation form of
  $dx/dt = f(x, t, dt)$
- reimplement the `input_step` and `monitor_step` in a more intuitive
  way
- support to set [dt]{.title-ref} in the single object level (i.e.,
  single instance of DynamicSystem)
- common used DNN layers
- weight initializations
- refine synaptic connections

## brainpy 1.0.x

### Version 1.0.3 (2021.08.18)

Fix bugs on

- firing rate measurement
- stability analysis

### Version 1.0.2

This release continues to improve the user-friendliness.

Highlights of core changes:

- Remove support for Numba-CUDA backend
- Super initialization [super(XXX, self).\_\_init\_\_()]{.title-ref}
  can be done at anywhere (not required to add at the bottom of the
  [\_\_init\_\_()]{.title-ref} function).
- Add the output message of the step function running error.
- More powerful support for Monitoring
- More powerful support for running order scheduling
- Remove [unsqueeze()]{.title-ref} and [squeeze()]{.title-ref}
  operations in `brainpy.ops`
- Add [reshape()]{.title-ref} operation in `brainpy.ops`
- Improve docs for numerical solvers
- Improve tests for numerical solvers
- Add keywords checking in ODE numerical solvers
- Add more unified operations in brainpy.ops
- Support \"@every\" in steps and monitor functions
- Fix ODE solver bugs for class bounded function
- Add build phase in Monitor

### Version 1.0.1

- Fix bugs

### Version 1.0.0

- **NEW VERSION OF BRAINPY**
- Change the coding style into the object-oriented programming
- Systematically improve the documentation

## brainpy 0.x

### Version 0.3.5

- Add \'timeout\' in sympy solver in neuron dynamics analysis
- Reconstruct and generalize phase plane analysis
- Generalize the repeat mode of `Network` to different running
  duration between two runs
- Update benchmarks
- Update detailed documentation

### Version 0.3.1

- Add a more flexible way for NeuState/SynState initialization
- Fix bugs of \"is_multi_return\"
- Add \"hand_overs\", \"requires\" and \"satisfies\".
- Update documentation
- Auto-transform [range]{.title-ref} to [numba.prange]{.title-ref}
- Support [\_obj_i]{.title-ref}, [\_pre_i]{.title-ref},
  [\_post_i]{.title-ref} for more flexible operation in scalar-based
  models

### Version 0.3.0

#### Computation API

- Rename \"brainpy.numpy\" to \"brainpy.backend\"
- Delete \"pytorch\", \"tensorflow\" backends
- Add \"numba\" requirement
- Add GPU support

#### Profile setting

- Delete \"backend\" profile setting, add \"jit\"

#### Core systems

- Delete \"autopepe8\" requirement
- Delete the format code prefix
- Change keywords \"\_[t](), \_[dt](), \_[i]()\" to \"\_t, \_dt, \_i\"
- Change the \"ST\" declaration out of \"requires\"
- Add \"repeat\" mode run in Network
- Change \"vector-based\" to \"mode\" in NeuType and SynType
  definition

#### Package installation

- Remove \"pypi\" installation, installation now only rely on
  \"conda\"

### Version 0.2.4

#### API changes

- Fix bugs

### Version 0.2.3

#### API changes

- Add \"animate_1D\" in `visualization` module
- Add \"PoissonInput\", \"SpikeTimeInput\" and \"FreqInput\" in
  `inputs` module
- Update phase_portrait_analyzer.py

#### Models and examples

- Add CANN examples

### Version 0.2.2

#### API changes

- Redesign visualization
- Redesign connectivity
- Update docs

### Version 0.2.1

#### API changes

- Fix bugs in [numba import]{.title-ref}
- Fix bugs in [numpy]{.title-ref} mode with [scalar]{.title-ref} model

### Version 0.2.0

#### API changes

- For computation: `numpy`, `numba`
- For model definition: `NeuType`, `SynConn`
- For model running: `Network`, `NeuGroup`, `SynConn`, `Runner`
- For numerical integration: `integrate`, `Integrator`, `DiffEquation`
- For connectivity: `One2One`, `All2All`, `GridFour`, `grid_four`,
  `GridEight`, `grid_eight`, `GridN`, `FixedPostNum`, `FixedPreNum`,
  `FixedProb`, `GaussianProb`, `GaussianWeight`, `DOG`
- For visualization: `plot_value`, `plot_potential`, `plot_raster`,
  `animation_potential`
- For measurement: `cross_correlation`, `voltage_fluctuation`,
  `raster_plot`, `firing_rate`
- For inputs: `constant_current`, `spike_current`, `ramp_current`.

#### Models and examples

- Neuron models: `HH model`, `LIF model`, `Izhikevich model`
- Synapse models: `AMPA`, `GABA`, `NMDA`, `STP`, `GapJunction`
- Network models: `gamma oscillation`
