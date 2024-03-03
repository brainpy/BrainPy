# Release notes (``brainpy``)

## brainpy 2.2.x

BrainPy 2.2.x is a complete re-design of the framework, tackling the
shortcomings of brainpy 2.1.x generation, effectively bringing it to
research needs and standards.

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
