# Change from Version 2.3.0 to Version 2.3.1



This release (under the release branch of ``brainpy=2.3.x``) continues to add supports for brain-inspired computation.



```python
import brainpy as bp
import brainpy.math as bm
```



## Backwards Incompatible Changes



#### 1. Error: module 'brainpy' has no attribute 'datasets'

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



#### 2. Error: DSTrainer must receive an instance with BatchingMode

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



#### 3. Error: inputs_are_batching is no longer supported. 

This is because if the training target is in ``batching`` mode, this has already indicated that the inputs should be batching. 

Simple remove the ``inputs_are_batching`` from your functional call of ``.predict()`` will solve the issue. 





## New Features



### 1. ``brainpy.math`` module upgrade

#### ``brainpy.math.surrogate`` module for surrogate gradient functions.

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



#### New transformation function ``brainpy.math.to_dynsys``

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



#### Default data types 

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



#### Environment context manager 

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



### 2. ``brainpy.dyn`` module



#### ``brainpy.dyn.transfom`` module for transforming a ``DynamicalSystem`` instance to a callable ``BrainPyObject``. 

Specifically, we provide

- `LoopOverTime` for unrolling a dynamical system over time.
- `NoSharedArg` for removing the dependency of shared arguments.





### 3. Running supports in BrainPy



#### All ``brainpy.Runner`` now are subclasses of ``BrainPyObject``

This means that all ``brainpy.Runner`` can be used as a part of the high-level program or transformation. 



#### Enable the continuous running of a differential equation (ODE, SDE, FDE, DDE, etc.) with `IntegratorRunner`. 

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



#### Enable call a customized function during fitting of ``brainpy.BPTrainer``.

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



#### Enable data with ``data_first_axis`` format when predicting or fitting in a ``brainpy.DSRunner`` and ``brainpy.DSTrainer``. 

Previous version of BrainPy only supports data with the batch dimension at the first axis. Currently, ``brainpy.DSRunner`` and ``brainpy.DSTrainer`` can support the data with the time dimension at the first axis. This can be set through ``data_first_axis='T'`` when initializing a runner or trainer. 

```python
runner = bp.DSRunner(..., data_first_axis='T')
trainer = bp.DSTrainer(..., data_first_axis='T')
```



### 4. Utility in BrainPy



#### ``brainpy.encoding`` module for encoding rate values into spike trains

 Currently, we support

- `brainpy.encoding.LatencyEncoder`
- `brainpy.encoding.PoissonEncoder`
- `brainpy.encoding.WeightedPhaseEncoder`



#### ``brainpy.checkpoints`` module for model state serialization. 

This version of BrainPy supports to save a checkpoint of the model into the physical disk. Inspired from the Flax API, we provide the following checkpoint APIs:

- ``brainpy.checkpoints.save()`` for saving a checkpoint of the model.
- ``brainpy.checkpoints.multiprocess_save()`` for saving a checkpoint of the model in multi-process environment.
- ``brainpy.checkpoints.load()`` for loading the last or best checkpoint from the given checkpoint path.
- ``brainpy.checkpoints.load_latest()`` for retrieval the path of the latest checkpoint in a directory.





## Deprecations



### 1. Deprecations in the running supports of BrainPy

#### ``func_monitors`` is no longer supported in all ``brainpy.Runner`` subclasses.

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



#### ``func_inputs`` is no longer supported in all ``brainpy.Runner`` subclasses.

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



#### ``inputs_are_batching`` is deprecated. 

``inputs_are_batching`` is deprecated in ``predict()``/``.run()`` of all ``brainpy.Runner`` subclasses. 



#### ``args`` and ``dyn_args`` are now  deprecated in ``IntegratorRunner``.

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



###  2. Deprecations in ``brainpy.math`` module

#### `ditype()` and `dftype()` are deprecated.

`brainpy.math.ditype()` and `brainpy.math.dftype()` are deprecated. Using `brainpy.math.int_` and `brainpy.math.float()` instead. 



#### ``brainpy.modes`` module is  now moved into ``brainpy.math``

The correspondences are listed as the follows:

- ``brainpy.modes.Mode``  => ``brainpy.math.Mode``
- ``brainpy.modes.NormalMode `` => ``brainpy.math.NonBatchingMode`` 
- ``brainpy.modes.BatchingMode `` => ``brainpy.math.BatchingMode`` 
- ``brainpy.modes.TrainingMode `` => ``brainpy.math.TrainingMode`` 
- ``brainpy.modes.normal `` => ``brainpy.math.nonbatching_mode`` 
- ``brainpy.modes.batching `` => ``brainpy.math.batching_mode`` 
- ``brainpy.modes.training `` => ``brainpy.math.training_mode`` 
