

This release continues to add the supports for brain-inspired computation.




## New Features

1. ``brainpy.encoding`` module for encoding rate values into spike trains. Currently, we support

   - `brainpy.encoding.LatencyEncoder`
   - `brainpy.encoding.PoissonEncoder`
   - `brainpy.encoding.WeightedPhaseEncoder`

2. ``brainpy.math.surrogate`` module for surrogate gradient functions. Currently, we support

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

3. ``brainpy.dyn.transfom`` module for transforming a ``DynamicalSystem`` instance to a callable ``BrainPyObject``. Specifically, we provide

    - `LoopOverTime` for unrolling a dynamical system over time.
    - `NoSharedArg` for removing the dependency of shared arguments.

4. Change all ``brainpy.Runner`` as the subclasses of ``BrainPyObject``, which means that all ``brainpy.Runner`` can be used as a part of the high-level program or transformation. 

5. Enable the continuous running of a differential equation (ODE, SDE, FDE, DDE, etc.) with `IntegratorRunner`. For example,

   ```python
   import brainpy as bp
   
   # differential equation
   a, b, tau = 0.7, 0.8, 12.5
   dV = lambda V, t, w, Iext: V - V * V * V / 3 - w + Iext
   dw = lambda w, t, V: (V + a - b * w) / tau
   fhn = bp.odeint(bp.JointEq([dV, dw]), method='rk4', dt=0.1)
   
   # differential integrator runner
   runner = bp.IntegratorRunner(fhn,
                                monitors=['V', 'w'],
                                inits=[1., 1.])
   
   # run 1
   Iext, duration = bp.inputs.section_input([0., 1., 0.5], [200, 200, 200], return_length=True)
   runner.run(duration, dyn_args=dict(Iext=Iext))
   bp.visualize.line_plot(runner.mon.ts, runner.mon['V'], legend='V')
   
   # run 2
   Iext, duration = bp.inputs.section_input([0.5], [200], return_length=True)
   runner.run(duration, dyn_args=dict(Iext=Iext))
   bp.visualize.line_plot(runner.mon.ts, runner.mon['V'], legend='V-run2', show=True)
   
   ```

6. New transformation function ``brainpy.math.to_dynsys`` supports to transform a pure Python function into a ``DynamicalSystem``. This will be useful when running a `DynamicalSystem` with arbitrary customized inputs.

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

7. 

8. 





## Deprecations


1. ``func_monitors`` is no longer supported in all ``brainpy.Runner`` subclasses. We will remove its supports since version 2.4.0. Instead, monitoring with a dict of callable functions can be set  in ``monitors``. For example, 


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

2. ``func_inputs`` is no longer supported in all ``brainpy.Runner`` subclasses. Instead, giving inputs with a callable function should be done with ``inputs``. 

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

3. ``inputs_are_batching`` is deprecated in ``predict()``/``.run()`` of all ``brainpy.Runner`` subclasses. 

4. ``args`` and ``dyn_args`` are now  deprecated in ``IntegratorRunner``. Instead, users should specify ``args`` and ``dyn_args`` when using ``IntegratorRunner.run()`` function.  

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







