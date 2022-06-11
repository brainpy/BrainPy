# -*- coding: utf-8 -*-

import brainpy as bp
import brainpy.math as bm

I = 5.
model = bp.dyn.neurons.HH(1)
runner = bp.dyn.DSRunner(model, inputs=('input', I), monitors=['V'])
runner.run(100)
bp.visualize.line_plot(runner.mon.ts, runner.mon.V, legend='V', show=True)

# analysis
model = bp.dyn.neurons.HH(1, method='euler')
finder = bp.analysis.SlowPointFinder(
  model,
  inputs=(model.input, I),
  included_vars={'V': model.V,
                 'm': model.m,
                 'h': model.h,
                 'n': model.n},
  dt=1.
)
candidates = {'V': bm.random.normal(0., 5., (1000, model.num)) - 50.,
              'm': bm.random.random((1000, model.num)),
              'h': bm.random.random((1000, model.num)),
              'n': bm.random.random((1000, model.num))}
finder.find_fps_with_opt_solver(candidates=candidates)
finder.filter_loss(1e-7)
finder.keep_unique(tolerance=1e-1)
print('fixed_points: ', finder.fixed_points)
print('losses:', finder.losses)
if finder.num_fps > 0:
  jac = finder.compute_jacobians(finder.fixed_points, plot=True)

# verify
for i in range(finder.num_fps):
  model = bp.dyn.neurons.HH(1)
  model.V[:] = finder._fixed_points['V'][i]
  model.m[:] = finder._fixed_points['m'][i]
  model.h[:] = finder._fixed_points['h'][i]
  model.n[:] = finder._fixed_points['n'][i]
  runner = bp.dyn.DSRunner(model, inputs=(model.input, I), monitors=['V'])
  runner.run(100)
  bp.visualize.line_plot(runner.mon.ts, runner.mon.V, legend='V', title=f'FP {i}', show=True)
