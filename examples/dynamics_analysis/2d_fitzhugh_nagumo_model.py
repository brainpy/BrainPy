# -*- coding: utf-8 -*-


import brainpy as bp
import brainpy.math as bm

bp.math.enable_x64()


class FitzHughNagumoModel(bp.dyn.DynamicalSystem):
  def __init__(self, method='exp_auto'):
    super(FitzHughNagumoModel, self).__init__()

    # parameters
    self.a = 0.7
    self.b = 0.8
    self.tau = 12.5

    # variables
    self.V = bm.Variable(bm.zeros(1))
    self.w = bm.Variable(bm.zeros(1))
    self.Iext = bm.Variable(bm.zeros(1))

    # functions
    def dV(V, t, w, Iext=0.):
      dV = V - V * V * V / 3 - w + Iext
      return dV

    def dw(w, t, V, a=0.7, b=0.8):
      dw = (V + a - b * w) / self.tau
      return dw

    self.int_V = bp.odeint(dV, method=method)
    self.int_w = bp.odeint(dw, method=method)

  def update(self, tdi):
    t, dt = tdi['t'], tdi['dt']
    self.V.value = self.int_V(self.V, t, self.w, self.Iext, dt)
    self.w.value = self.int_w(self.w, t, self.V, self.a, self.b, dt)
    self.Iext[:] = 0.


model = FitzHughNagumoModel()

# simulation
runner = bp.dyn.DSRunner(model, monitors=['V', 'w'], inputs=['Iext', 0.])
runner.run(100.)

bp.visualize.line_plot(runner.mon.ts, runner.mon.V, legend='V')
bp.visualize.line_plot(runner.mon.ts, runner.mon.w, legend='w', show=True)

# phase plane analysis
pp = bp.analysis.PhasePlane2D(
  model=model,
  target_vars={'V': [-3, 3], 'w': [-1, 3]},
  pars_update={'Iext': 1.},
  resolutions=0.01,
)
pp.plot_vector_field()
pp.plot_nullcline(coords={'V': 'w-V'})
pp.plot_fixed_point()
pp.plot_trajectory(initials={'V': [0.], 'w': [1.]},
                   duration=100, plot_durations=[50, 100])
pp.show_figure()


# codimension 1 bifurcation
bif = bp.analysis.Bifurcation2D(
  model=model,
  target_vars={'V': [-3., 3.], 'w': [-1, 3.]},
  target_pars={'Iext': [-1., 2.]},
  resolutions=0.01
)
bif.plot_bifurcation(num_par_segments=2)
bif.plot_limit_cycle_by_sim()
bif.show_figure()
