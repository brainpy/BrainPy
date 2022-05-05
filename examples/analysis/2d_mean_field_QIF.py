import brainpy as bp
import brainpy.math as bm
bp.math.enable_x64()


class MeanFieldQIF(bp.dyn.DynamicalSystem):
  """A mean-field model of a quadratic integrate-and-fire neuron population.

  References
  ----------
  .. [1] E. Montbrió, D. Pazó, A. Roxin (2015) Macroscopic description for
         networks of spiking neurons. Physical Review X, 5:021028,
         https://doi.org/10.1103/PhysRevX.5.021028.
  """

  def __init__(self, method='exp_auto'):
    super(MeanFieldQIF, self).__init__()

    # parameters
    self.tau = 1.  # the population time constant
    self.eta = -5.0  # the mean of a Lorenzian distribution over the neural excitability in the population
    self.delta = 1.0  # the half-width at half maximum of the Lorenzian distribution over the neural excitability
    self.J = 15.  # the strength of the recurrent coupling inside the population

    # variables
    self.r = bm.Variable(bm.ones(1))
    self.v = bm.Variable(bm.ones(1))
    self.Iext = bm.Variable(bm.zeros(1))

    # functions
    def dr(r, t, v, delta=1.0):
      return (delta / (bm.pi * self.tau) + 2. * r * v) / self.tau

    def dv(v, t, r, Iext=0., eta=-5.0):
      return (v ** 2 + eta + Iext + self.J * r * self.tau -
              (bm.pi * r * self.tau) ** 2) / self.tau

    self.int_r = bp.odeint(dr, method=method)
    self.int_v = bp.odeint(dv, method=method)

  def update(self, t, dt):
    self.r.value = self.int_r(self.r, t, self.v, self.delta, dt)
    self.v.value = self.int_v(self.v, t, self.r, self.Iext, self.eta, dt)
    self.Iext[:] = 0.


qif = MeanFieldQIF()


# simulation
runner = bp.dyn.DSRunner(qif, inputs=['Iext', 1.], monitors=['r', 'v'])
runner.run(100.)
bp.visualize.line_plot(runner.mon.ts, runner.mon.r, legend='r')
bp.visualize.line_plot(runner.mon.ts, runner.mon.v, legend='v', show=True)


# phase plane analysis
pp = bp.analysis.PhasePlane2D(
  qif,
  target_vars={'r': [0., 4.], 'v': [-3., 3.]},
  resolutions=0.01
)
pp.plot_vector_field()
pp.plot_nullcline()
pp.plot_fixed_point()
pp.show_figure()


# codimension 1 bifurcation
bif = bp.analysis.Bifurcation2D(
  qif,
  target_vars={'r': [0., 4.], 'v': [-3., 3.]},
  target_pars={'Iext': [-1, 1.]},
  resolutions=0.01
)
bif.plot_bifurcation()
bif.show_figure()


# codimension 2 bifurcation
bif = bp.analysis.Bifurcation2D(
  qif,
  target_vars={'r': [0., 4.], 'v': [-3., 3.]},
  target_pars={'delta': [0.5, 1.5], 'eta': [-10., -3.]},
  resolutions={'r': 0.01, 'v': 0.01, 'delta': 0.02, 'eta': 0.02},
  options={bp.analysis.C.x_by_y_in_fx: lambda v, delta: -delta / (bm.pi * qif.tau * 2 * v)}
)
bif.plot_bifurcation(num_par_segments=1, num_fp_segment=4)
bif.show_figure()
