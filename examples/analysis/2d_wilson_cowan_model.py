import brainpy as bp
import brainpy.math as bm

bp.math.enable_x64()


class WilsonCowanModel(bp.DynamicalSystem):
  def __init__(self, method='exp_auto'):
    super(WilsonCowanModel, self).__init__()

    # Connection weights
    self.wEE = 12
    self.wEI = 4
    self.wIE = 13
    self.wII = 11

    # Refractory parameter
    self.r = 1

    # Excitatory parameters
    self.E_tau = 1  # Timescale of excitatory population
    self.E_a = 1.2  # Gain of excitatory population
    self.E_theta = 2.8  # Threshold of excitatory population

    # Inhibitory parameters
    self.I_tau = 1  # Timescale of inhibitory population
    self.I_a = 1  # Gain of inhibitory population
    self.I_theta = 4  # Threshold of inhibitory population

    # variables
    self.i = bm.Variable(bm.ones(1))
    self.e = bm.Variable(bm.ones(1))
    self.Iext = bm.Variable(bm.zeros(1))

    # functions
    def F(x, a, theta):
      return 1 / (1 + bm.exp(-a * (x - theta))) - 1 / (1 + bm.exp(a * theta))

    def de(e, t, i, Iext=0.):
      x = self.wEE * e - self.wEI * i + Iext
      return (-e + (1 - self.r * e) * F(x, self.E_a, self.E_theta)) / self.E_tau

    def di(i, t, e):
      x = self.wIE * e - self.wII * i
      return (-i + (1 - self.r * i) * F(x, self.I_a, self.I_theta)) / self.I_tau

    self.int_e = bp.odeint(de, method=method)
    self.int_i = bp.odeint(di, method=method)

  def update(self, _t, _dt):
    self.e.value = self.int_e(self.e, _t, self.i, self.Iext, _dt)
    self.i.value = self.int_i(self.i, _t, self.e, _dt)
    self.Iext[:] = 0.


model = WilsonCowanModel()

# simulation
runner = bp.StructRunner(model, monitors=['e', 'i'], inputs=['Iext', 0.])

# phase plane analysis
pp = bp.analysis.PhasePlane2D(
  model,
  target_vars={'e': [-0.2, 1.], 'i': [-0.2, 1.]},
  resolutions=0.001,
)
pp.plot_vector_field()
pp.plot_nullcline(coords={'i': 'i-e'})
pp.plot_fixed_point()
pp.plot_trajectory(initials={'i': [0.5, 0.6], 'e': [-0.1, 0.4]},
                   duration=10, dt=0.1)
pp.show_figure()
