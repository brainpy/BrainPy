# -*- coding: utf-8 -*-

import brainpy as bp
import brainpy.math as bm

bp.math.enable_x64()


class DecisionMakingModel(bp.DynamicalSystem):
  def __init__(self, method='exp_auto'):
    super(DecisionMakingModel, self).__init__()

    # parameters
    self.gamma = 0.641  # Saturation factor for gating variable
    self.tau = 0.06  # Synaptic time constant [sec]
    self.a = 270.
    self.b = 108.
    self.d = 0.154

    self.JE = 0.3725  # self-coupling strength [nA]
    self.JI = -0.1137  # cross-coupling strength [nA]
    self.JAext = 0.00117  # Stimulus input strength [nA]

    self.mu = 20.  # Stimulus firing rate [spikes/sec]
    self.coh = 0.5  # Stimulus coherence [%]
    self.Ib1 = 0.3297
    self.Ib2 = 0.3297

    # variables
    self.s1 = bm.Variable(bm.zeros(1))
    self.s2 = bm.Variable(bm.zeros(1))

    # functions
    def ds1(s1, t, s2, coh=0.5, mu=20.):
      I1 = self.JE * s1 + self.JI * s2 + self.Ib1 + self.JAext * mu * (1. + coh)
      r1 = (self.a * I1 - self.b) / (1. - bm.exp(-self.d * (self.a * I1 - self.b)))
      return - s1 / self.tau + (1. - s1) * self.gamma * r1

    def ds2(s2, t, s1, coh=0.5, mu=20.):
      I2 = self.JE * s2 + self.JI * s1 + self.Ib2 + self.JAext * mu * (1. - coh)
      r2 = (self.a * I2 - self.b) / (1. - bm.exp(-self.d * (self.a * I2 - self.b)))
      return - s2 / self.tau + (1. - s2) * self.gamma * r2

    self.int_s1 = bp.odeint(ds1, method=method)
    self.int_s2 = bp.odeint(ds2, method=method)

  def update(self, _t, _dt):
    self.s1.value = self.int_s1(self.s1, _t, self.s2, self.coh, self.mu, _dt)
    self.s2.value = self.int_s2(self.s2, _t, self.s1, self.coh, self.mu, _dt)


model = DecisionMakingModel()

# phase plane analysis
analyzer = bp.analysis.PhasePlane2D(
  model=model,
  target_vars={'s1': [0, 1], 's2': [0, 1]},
  resolutions=0.001,
)
analyzer.plot_vector_field()
analyzer.plot_nullcline(coords=dict(s2='s2-s1'))
analyzer.plot_fixed_point()
analyzer.show_figure()

# codimension 1 bifurcation
analyzer = bp.analysis.Bifurcation2D(
  model=model,
  target_vars={'s1': [0., 1.], 's2': [0., 1.]},
  target_pars={'coh': [0., 1.]},
  pars_update={'mu': 40.},
  resolutions={'s1': 0.001, 's2': 0.001, 'coh': 0.01}
)
analyzer.plot_bifurcation(num_par_segments=4,
                          num_fp_segment=4,
                          nullcline_aux_filter=0.1,
                          select_candidates='aux_rank',
                          num_rank=50)
analyzer.show_figure()
