# -*- coding: utf-8 -*-

import brainpy as bp

bp.math.enable_x64()
bp.math.set_platform('cpu')


def wilson_cowan_model():
  """Wilson-Cowan model."""

  # Connection weights
  wEE = 12
  wEI = 4
  wIE = 13
  wII = 11

  # Refractory parameter
  r = 1

  # Excitatory parameters
  tau_E = 1  # Timescale of excitatory population
  a_E = 1.2  # Gain of excitatory population
  theta_E = 2.8  # Threshold of excitatory population

  # Inhibitory parameters
  tau_I = 1  # Timescale of inhibitory population
  a_I = 1  # Gain of inhibitory population
  theta_I = 4  # Threshold of inhibitory population

  def F(x, a, theta):
    return 1 / (1 + bp.math.exp(-a * (x - theta))) - 1 / (1 + bp.math.exp(a * theta))

  @bp.odeint
  def int_e(e, t, i, I_ext=0):
    return (-e + (1 - r * e) * F(wEE * e - wEI * i + I_ext, a_E, theta_E)) / tau_E

  @bp.odeint
  def int_i(i, t, e):
    return (-i + (1 - r * i) * F(wIE * e - wII * i, a_I, theta_I)) / tau_I

  pp = bp.analysis.PhasePlane2D([int_e, int_i],
                                target_vars={'e': [-0.2, 1.], 'i': [-0.2, 1.]},
                                resolutions=0.001)
  pp.plot_vector_field()
  pp.plot_nullcline(coords={'i': 'i-e'})
  pp.plot_fixed_point()
  pp.plot_trajectory(initials={'i': [0.5, 0.6], 'e': [-0.1, 0.4]},
                     duration=10, dt=0.1, show=True)


def decision_making_model():
  """Decision Making Model."""
  gamma = 0.641  # Saturation factor for gating variable
  tau = 0.06  # Synaptic time constant [sec]
  tau0 = 0.002  # Noise time constant [sec]
  a = 270.
  b = 108.
  d = 0.154

  I0 = 0.3255  # background current [nA]
  JE = 0.3725  # self-coupling strength [nA]
  JI = -0.1137  # cross-coupling strength [nA]
  JAext = 0.00117  # Stimulus input strength [nA]
  sigma = 1.02  # nA

  mu0 = 20.  # Stimulus firing rate [spikes/sec]
  coh = 0.5  # # Stimulus coherence [%]
  Ib1 = 0.3297
  Ib2 = 0.3297

  @bp.odeint
  def int_s1(s1, t, s2, gamma=0.641):
    I1 = JE * s1 + JI * s2 + Ib1 + JAext * mu0 * (1. + coh)
    r1 = (a * I1 - b) / (1. - bp.math.exp(-d * (a * I1 - b)))
    ds1dt = - s1 / tau + (1. - s1) * gamma * r1
    return ds1dt

  @bp.odeint
  def int_s2(s2, t, s1, gamma=0.641):
    I2 = JE * s2 + JI * s1 + Ib2 + JAext * mu0 * (1. - coh)
    r2 = (a * I2 - b) / (1. - bp.math.exp(-d * (a * I2 - b)))
    ds2dt = - s2 / tau + (1. - s2) * gamma * r2
    return ds2dt

  analyzer = bp.analysis.PhasePlane2D(
    model=[int_s1, int_s2],
    target_vars={'s1': [0, 1], 's2': [0, 1]},
    pars_update={'gamma': 0.641},
    resolutions=0.001,
  )
  analyzer.plot_vector_field()
  analyzer.plot_nullcline(coords=dict(s2='s2-s1'))
  analyzer.plot_fixed_point(show=True, tol_opt_screen=None)

  analyzer = bp.analysis.Bifurcation2D(
    model=[int_s1, int_s2],
    target_vars={'s1': [0., 1.], 's2': [0., 1.]},
    target_pars={'gamma': [0.4, 0.8]},
    resolutions={'s1': 0.001, 's2': 0.001, 'gamma': 0.01}
  )
  analyzer.plot_bifurcation(show=True,
                            num_par_segments=4,
                            num_fp_segment=4,
                            nullcline_aux_filter=0.1,
                            select_candidates='aux_rank',
                            num_rank=100)


def fhn_model():
  """FitzHugh–Nagumo model."""
  a = 0.7
  b = 0.8
  tau = 12.5

  @bp.odeint
  def int_V(V, t, w, Iext):
    dV = V - V * V * V / 3 - w + Iext
    return dV

  @bp.odeint
  def int_w(w, t, V, ):
    dw = (V + a - b * w) / tau
    return dw

  pp = bp.analysis.PhasePlane2D(
    model=[int_V, int_w],
    target_vars={'V': [-3, 3], 'w': [-1, 3]},
    pars_update={'Iext': 1.},
    resolutions=0.01,
  )
  pp.plot_vector_field()
  pp.plot_nullcline(coords={'V': 'w-V'})
  pp.plot_fixed_point(show=True)


if __name__ == '__main__':
  wilson_cowan_model()
  decision_making_model()
  fhn_model()

