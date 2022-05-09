# -*- coding: utf-8 -*-

import unittest

import brainpy as bp

block = False


class TestPhasePlane(unittest.TestCase):
  def test_1d(self):
    import matplotlib.pyplot as plt
    bp.math.enable_x64()

    @bp.odeint
    def int_x(x, t, Iext):
      dx = x ** 3 - x + Iext
      return dx

    analyzer = bp.analysis.PhasePlane1D(model=int_x,
                                        target_vars={'x': [-2, 2]},
                                        pars_update={'Iext': 0.},
                                        resolutions=0.01)

    plt.ion()
    analyzer.plot_vector_field()
    analyzer.plot_fixed_point()
    plt.show(block=block)
    bp.math.disable_x64()

  def test_2d_decision_making_model(self):
    import matplotlib.pyplot as plt

    bp.math.enable_x64()
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

    mu0 = 40.  # Stimulus firing rate [spikes/sec]
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
    def int_s2(s2, t, s1):
      I2 = JE * s2 + JI * s1 + Ib2 + JAext * mu0 * (1. - coh)
      r2 = (a * I2 - b) / (1. - bp.math.exp(-d * (a * I2 - b)))
      ds2dt = - s2 / tau + (1. - s2) * gamma * r2
      return ds2dt

    analyzer = bp.analysis.PhasePlane2D(
      model=[int_s1, int_s2],
      target_vars={'s1': [0, 1], 's2': [0, 1]},
      resolutions=0.001
    )
    plt.ion()
    analyzer.plot_vector_field()
    analyzer.plot_nullcline(coords=dict(s2='s2-s1'))
    analyzer.plot_fixed_point()
    plt.show(block=block)
    bp.math.disable_x64()
