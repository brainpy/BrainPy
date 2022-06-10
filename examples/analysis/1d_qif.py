# -*- coding: utf-8 -*-


import brainpy as bp

bp.math.enable_x64()  # important!


def qif(V, t, c=.07, R=1., tau=10., Iext=0., V_rest=-65., V_c=-50.0, ):
  return (c * (V - V_rest) * (V - V_c) + R * Iext) / tau


pp = bp.analysis.PhasePlane1D(
  qif,
  target_vars={'V': [-80, -30]},
  pars_update={'Iext': .0},
  resolutions=0.01
)

pp.plot_fixed_point()
pp.plot_vector_field(show=True)
