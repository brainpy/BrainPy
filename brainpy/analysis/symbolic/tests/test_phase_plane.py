# -*- coding: utf-8 -*-


import unittest
import brainpy as bp


class Test1DPhasePlane(unittest.TestCase):
  def test1(self):
    @bp.odeint
    def int_x(x, t, Iext):
      dx = x ** 3 - x + Iext
      return dx

    analyzer = bp.symbolic.OldPhasePlane(int_x,
                                         target_vars={'x': [-10, 10]},
                                         pars_update={'Iext': 1.})

    with self.assertRaises(NotImplementedError):
      analyzer.plot_nullcline()


class Test2DPhasePlane(unittest.TestCase):
  def test1(self):
    @bp.odeint
    def fhn(V, w, t, Iext, a, b, tau):
      dw = (V + a - b * w) / tau
      dV = V - V * V * V / 3 - w + Iext
      return dV, dw

    phase = bp.symbolic.OldPhasePlane(fhn,
                                      target_vars={'V': [-3, 2], 'w': [-2, 2]},
                                      pars_update={'Iext': 1., "a": 0.7, 'b': 0.8, 'tau': 12.5})
    phase.plot_nullcline()
    phase.plot_fixed_point()
    phase.plot_trajectory(initials={'V': -1, 'w': 1}, duration=100.)
    phase.plot_limit_cycle_by_sim(initials={'V': -1, 'w': 1}, duration=100.)
    phase.plot_vector_field(show=True)
