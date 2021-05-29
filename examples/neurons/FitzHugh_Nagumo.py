# -*- coding: utf-8 -*-

import brainpy as bp

bp.backend.set('numpy', dt=0.02)


class FitzHughNagumo(bp.NeuGroup):
    target_backend = 'general'

    def __init__(self, size, a=0.7, b=0.8, tau=12.5, Vth=1.9, **kwargs):
        self.a = a
        self.b = b
        self.tau = tau
        self.Vth = Vth

        self.V = bp.ops.zeros(size)
        self.w = bp.ops.zeros(size)
        self.spike = bp.ops.zeros(size)
        self.input = bp.ops.zeros(size)

        super(FitzHughNagumo, self).__init__(size=size, **kwargs)

    @staticmethod
    @bp.odeint(method='rk4')
    def integral(V, w, t, Iext, a, b, tau):
        dw = (V + a - b * w) / tau
        dV = V - V * V * V / 3 - w + Iext
        return dV, dw

    def update(self, _t):
        V, self.w = self.integral(self.V, self.w, _t, self.input, self.a, self.b, self.tau)
        self.spike = (V >= self.Vth) * (self.V < self.Vth)
        self.V = V
        self.input[:] = 0.


if __name__ == '__main__':
    FNs = FitzHughNagumo(100, monitors=['V'])

    # simulation
    FNs.run(duration=300., inputs=('input', 1.), report=True)
    bp.visualize.line_plot(FNs.mon.ts, FNs.mon.V, show=True)

    FNs.run(duration=(300., 600.), inputs=('input', 0.6), report=True)
    bp.visualize.line_plot(FNs.mon.ts, FNs.mon.V, show=True)

    # phase plane analysis
    phase = bp.analysis.PhasePlane(FNs.integral,
                                   target_vars={'V': [-3, 2], 'w': [-2, 2]},
                                   fixed_vars=None,
                                   pars_update={'Iext': 1., "a": 0.7, 'b': 0.8, 'tau': 12.5})
    phase.plot_nullcline()
    phase.plot_fixed_point()
    # phase.plot_trajectory(initials={'V': -1, 'w': 1}, duration=100.)
    phase.plot_limit_cycle_by_sim(initials={'V': -1, 'w': 1}, duration=100.)
    phase.plot_vector_field(show=True)

    # bifurcation analysis
    bifurcation = bp.analysis.Bifurcation(FNs.integral,
                                          target_pars=dict(Iext=[-1, 1], a=[0.3, 0.8]),
                                          target_vars={'V': [-3, 2], 'w': [-2, 2]},
                                          fixed_vars=None,
                                          pars_update={'b': 0.8, 'tau': 12.5},
                                          numerical_resolution=0.01)
    bifurcation.plot_bifurcation(show=True)
