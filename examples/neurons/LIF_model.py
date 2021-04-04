# -*- coding: utf-8 -*-


import brainpy as bp

bp.backend.set('numpy')


class LIF(bp.NeuGroup):
    target_backend = ['numpy', 'numba']

    def __init__(self, size, t_refractory=1., V_rest=0.,
                 V_reset=-5., V_th=20., R=1., tau=10., **kwargs):
        # parameters
        self.V_rest = V_rest
        self.V_reset = V_reset
        self.V_th = V_th
        self.R = R
        self.tau = tau
        self.t_refractory = t_refractory

        # variables
        self.t_last_spike = bp.backend.ones(size) * -1e7
        self.refractory = bp.backend.zeros(size)
        self.input = bp.backend.zeros(size)
        self.spike = bp.backend.zeros(size)
        self.V = bp.backend.ones(size) * V_reset

        super(LIF, self).__init__(size=size, **kwargs)

    @staticmethod
    @bp.odeint
    def int_V(V, t, Iext, V_rest, R, tau):
        return (- (V - V_rest) + R * Iext) / tau

    def update(self, _t):
        for i in range(self.num):
            if _t - self.t_last_spike[i] <= self.t_refractory:
                self.refractory[i] = 1.
            else:
                self.refractory[0] = 0.
                V = self.int_V(self.V[i], _t, self.input[i], self.V_rest, self.R, self.tau)
                if V >= self.V_th:
                    self.V[i] = self.V_reset
                    self.spike[i] = 1.
                    self.t_last_spike[i] = _t
                else:
                    self.spike[i] = 0.
                    self.V[i] = V
            self.input[i] = 0.


if __name__ == '__main__':
    group = LIF(100, monitors=['V'], show_code=True)

    group.run(duration=200., inputs=('input', 26.), report=True)
    bp.visualize.line_plot(group.mon.ts, group.mon.V, show=True)

    group.run(duration=(200, 400.), report=True)
    bp.visualize.line_plot(group.mon.ts, group.mon.V, show=True)
