# -*- coding: utf-8 -*-


import brainpy as bp

bp.backend.set('pytorch', dt=0.02)


class HH(bp.NeuGroup):
    target_backend = 'general'

    def __init__(self, size, ENa=50., EK=-77., EL=-54.387,
                 C=1.0, gNa=120., gK=36., gL=0.03, V_th=20.,
                 **kwargs):
        # parameters
        self.ENa = ENa
        self.EK = EK
        self.EL = EL
        self.C = C
        self.gNa = gNa
        self.gK = gK
        self.gL = gL
        self.V_th = V_th

        # variables
        self.V = bp.backend.ones(size) * -65.
        self.m = bp.backend.ones(size) * 0.5
        self.h = bp.backend.ones(size) * 0.6
        self.n = bp.backend.ones(size) * 0.32
        self.spike = bp.backend.zeros(size)
        self.input = bp.backend.zeros(size)

        super(HH, self).__init__(size=size, steps=[self.update], **kwargs)

    @staticmethod
    @bp.odeint(method='rk4', show_code=True)
    def integral(V, m, h, n, t, Iext, gNa, ENa, gK, EK, gL, EL, C):
        alpha = 0.1 * (V + 40) / (1 - bp.backend.exp(-(V + 40) / 10))
        beta = 4.0 * bp.backend.exp(-(V + 65) / 18)
        dmdt = alpha * (1 - m) - beta * m

        alpha = 0.07 * bp.backend.exp(-(V + 65) / 20.)
        beta = 1 / (1 + bp.backend.exp(-(V + 35) / 10))
        dhdt = alpha * (1 - h) - beta * h

        alpha = 0.01 * (V + 55) / (1 - bp.backend.exp(-(V + 55) / 10))
        beta = 0.125 * bp.backend.exp(-(V + 65) / 80)
        dndt = alpha * (1 - n) - beta * n

        I_Na = (gNa * m ** 3.0 * h) * (V - ENa)
        I_K = (gK * n ** 4.0) * (V - EK)
        I_leak = gL * (V - EL)
        dVdt = (- I_Na - I_K - I_leak + Iext) / C

        return dVdt, dmdt, dhdt, dndt

    def update(self, _t):
        V, m, h, n = self.integral(self.V, self.m, self.h, self.n, _t,
                                   self.input, self.gNa, self.ENa, self.gK,
                                   self.EK, self.gL, self.EL, self.C)
        self.spike = (self.V < self.V_th) * (V >= self.V_th)
        self.V = V
        self.m = m
        self.h = h
        self.n = n
        self.input[:] = 0


if __name__ == '__main__':
    group = HH(100, monitors=['V'], show_code=True)

    group.run(200., inputs=('input', 10.), report=True)
    bp.visualize.line_plot(group.mon.ts, group.mon.V, show=True)

    group.run(200., report=True)
    bp.visualize.line_plot(group.mon.ts, group.mon.V, show=True)
