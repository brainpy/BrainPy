# -*- coding: utf-8 -*-


import brainpy as bp

bp.backend.set('numba-cuda', dt=0.1)
bp.set_default_odeint('exponential_euler')


class HH(bp.NeuGroup):
    target_backend = 'numba-cuda'

    def __init__(self, size, ENa=50., EK=-77., EL=-54.387, C=1.0, gNa=120.,
                 gK=36., gL=0.03, V_th=20., **kwargs):
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
        self.V = bp.ops.ones(size) * -65.
        self.m = bp.ops.ones(size) * 0.5
        self.h = bp.ops.ones(size) * 0.6
        self.n = bp.ops.ones(size) * 0.32
        self.spike = bp.ops.zeros(size)
        self.input = bp.ops.zeros(size)

        super(HH, self).__init__(size=size, **kwargs)

    @staticmethod
    @bp.odeint
    def int_V(V, t, m, h, n, Iext, gNa, ENa, gK, EK, gL, EL, C):
        I_Na = (gNa * m * m * m * h) * (V - ENa)
        I_K = (gK * n * n * n * n) * (V - EK)
        I_leak = gL * (V - EL)
        dVdt = (- I_Na - I_K - I_leak + Iext) / C
        return dVdt

    @staticmethod
    @bp.odeint
    def int_m(m, t, V):
        alpha = 0.1 * (V + 40) / (1 - bp.ops.exp(-(V + 40) / 10))
        beta = 4.0 * bp.ops.exp(-(V + 65) / 18)
        dmdt = alpha * (1 - m) - beta * m
        return dmdt

    @staticmethod
    @bp.odeint
    def int_h(h, t, V):
        alpha = 0.07 * bp.ops.exp(-(V + 65) / 20.)
        beta = 1 / (1 + bp.ops.exp(-(V + 35) / 10))
        dhdt = alpha * (1 - h) - beta * h
        return dhdt

    @staticmethod
    @bp.odeint
    def int_n(n, t, V):
        alpha = 0.01 * (V + 55) / (1 - bp.ops.exp(-(V + 55) / 10))
        beta = 0.125 * bp.ops.exp(-(V + 65) / 80)
        dndt = alpha * (1 - n) - beta * n
        return dndt

    def update(self, _t):
        for i in range(self.num):
            m = self.int_m(self.m[i], _t, self.V[i])
            h = self.int_h(self.h[i], _t, self.V[i])
            n = self.int_n(self.n[i], _t, self.V[i])
            V = self.int_V(self.V[i], _t, self.m[i], self.h[i], self.n[i], self.input[i],
                           self.gNa, self.ENa, self.gK, self.EK, self.gL, self.EL, self.C)
            self.spike[i] = (self.V[i] < self.V_th) * (V >= self.V_th)
            self.V[i] = V
            self.m[i] = m
            self.h[i] = h
            self.n[i] = n
            self.input[i] = 0.


if __name__ == '__main__':
    group = HH(10000, monitors=['V'])

    group.run(200., inputs=('input', 10.), report=True)
    group.driver.to_host()
    bp.visualize.line_plot(group.mon.ts, group.mon.V, show=True)

    group.run(200., report=True)
    group.driver.to_host()
    bp.visualize.line_plot(group.mon.ts, group.mon.V, show=True)
