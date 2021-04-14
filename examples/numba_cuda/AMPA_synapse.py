# -*- coding: utf-8 -*-

import numpy as np
import brainpy as bp
from numba import cuda

bp.backend.set(backend='numba-cuda', dt=0.05)
bp.integrators.set_default_odeint('exponential_euler')


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


class AMPA1(bp.TwoEndConn):
    target_backend = 'numba-cuda'

    def __init__(self, pre, post, conn, delay=0., g_max=0.10, E=0., tau=2.0, **kwargs):
        # parameters
        self.g_max = g_max
        self.E = E
        self.tau = tau
        self.delay = delay

        # connections
        self.conn = conn(pre.size, post.size)
        self.pre_ids, self.post_ids = conn.requires('pre_ids', 'post_ids')
        self.num = len(self.pre_ids)

        # data
        self.s = bp.ops.zeros(self.num)
        self.s0 = bp.ops.zeros(1)
        self.g = self.register_constant_delay('g', size=self.num, delay_time=delay)

        super(AMPA1, self).__init__(pre=pre, post=post, **kwargs)

    @staticmethod
    @bp.odeint
    def int_s(s, t, tau):
        ds = - s / tau
        return ds

    def update(self, _t):
        for i in range(self.num):
            pre_id = self.pre_ids[i]
            self.s[i] = self.int_s(self.s[i], _t, self.tau)
            self.s[i] += self.pre.spike[pre_id]
            self.g.push(i, self.g_max * self.s[i])
            post_id = self.post_ids[i]
            self.post.input[post_id] -= self.g.pull(i) * (self.post.V[post_id] - self.E)
            if i == 0:
                self.s0[0] = self.s[i]
            cuda.syncthreads()


def uniform_delay():
    hh = HH(4000, monitors=['V'])
    ampa = AMPA1(pre=hh, post=hh, conn=bp.connect.All2All(), delay=1., monitors=['s0'])
    ampa.g_max /= hh.num
    net = bp.Network(hh, ampa)

    net.run(100., inputs=(hh, 'input', 10.), report=True)

    fig, gs = bp.visualize.get_figure(row_num=2, col_num=1, )
    fig.add_subplot(gs[0, 0])
    bp.visualize.line_plot(hh.mon.ts, hh.mon.V)
    fig.add_subplot(gs[1, 0])
    bp.visualize.line_plot(ampa.mon.ts, ampa.mon.s0, show=True)


def non_uniform_delay():
    hh = HH(4000, monitors=['V'])
    ampa = AMPA1(pre=hh, post=hh, conn=bp.connect.All2All(),
                 delay=lambda: np.random.random() * 1., monitors=['s0'])
    ampa.g_max /= hh.num
    net = bp.Network(hh, ampa)

    net.run(100., inputs=(hh, 'input', 10.), report=True)

    fig, gs = bp.visualize.get_figure(row_num=2, col_num=1, )
    fig.add_subplot(gs[0, 0])
    bp.visualize.line_plot(hh.mon.ts, hh.mon.V)
    fig.add_subplot(gs[1, 0])
    bp.visualize.line_plot(ampa.mon.ts, ampa.mon.s0, show=True)


if __name__ == '__main__':
    # uniform_delay()
    non_uniform_delay()
