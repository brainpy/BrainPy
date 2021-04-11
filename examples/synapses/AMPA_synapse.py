# -*- coding: utf-8 -*-

import numpy as np
from numba import prange

import brainpy as bp

bp.integrators.set_default_odeint('rk4')
bp.backend.set(backend='numba', dt=0.01)


class HH(bp.NeuGroup):
    target_backend = ['numpy', 'numba', 'numba-parallel']

    def __init__(self, size, ENa=50., EK=-77., EL=-54.387, C=1.0,
                 gNa=120., gK=36., gL=0.03, V_th=20., **kwargs):
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
        self.V = np.ones(size) * -65.
        self.m = np.ones(size) * 0.5
        self.h = np.ones(size) * 0.6
        self.n = np.ones(size) * 0.32
        self.spike = np.zeros(size)
        self.input = np.zeros(size)

        super(HH, self).__init__(size=size, **kwargs)

    @staticmethod
    @bp.odeint
    def integral(V, m, h, n, t, Iext, gNa, ENa, gK, EK, gL, EL, C):
        alpha = 0.1 * (V + 40) / (1 - np.exp(-(V + 40) / 10))
        beta = 4.0 * np.exp(-(V + 65) / 18)
        dmdt = alpha * (1 - m) - beta * m

        alpha = 0.07 * np.exp(-(V + 65) / 20.)
        beta = 1 / (1 + np.exp(-(V + 35) / 10))
        dhdt = alpha * (1 - h) - beta * h

        alpha = 0.01 * (V + 55) / (1 - np.exp(-(V + 55) / 10))
        beta = 0.125 * np.exp(-(V + 65) / 80)
        dndt = alpha * (1 - n) - beta * n

        I_Na = (gNa * np.power(m, 3.0) * h) * (V - ENa)
        I_K = (gK * np.power(n, 4.0)) * (V - EK)
        I_leak = gL * (V - EL)
        dVdt = (- I_Na - I_K - I_leak + Iext) / C

        return dVdt, dmdt, dhdt, dndt

    def update(self, _t):
        V, m, h, n = self.integral(self.V, self.m, self.h, self.n, _t,
                                   self.input, self.gNa, self.ENa, self.gK,
                                   self.EK, self.gL, self.EL, self.C)
        self.spike = np.logical_and(self.V < self.V_th, V >= self.V_th)
        self.V = V
        self.m = m
        self.h = h
        self.n = n
        self.input[:] = 0


class AMPA1_vec(bp.TwoEndConn):
    target_backend = ['numpy', 'numba', 'numba-parallel', 'numa-cuda']

    def __init__(self, pre, post, conn, delay=0., g_max=0.10, E=0., tau=2.0, **kwargs):
        # parameters
        self.g_max = g_max
        self.E = E
        self.tau = tau
        self.delay = delay

        # connections
        self.conn = conn(pre.size, post.size)
        self.pre_ids, self.post_ids = conn.requires('pre_ids', 'post_ids')
        self.size = len(self.pre_ids)

        # data
        self.s = bp.ops.zeros(self.size)
        self.g = self.register_constant_delay('g', size=self.size, delay_time=delay)

        super(AMPA1_vec, self).__init__(pre=pre, post=post, **kwargs)

    @staticmethod
    @bp.odeint(method='euler')
    def int_s(s, t, tau):
        return - s / tau

    def update(self, _t):
        for i in prange(self.size):
            pre_id = self.pre_ids[i]
            self.s[i] = self.int_s(self.s[i], _t, self.tau)
            self.s[i] += self.pre.spike[pre_id]
            self.g.push(i,self.g_max * self.s[i])
            post_id = self.post_ids[i]
            self.post.input[post_id] -= self.g.pull(i) * (self.post.V[post_id] - self.E)


class AMPA1_mat(bp.TwoEndConn):
    target_backend = ['numpy', 'numba']

    def __init__(self, pre, post, conn, delay=0., g_max=0.10, E=0., tau=2.0, **kwargs):
        # parameters
        self.g_max = g_max
        self.E = E
        self.tau = tau
        self.delay = delay

        # connections
        self.conn = conn(pre.size, post.size)
        self.conn_mat = conn.requires('conn_mat')
        self.size = bp.ops.shape(self.conn_mat)

        # variables
        self.s = bp.ops.zeros(self.size)
        self.g = self.register_constant_delay('g', size=self.size, delay_time=delay)

        super(AMPA1_mat, self).__init__(pre=pre, post=post, **kwargs)

    @staticmethod
    @bp.odeint
    def int_s(s, t, tau):
        return - s / tau

    def update(self, _t):
        self.s = self.int_s(self.s, _t, self.tau)
        for i in range(self.pre.size[0]):
            if self.pre.spike[i] > 0:
                self.s[i] += self.conn_mat[i]
        self.g.push(self.g_max * self.s)
        g = self.g.pull()
        self.post.input -= bp.ops.sum(g, axis=0) * (self.post.V - self.E)


if __name__ == '__main__':
    hh = HH(100, monitors=['V'])
    ampa = AMPA1_vec(pre=hh, post=hh, conn=bp.connect.All2All(), delay=10., monitors=['s'])
    net = bp.Network(hh, ampa)
    net.run(100., inputs=(hh, 'input', 10.), report=True)

    fig, gs = bp.visualize.get_figure(row_num=2, col_num=1, )
    fig.add_subplot(gs[0, 0])
    bp.visualize.line_plot(hh.mon.ts, hh.mon.V)
    fig.add_subplot(gs[1, 0])
    bp.visualize.line_plot(ampa.mon.ts, ampa.mon.s, show=True)
