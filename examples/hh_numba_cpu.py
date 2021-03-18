# -*- coding: utf-8 -*-


import numba as nb
import numpy as np

import brainpy as bp

bp.backend.set('numba')
bp.profile.set(dt=0.02)


class HH(bp.NeuGroup):
    target_backend = ['numpy', 'numba']

    def __init__(self, size, monitors, E_Na=50., E_K=-77., E_leak=-54.387,
                 C=1.0, g_Na=120., g_K=36., g_leak=0.03, V_th=20.):
        self.E_Na = E_Na
        self.E_K = E_K
        self.E_leak = E_leak
        self.C = C
        self.g_Na = g_Na
        self.g_K = g_K
        self.g_leak = g_leak
        self.V_th = V_th

        @nb.njit
        @bp.odeint(method='rk4', show_code=True)
        @nb.njit
        def integral(V, m, h, n, t, Iext):
            alpha = 0.1 * (V + 40) / (1 - np.exp(-(V + 40) / 10))
            beta = 4.0 * np.exp(-(V + 65) / 18)
            dmdt = alpha * (1 - m) - beta * m

            alpha = 0.07 * np.exp(-(V + 65) / 20.)
            beta = 1 / (1 + np.exp(-(V + 35) / 10))
            dhdt = alpha * (1 - h) - beta * h

            alpha = 0.01 * (V + 55) / (1 - np.exp(-(V + 55) / 10))
            beta = 0.125 * np.exp(-(V + 65) / 80)
            dndt = alpha * (1 - n) - beta * n

            I_Na = (g_Na * np.power(m, 3.0) * h) * (V - E_Na)
            I_K = (g_K * np.power(n, 4.0)) * (V - E_K)
            I_leak = g_leak * (V - E_leak)
            dVdt = (- I_Na - I_K - I_leak + Iext) / C

            return dVdt, dmdt, dhdt, dndt
        self.integral = integral

        self.V = np.ones(size) * -65.
        self.m = np.ones(size) * 0.5
        self.h = np.ones(size) * 0.6
        self.n = np.ones(size) * 0.32
        self.spike = np.zeros(size)
        self.input = np.zeros(size)

        super(HH, self).__init__(
            size=size,
            steps=[self.update],
            monitors=monitors,
            name='HH',
            show_code=True,
        )

    def update(self, _t):
        V, m, h, n = self.integral(self.V, self.m, self.h, self.n, _t, self.input)
        self.spike = np.logical_and(self.V < self.V_th, V >= self.V_th)
        self.V = V
        self.m = m
        self.h = h
        self.n = n
        self.input = 0


group = HH(100, monitors=['V'])
group.run(200., report=True)
bp.visualize.line_plot(group.mon.ts, group.mon.V, show=True)
group.run(200., inputs=('input', 10.), report=True)
bp.visualize.line_plot(group.mon.ts, group.mon.V, show=True)
