# -*- coding: utf-8 -*-

import numpy as np

import brainpy as bp

bp.backend.set('numpy')


class FitzHughNagumo(bp.NeuGroup):

    def __init__(self, size, a=0.7, b=0.8, tau=12.5, Vth=1.9, monitors=None):
        self.a = a
        self.b = b
        self.tau = tau
        self.Vth = Vth

        @bp.odeint(method='rk4')
        def integral(v, w, t, Iext):
            dw = (v + a - b * w) / tau
            dv = v - v * v * v / 3 - w + Iext
            return dv, dw

        self.integral = integral

        self.V = np.zeros(size)
        self.w = np.zeros(size)
        self.spike = np.zeros(size)
        self.input = np.zeros(size)

        super(FitzHughNagumo, self).__init__(
            size=size,
            steps=[self.update],
            monitors=monitors,
            name='FN_model',
            show_code=True,
            target_backend=['numpy'],
        )

    def update(self, _t):
        v, self.w = self.integral(self.V, self.w, _t, self.input)
        self.spike = np.logical_and(v >= self.Vth, self.V < self.Vth)
        self.V = v
        self.input = 0.


neurons = FitzHughNagumo(100, monitors=['V'])
neurons.run(300., inputs=('input', 1.), report=True)
bp.visualize.line_plot(neurons.mon.ts, neurons.mon.V, show=True)
