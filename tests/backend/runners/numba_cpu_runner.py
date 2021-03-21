# -*- coding: utf-8 -*-

import inspect
from brainpy.backend.runners.numba_cpu_runner import analyze_step_func
from brainpy.backend.runners.numba_cpu_runner import StepFuncReader

from pprint import pprint
import ast
import numpy as np
import brainpy as bp


def test_analyze_step1():
    class HH(bp.NeuGroup):
        target_backend = ['numpy']

        def __init__(self, size, monitors=None, E_Na=50., E_K=-77., E_leak=-54.387, C=1.0,
                     g_Na=120., g_K=36., g_leak=0.03, V_th=20.):
            @bp.odeint(method='rkdp', show_code=False)
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

            self.E_Na = E_Na
            self.E_K = E_K
            self.E_leak = E_leak
            self.C = C
            self.g_Na = g_Na
            self.g_K = g_K
            self.g_leak = g_leak
            self.V_th = V_th

            self.integral = integral

            self.V = np.ones(size) * -65.
            self.m = np.ones(size) * 0.5
            self.h = np.ones(size) * 0.6
            self.n = np.ones(size) * 0.32
            self.spike = np.zeros(size)
            self.input = np.zeros(size)

            super(HH, self).__init__(size=size,
                                     steps=[self.update],
                                     monitors=monitors,
                                     name='HH')

        def update(self, _t):
            V, m, h, n = self.integral(self.V, self.m, self.h, self.n, _t, self.input)
            self.spike = np.logical_and(self.V < self.V_th, V >= self.V_th)
            self.V = V
            self.m = m
            self.h = h
            self.n = n
            self.input = 0.

    group = HH(100, ['V'])
    r = analyze_step_func(group.update)

    print('Code of the function:')
    print(r[0])
    print('Code Scope:')
    print(r[1])
    print('Data need pass:')
    print(r[2])
    print('Data need return:')
    print(r[3])


def test_analyze_step2():
    class HH(bp.NeuGroup):
        target_backend = ['numpy']

        def __init__(self, size, monitors=None, E_Na=50., E_K=-77., E_leak=-54.387, C=1.0,
                     g_Na=120., g_K=36., g_leak=0.03, V_th=20.):
            @bp.odeint(method='rkdp', show_code=False)
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

            self.E_Na = E_Na
            self.E_K = E_K
            self.E_leak = E_leak
            self.C = C
            self.g_Na = g_Na
            self.g_K = g_K
            self.g_leak = g_leak
            self.V_th = V_th

            self.integral = integral

            self.V = np.ones(size) * -65.
            self.m = np.ones(size) * 0.5
            self.h = np.ones(size) * 0.6
            self.n = np.ones(size) * 0.32
            self.spike = np.zeros(size)
            self.input = np.zeros(size)

            super(HH, self).__init__(size=size,
                                     steps=[self.update],
                                     monitors=monitors,
                                     name='HH')

        def update(self, _t):
            V, m, h, n = self.integral(self.V, self.m, self.h, self.n, _t, self.input)
            self.spike = np.logical_and(self.V < self.V_th, V >= self.V_th)
            self.V[:] = V
            self.m[:] = m
            self.h[:] = h
            self.n[:] = n
            self.input[:] = 0.


    group = HH(100, ['V'])
    r = analyze_step_func(group.update)

    print('Code of the function:')
    print(r[0])
    print('Code Scope:')
    print(r[1])
    print('Data need pass:')
    print(r[2])
    print('Data need return:')
    print(r[3])


def test_StepFuncReader1():
    class HH(bp.NeuGroup):
        target_backend = ['numpy', 'numba', 'numba-parallel']

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
            self.V = np.ones(size) * -65.
            self.m = np.ones(size) * 0.5
            self.h = np.ones(size) * 0.6
            self.n = np.ones(size) * 0.32
            self.spike = np.zeros(size)
            self.input = np.zeros(size)

            super(HH, self).__init__(size=size, steps=[self.update], **kwargs)

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
            self.input = 0

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
            self.s = bp.backend.zeros(self.size)
            self.g = self.register_constant_delay('g', size=self.size, delay_time=delay)

            super(AMPA1_vec, self).__init__(steps=[self.update, ],
                                            pre=pre, post=post, **kwargs)

        @staticmethod
        @bp.odeint(method='euler')
        def int_s(s, t, tau):
            return - s / tau

        def update(self, _t):
            for i in range(self.size):
                pre_id = self.pre_ids[i]
                self.s[i] = self.int_s(self.s[i], _t, self.tau)
                self.s[i] += self.pre.spike[pre_id]
                self.g.push(i, self.g_max * self.s[i])

                post_id = self.post_ids[i]
                self.post.input[post_id] -= self.g.pull(i) * (self.post.V[post_id] - self.E)

    hh = HH(2)
    ampa = AMPA1_vec(pre=hh, post=hh, conn=bp.connect.All2All())

    update_code = bp.tools.deindent(inspect.getsource(ampa.update))
    # output_code = bp.tools.deindent(inspect.getsource(ampa.output))

    formatter = StepFuncReader(host=ampa)
    formatter.visit(ast.parse(update_code))

    print('lefts:')
    pprint(formatter.lefts)
    print()
    print('rights:')
    pprint(formatter.rights)
    print()
    print('lines:')
    pprint(formatter.lines)
    print()
    print('delay_call:')
    pprint(formatter.delay_call)
    print()


test_StepFuncReader1()


