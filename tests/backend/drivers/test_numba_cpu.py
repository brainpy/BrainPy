# -*- coding: utf-8 -*-

import re
import ast
import inspect
from pprint import pprint

import numpy as np

import brainpy as bp
from brainpy.simulation.delays import ConstantDelay
from brainpy.backend.drivers.numba_cpu import _CPUReader
from brainpy.backend.drivers.numba_cpu import _analyze_step_func
from brainpy.backend.drivers.numba_cpu import _class2func


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
                                 monitors=monitors,
                                 name='HH')

    def update(self, _t):
        V, m, h, n = self.integral(self.V, self.m, self.h, self.n, _t, self.input)
        self.spike = np.logical_and(self.V < self.V_th, V >= self.V_th)
        self.V = V
        self.m = m
        self.h = h
        self.n = n
        self.input[:] = 0.


class LIF(bp.NeuGroup):
    target_backend = ['numba', 'numpy']

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
        self.t_last_spike = bp.ops.ones(size) * -1e7
        self.refractory = bp.ops.zeros(size)
        self.input = bp.ops.zeros(size)
        self.spike = bp.ops.zeros(size)
        self.V = bp.ops.ones(size) * V_reset

        super(LIF, self).__init__(size=size, **kwargs)

    @staticmethod
    @bp.odeint
    def int_V(V, t, Iext, V_rest, R, tau):
        return (- (V - V_rest) + R * Iext) / tau

    def update(self, _t):
        for i in range(self.size[0]):
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


class HH2(bp.NeuGroup):
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

        super(HH2, self).__init__(size=size,
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
        for i in range(self.size):
            pre_id = self.pre_ids[i]
            self.s[i] = self.int_s(self.s[i], _t, self.tau)
            self.s[i] += self.pre.spike[pre_id]
            self.g.push(i, self.g_max * self.s[i])

            post_id = self.post_ids[i]
            self.post.input[post_id] -= self.g.pull(i) * (self.post.V[post_id] - self.E)


def test_analyze_step1():
    group = HH(100, ['V'])
    r = _analyze_step_func(group, group.update)

    print('Code of the function:')
    print(r['code_string'])

    print('Code Scope:')
    print(r['code_scope'])

    print('self_data_in_right:')
    print(r['self_data_in_right'])

    print('self_data_without_index_in_left:')
    print(r['self_data_without_index_in_left'])

    print('self_data_with_index_in_left:')
    print(r['self_data_with_index_in_left'])


def test_analyze_step2():
    group = HH2(100, ['V'])
    r = _analyze_step_func(group, group.update)

    print('Code of the function:')
    print(r['code_string'])

    print('Code Scope:')
    print(r['code_scope'])

    print('self_data_in_right:')
    print(r['self_data_in_right'])

    print('self_data_without_index_in_left:')
    print(r['self_data_without_index_in_left'])

    print('self_data_with_index_in_left:')
    print(r['self_data_with_index_in_left'])


def test_StepFuncReader_for_AMPA1_vec():
    hh = HH(2)
    ampa = AMPA1_vec(pre=hh, post=hh, conn=bp.connect.All2All())

    update_code = bp.tools.deindent(inspect.getsource(ampa.update))
    # output_code = bp.tools.deindent(inspect.getsource(ampa.output))

    formatter = _CPUReader(host=ampa)
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
    pprint(formatter.visited_calls.keys())
    for v in formatter.visited_calls.values():
        pprint(v)
    print()


def test_StepFuncReader_for_delay1():
    update_code = '''
def non_uniform_push_for_tensor_bk(self):
    didx = self.delay_in_idx[idx_or_val]
    self.delay_data[didx, idx_or_val] = value
    '''

    arg = 'self'
    class_p1 = '\\b' + arg + '\\.[A-Za-z_][A-Za-z0-9_.]*\\b'
    self_data_without_index_in_left = set(re.findall(class_p1, update_code))
    class_p2 = '(\\b' + arg + '\\.[A-Za-z_][A-Za-z0-9_.]*)\\[.*\\]'
    self_data_with_index_in_left = set(re.findall(class_p2, update_code))

    print('self_data_without_index_in_left:')
    print(self_data_without_index_in_left)
    print()

    print('self_data_with_index_in_left:')
    print(self_data_with_index_in_left)
    print()

    delay = ConstantDelay(size=10, delay_time=np.random.randint(10, size=10))
    formatter = _CPUReader(host=delay)
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
    pprint(formatter.visited_calls.keys())
    for v in formatter.visited_calls.values():
        pprint(v)
    print()


def test_StepFuncReader_for_delay2():
    update_code = '''
def non_uniform_push_for_tensor_bk(self):
    self.delay_data[self.delay_in_idx[idx_or_val], idx_or_val] = value
    '''

    arg = 'self'
    class_p1 = '\\b' + arg + '\\.[A-Za-z_][A-Za-z0-9_.]*\\b'
    self_data_without_index_in_left = set(re.findall(class_p1, update_code))
    class_p2 = '(\\b' + arg + '\\.[A-Za-z_][A-Za-z0-9_.]*)\\['
    self_data_with_index_in_left = set(re.findall(class_p2, update_code))

    print('self_data_without_index_in_left:')
    print(self_data_without_index_in_left)
    print()

    print('self_data_with_index_in_left:')
    print(self_data_with_index_in_left)
    print()

    delay = ConstantDelay(size=10, delay_time=np.random.randint(10, size=10))
    formatter = _CPUReader(host=delay)
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
    pprint(formatter.visited_calls.keys())
    for v in formatter.visited_calls.values():
        pprint(v)
    print()


def test_StepFuncReader_for_lif():
    lif = LIF(10)
    code = bp.tools.deindent(inspect.getsource(lif.update))

    formatter = _CPUReader(host=lif)
    formatter.visit(ast.parse(code))

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
    pprint(formatter.visited_calls.keys())
    for v in formatter.visited_calls.values():
        pprint(v)
    print()


def test_class2func_for_lif():
    lif = LIF(10)

    func, calls, assigns = _class2func(cls_func=lif.update, host=lif, show_code=True)
    print(func)
    pprint(calls)
    pprint(assigns)


def test_class2func_for_AMPA1_vec():
    hh = HH(2)
    ampa = AMPA1_vec(pre=hh, post=hh, conn=bp.connect.All2All())
    func, calls, assigns = _class2func(cls_func=ampa.update, host=ampa, show_code=True)
    print(func)
    pprint(calls)
    pprint(assigns)


def test_class2func_for_AMPA1_vec2():
    hh = HH(2)
    ampa = AMPA1_vec(pre=hh, post=hh, conn=bp.connect.All2All(), delay=np.random.random(4) * 2)
    func, calls, assigns = _class2func(cls_func=ampa.update, host=ampa, show_code=True)
    print(func)
    pprint(calls)
    pprint(assigns)



test_analyze_step1()
test_analyze_step2()
test_StepFuncReader_for_lif()
test_class2func_for_lif()
test_StepFuncReader_for_AMPA1_vec()
test_class2func_for_AMPA1_vec()
test_class2func_for_AMPA1_vec2()
test_StepFuncReader_for_delay1()
test_StepFuncReader_for_delay2()

