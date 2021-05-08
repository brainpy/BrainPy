# -*- coding: utf-8 -*-


from pprint import pprint

import pytest
from numba import cuda

if not cuda.is_available():
    pytest.skip("cuda is not available", allow_module_level=True)

import brainpy as bp
from brainpy.backend.drivers.numba_cuda import set_monitor_done_in
from brainpy.backend.drivers.numba_cuda import NumbaCUDANodeDriver

bp.backend.set('numba-cuda', dt=0.02)


class StochasticLIF(bp.NeuGroup):
    target_backend = 'general'

    def __init__(self, size, t_refractory=1., V_rest=0.,
                 V_reset=-5., V_th=20., R=1., tau=10.,
                 has_noise=True, **kwargs):
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

        if has_noise:
            self.int_V = bp.sdeint(f=self.f_v, g=self.g_v)
        else:
            self.int_V = bp.odeint(f=self.f_v)

        super(StochasticLIF, self).__init__(size=size, **kwargs)

    @staticmethod
    def f_v(V, t, Iext, V_rest, R, tau):
        return (- (V - V_rest) + R * Iext) / tau

    @staticmethod
    def g_v(V, t, Iext, V_rest, R, tau):
        return 1.

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
        self.num = len(self.pre_ids)

        # data
        self.s = bp.ops.zeros(self.num)
        self.g = self.register_constant_delay('g', size=self.num, delay_time=delay)

        super(AMPA1_vec, self).__init__(pre=pre, post=post, **kwargs)

    @staticmethod
    @bp.odeint(method='euler')
    def int_s(s, t, tau):
        return - s / tau

    def update(self, _t):
        for i in range(self.num):
            pre_id = self.pre_ids[i]
            self.s[i] = self.int_s(self.s[i], _t, self.tau)
            self.s[i] += self.pre.spike[pre_id]
            self.g.push(i, self.g_max * self.s[i])

            post_id = self.post_ids[i]
            self.post.input[post_id] -= self.g.pull(i) * (self.post.V[post_id] - self.E)


def test_stochastic_lif_monitors1():

    for place in ['cpu', 'cuda']:
        set_monitor_done_in(place)
        lif = StochasticLIF(1, monitors=['V', 'input', 'spike'])
        driver = NumbaCUDANodeDriver(pop=lif)
        driver.get_monitor_func(mon_length=100, show_code=True)
        pprint(driver.formatted_funcs)
        print()
        print()



# test_stochastic_lif_monitors1()

