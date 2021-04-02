# -*- coding: utf-8 -*-


import numpy as np
import numba as nb
import brainpy as bp

bp.backend.set('numba')


class LIF(bp.NeuGroup):
    target_backend = ['numpy', 'numba']

    def __init__(self, size, mu, t_refractory=2., V_reset=16., V_th=20.,tau=10., **kwargs):
        # parameters
        self.mu = mu
        self.V_reset = V_reset
        self.V_th = V_th
        self.tau = tau
        self.t_refractory = t_refractory

        # variables
        self.t_last_spike = bp.backend.ones(size) * -1e7
        self.input = bp.backend.zeros(size)
        self.not_ref = bp.backend.ones(size)
        self.spike = bp.backend.zeros(size, dtype=bool)
        self.V = bp.backend.ones(size) * (V_th - V_reset) + V_reset

        def f_V(V, t, Iext, tau):
            dc = (-V + Iext) / tau
            return dc

        def g_V(V, t, Iext, tau):
            return 1.

        self.int_V = bp.sdeint(f=f_V, g=g_V, method='exponential_euler')

        super(LIF, self).__init__(size=size, **kwargs)

    def update(self, _t):
        for i in nb.prange(self.size[0]):
            if _t - self.t_last_spike[i] <= self.t_refractory:
                self.not_ref[i] = 0.
            else:
                self.not_ref[0] = 1.
                V = self.int_V(self.V[i], _t, self.input[i], self.tau)
                if V >= self.V_th:
                    self.V[i] = self.V_reset
                    self.spike[i] = True
                    self.t_last_spike[i] = _t
                else:
                    self.spike[i] = False
                    self.V[i] = V
            self.input[i] = self.mu


class SynWithSTP(bp.TwoEndConn):
    target_backend = ['numpy', 'numba']

    def __init__(self, pre, post, conn, delay, U=0.2, tau_f=1500., tau_d=200., **kwargs):
        self.U = U
        self.tau_f = tau_f
        self.tau_d = tau_d
        self.delay = delay

        assert isinstance(conn, dict)
        self.g_max = conn['weights']
        self.pre2syn = conn['pre2syn']
        self.post2syn = conn['post2syn']
        self.pre_ids = conn['pre_ids']
        self.post_ids = conn['post_ids']

        self.num = len(self.post_ids)
        self.u = bp.backend.ones(self.num) * U
        self.x = bp.backend.ones(self.num)
        self.g = self.register_constant_delay('g', self.num, delay_time=delay)

        super(SynWithSTP, self).__init__(pre=pre, post=post, **kwargs)

    @staticmethod
    @bp.odeint(method='exponential_euler')
    def integral(u, x, t, tau_f, tau_d, U):
        du = U - u / tau_f
        dx = (1 - x) / tau_d
        return du, dx

    def update(self, _t):
        # update
        u, x = self.integral(self.u, self.x, _t, self.tau_f, self.tau_d, self.U)
        for pre_id, spike in enumerate(self.pre.spike):
            if spike:
                for syn_id in self.pre2syn[pre_id]:
                    u[syn_id] += self.U * (1 - self.u[syn_id])
                    x[syn_id] -= u[syn_id] * self.x[syn_id]
        self.g.push(self.g_max * u * self.x)
        # output
        g = self.g.pull()
        for post_id in range(self.post.num):
            for syn_id in self.post2syn[post_id]:
                self.post.input[post_id] += g[syn_id] * self.post.not_ref[post_id]


class SynWithoutSTP(bp.TwoEndConn):
    target_backend = ['numpy', 'numba']

    def __init__(self, pre, post, conn, delay, g_max, **kwargs):
        self.g_max = g_max
        self.conn = conn(pre.size, post.size)
        self.pre2post = self.conn.requires('pre2post')
        self.g = self.register_constant_delay('g', post.num, delay_time=delay)
        super(SynWithoutSTP, self).__init__(pre=pre, post=post, **kwargs)

    def update(self, _t):
        g = bp.backend.zeros(self.post.num)
        for pre_id, spike in enumerate(self.pre.spike):
            for post_id in self.pre2post[pre_id]:
                if spike:
                    g[post_id] += self.g_max
                self.post.input[post_id] += self.g.pull(post_id) * self.post.not_ref[post_id]
        self.g.push(g)


num_exc = 8000
num_inh = 2000
c = 0.2
J_EI = 0.25
J_IE = 0.135
J_II = 0.2
J_B = 0.1
J_P = 0.45


@nb.njit
def get_type(id_):
    if id_ > num_exc / 2:
        return 5
    else:
        return id_ // int((num_exc * 0.1))


@nb.njit
def get_weight(i, j):
    i_type = get_type(i)
    j_type = get_type(j)

    if i_type == 5:
        if np.random.random() <= 0.9:
            return J_B
        else:
            return J_P
        # return np.random.choice([J_B, J_P], p=[0.9, 0.1])
    else:
        if i_type == j_type:
            return J_P
        else:
            return J_B


def get_conn():
    print('Initialize E2E connection ...')
    conn = np.random.random((num_exc, num_exc)) <= c
    pre_ids, post_ids = np.where(conn)
    pre_ids = np.ascontiguousarray(pre_ids)
    post_ids = np.ascontiguousarray(post_ids)

    weights = []
    for i, j in zip(pre_ids, post_ids):
        weights.append(get_weight(i, j))
    weights = np.asarray(weights)

    pre2syn = bp.connect.pre2syn(pre_ids, num_pre=num_exc)
    post2syn = bp.connect.post2syn(post_ids, num_post=num_exc)
    print('Done.')
    return dict(pre_ids=pre_ids, post_ids=post_ids,
                pre2syn=pre2syn, post2syn=post2syn,
                weights=weights)


E = LIF(num_exc, mu=23.1, tau=15., V_reset=16., monitors=['spike'])
I = LIF(num_inh, mu=21., tau=10., V_reset=13., monitors=['spike'])

E2E = SynWithSTP(pre=E, post=E, conn=get_conn(), delay=1)
I2E = SynWithoutSTP(pre=I, post=E, conn=bp.connect.FixedProb(c), delay=1, g_max=-J_EI)
E2I = SynWithoutSTP(pre=E, post=I, conn=bp.connect.FixedProb(c), delay=1, g_max=J_IE)
I2I = SynWithoutSTP(pre=I, post=I, conn=bp.connect.FixedProb(c), delay=1, g_max=-J_II)

net = bp.Network(E, I, E2E, I2E, E2I, I2I)

net.run(10., report=True)

fig, gs = bp.visualize.get_figure(row_num=2, row_len=4, col_num=1, col_len=12)
fig.add_subplot(gs[0, 0])
bp.visualize.raster_plot(E.mon.ts, E.mon.spike)
fig.add_subplot(gs[1, 0])
bp.visualize.raster_plot(I.mon.ts, I.mon.spike, show=True)
