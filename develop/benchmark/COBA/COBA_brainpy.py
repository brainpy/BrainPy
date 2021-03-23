# -*- coding: utf-8 -*-


import time

import numpy as np

import brainpy as bp

dt = 0.05
bp.backend.set('numba', dt=dt)

# Parameters
num_exc = 3200
num_inh = 800
taum = 20
taue = 5
taui = 10
Vt = -50
Vr = -60
El = -60
Erev_exc = 0.
Erev_inh = -80.
I = 20.
we = 0.6  # excitatory synaptic weight (voltage)
wi = 6.7  # inhibitory synaptic weight
ref = 5.0


class LIF(bp.NeuGroup):
    target_backend = ['numpy', 'numba']

    def __init__(self, size, **kwargs):
        # variables
        self.V = bp.backend.zeros(size)
        self.spike = bp.backend.zeros(size)
        self.ge = bp.backend.zeros(size)
        self.gi = bp.backend.zeros(size)
        self.input = bp.backend.zeros(size)
        self.t_last_spike = bp.backend.ones(size) * -1e7

        super(LIF, self).__init__(size=size, **kwargs)

    @staticmethod
    @bp.odeint(method='euler')
    def int_g(ge, gi, t):
        dge = - ge / taue
        dgi = - gi / taui
        return dge, dgi

    @staticmethod
    @bp.odeint(method='euler')
    def int_V(V, t, ge, gi):
        dV = (ge * (Erev_exc - V) + gi * (Erev_inh - V) + El - V + I) / taum
        return dV

    def update(self, _t):
        self.ge, self.gi = self.int_g(self.ge, self.gi, _t)
        for i in range(self.size[0]):
            self.spike[i] = 0.
            if (_t - self.t_last_spike[i]) > ref:
                V = self.int_V(self.V[i], _t, self.ge[i], self.gi[i])
                if V >= Vt:
                    self.V[i] = Vr
                    self.spike[i] = 1.
                    self.t_last_spike[i] = _t
                else:
                    self.V[i] = V
            self.input[i] = I


class EecSyn(bp.TwoEndConn):
    target_backend = ['numpy', 'numba']

    def __init__(self, pre, post, conn, **kwargs):
        self.conn = conn(pre.size, post.size)
        self.pre2post = self.conn.requires('pre2post')
        super(EecSyn, self).__init__(pre=pre, post=post, **kwargs)

    def update(self, _t):
        for pre_id, spike in enumerate(self.pre.spike):
            if spike > 0:
                for post_i in self.pre2post[pre_id]:
                    self.post.ge[post_i] += we


class InhSyn(bp.TwoEndConn):
    target_backend = ['numpy', 'numba']

    def __init__(self, pre, post, conn, **kwargs):
        self.conn = conn(pre.size, post.size)
        self.pre2post = self.conn.requires('pre2post')
        super(InhSyn, self).__init__(pre=pre, post=post, **kwargs)

    def update(self, _t):
        for pre_id, spike in enumerate(self.pre.spike):
            if spike > 0:
                for post_i in self.pre2post[pre_id]:
                    self.post.gi[post_i] += wi


E_group = LIF(num_exc, monitors=['spike'])
E_group.V = np.random.randn(num_exc) * 5. - 55.
I_group = LIF(num_inh, monitors=['spike'])
I_group.V = np.random.randn(num_inh) * 5. - 55.
E2E = EecSyn(pre=E_group, post=E_group, conn=bp.connect.FixedProb(0.02))
E2I = EecSyn(pre=E_group, post=I_group, conn=bp.connect.FixedProb(0.02))
I2E = InhSyn(pre=I_group, post=E_group, conn=bp.connect.FixedProb(0.02))
I2I = InhSyn(pre=I_group, post=I_group, conn=bp.connect.FixedProb(0.02))

net = bp.Network(E_group, I_group, E2E, E2I, I2E, I2I)
t0 = time.time()

net.run(5000., report=True)
print('Used time {} s.'.format(time.time() - t0))

bp.visualize.raster_plot(net.ts, E_group.mon.spike, show=True)
