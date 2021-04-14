# -*- coding: utf-8 -*-


import time
import numpy as np
import brainpy as bp

np.random.seed(1234)
dt = 0.05
bp.backend.set('numba', dt=dt)

# Parameters
num = 4000 * 15
num_exc = int(num * 0.75)
num_inh = int(num * 0.25)
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
    target_backend = ['numpy', 'numba', 'numba-cuda']

    def __init__(self, size, **kwargs):
        # variables
        self.V = bp.ops.zeros(size)
        self.spike = bp.ops.zeros(size)
        self.ge = bp.ops.zeros(size)
        self.gi = bp.ops.zeros(size)
        self.input = bp.ops.zeros(size)
        self.t_last_spike = bp.ops.ones(size) * -1e7

        super(LIF, self).__init__(size=size, **kwargs)

    @staticmethod
    @bp.odeint
    def int_ge(ge, t):
        dge = - ge / taue
        return dge

    @staticmethod
    @bp.odeint
    def int_gi(gi, t):
        dgi = - gi / taui
        return dgi

    @staticmethod
    @bp.odeint
    def int_V(V, t, ge, gi):
        dV = (ge * (Erev_exc - V) + gi * (Erev_inh - V) + El - V + I) / taum
        return dV

    def update(self, _t):
        for i in range(self.num):
            self.ge[i] = self.int_ge(self.ge[i], _t)
            self.gi[i] = self.int_gi(self.gi[i], _t)
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


class ExcSyn(bp.TwoEndConn):
    target_backend = ['numpy', 'numba', 'numba-cuda']

    def __init__(self, pre, post, conn, **kwargs):
        self.conn = conn(pre.size, post.size)
        self.post_ids, self.pre_slice_syn = self.conn.requires('post_ids', 'pre_slice_syn')
        super(ExcSyn, self).__init__(pre=pre, post=post, **kwargs)

    def update(self, _t):
        for pre_id in range(self.pre.num):
            if self.pre.spike[pre_id]:
                start, end = self.pre_slice_syn[pre_id]
                for post_i in self.post_ids[start: end]:
                    self.post.ge[post_i] += we


class InhSyn(bp.TwoEndConn):
    target_backend = ['numpy', 'numba', 'numba-cuda']

    def __init__(self, pre, post, conn, **kwargs):
        self.conn = conn(pre.size, post.size)
        self.post_ids, self.pre_slice_syn = self.conn.requires('post_ids', 'pre_slice_syn')
        super(InhSyn, self).__init__(pre=pre, post=post, **kwargs)

    def update(self, _t):
        for pre_id in range(self.pre.num):
            if self.pre.spike[pre_id]:
                start, end = self.pre_slice_syn[pre_id]
                for post_i in self.post_ids[start: end]:
                    self.post.gi[post_i] += wi


E_group = LIF(num_exc, monitors=[])
E_group.V = np.random.randn(num_exc) * 5. - 55.
I_group = LIF(num_inh, monitors=[])
I_group.V = np.random.randn(num_inh) * 5. - 55.
E2E = ExcSyn(pre=E_group, post=E_group, conn=bp.connect.FixedProb(0.02))
E2I = ExcSyn(pre=E_group, post=I_group, conn=bp.connect.FixedProb(0.02))
I2E = InhSyn(pre=I_group, post=E_group, conn=bp.connect.FixedProb(0.02))
I2I = InhSyn(pre=I_group, post=I_group, conn=bp.connect.FixedProb(0.02))

net = bp.Network(E_group, I_group, E2E, E2I, I2E, I2I)
t0 = time.time()

net.run(5000., report=True)
print('Used time {} s.'.format(time.time() - t0))

# bp.visualize.raster_plot(net.ts, E_group.mon.spike, show=True)
