# -*- coding: utf-8 -*-

import numpy as np
import npbrain as nn
from numba import prange

npbrain.profile.set_numba(parallel=True)
npbrain.profile.set_backend('numba')
npbrain.profile.set_dt(dt=0.1)


def LIF(geometry, method=None, tau=10., Vr=0., Vth=10., noise=0.,
        ref=0., name='LIF'):
    var2index = {'V': 0}
    num, geometry = nn.format_geometry(geometry)

    state = nn.init_neu_state(1, num)
    state[0] = Vr
    dt = npbrain.profile.get_dt()

    def update_state(neu_state, t):
        for idx in prange(int(num)):
            if (t - neu_state[-2, idx]) > ref:
                v = neu_state[0, idx]
                Isyn = neu_state[-1, idx]
                v = v + (-v + Vr + Isyn) / tau * dt
                if v >= Vth:
                    neu_state[-5, idx] = 0.  # refractory state
                    neu_state[-3, idx] = 1.  # spike state
                    neu_state[-2, idx] = t  # spike time
                    v = Vr
                else:
                    neu_state[-5, idx] = 1.
                    neu_state[-3, idx] = 0.
                neu_state[0, idx] = v  # membrane potential
            else:
                neu_state[-5, idx] = 0.
                neu_state[-3, idx] = 0.
    return nn.Neurons(**locals())


def VoltageJumpSynapse(pre, post, weights, connection, delay=None):
    num_pre = pre.num
    num_post = post.num
    var2index = dict()

    pre_ids, post_ids, anchors = connection
    num = len(pre_ids)
    state = nn.init_syn_state(delay, num_pre, num_post, num)

    def update_state(syn_state, t, delay_idx):
        g = np.zeros(num_post)
        for pre_idx in prange(int(num_pre)):
            if syn_state[0][-1, pre_idx] > 0.:
                idx = anchors[:, pre_idx]
                post_idx = post_ids[idx[0]: idx[1]]
                for idx in post_idx:
                    g[idx] += weights
        # update `conductance`
        syn_state[1][delay_idx] = g

    def output_synapse(syn_state, output_idx, post_neu_state):
        g_val = syn_state[1][output_idx]
        for idx in prange(int(num_post)):
            post_neu_state[0, idx] += g_val[idx] * post_neu_state[-5, idx]

    def collect_spike(syn_state, pre_neu_state, post_neu_state):
        for i in prange(int(num_post)):
            syn_state[0][-1, i] = pre_neu_state[-3, i]

    return nn.Synapses(**locals())


Vr = 10
theta = 20
tau = 20
delta = 2
taurefr = 2
duration = 10 * 1000
N = 2000
sparseness = 0.1
J = .1
muext = 25
sigmaext = 1.


lif1 = LIF(N, Vr=Vr, Vth=theta, tau=tau, ref=taurefr, noise=sigmaext * np.sqrt(tau))
lif2 = LIF(N, Vr=Vr, Vth=theta, tau=tau, ref=taurefr, noise=sigmaext * np.sqrt(tau))
lif3 = LIF(N, Vr=Vr, Vth=theta, tau=tau, ref=taurefr, noise=sigmaext * np.sqrt(tau))
conn = nn.connect.fixed_prob(lif1.num, lif2.num, sparseness, False)
syn1 = VoltageJumpSynapse(lif1, lif2, J, delay=delta, connection=conn)
conn = nn.connect.fixed_prob(lif2.num, lif3.num, sparseness, False)
syn2 = VoltageJumpSynapse(lif2, lif3, J, delay=delta, connection=conn)


net = nn.Network(syn1, syn2, lif1, lif2, lif3)
print('Start run')
net.run(duration, inputs=(lif1, muext), report=True)


# t0 = time.time()
# for t in bnp.arange(0, duration, nn.profile.get_dt()):
#     lif.state[-1] = muext
#     syn.collect_spike(syn.state, lif.state, lif.state)
#     syn.update_state(syn.state, t, syn.delay_idx())
#     syn.output_synapse(syn.state, syn.output_idx(), lif.state, )
#     lif.update_state(lif.state, t)
# t1 = time.time()
# print('Time : ', t1 -  t0)

