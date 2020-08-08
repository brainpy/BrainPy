import time

import matplotlib.pyplot as plt
import numpy as np

import npbrain as nn

nn.profile.set_backend('numba')
nn.profile.define_signature = False

dt = 0.05
nn.profile.set_dt(dt)
np.random.seed(12345)

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


def COBA(geometry, ref=5.0, name='COBA'):
    var2index = dict(V=0, ge=1, gi=2)
    num, geometry = nn.format_geometry(geometry)

    state = nn.init_neu_state(num_neu=num, variables=len(var2index))
    state[0] = np.random.randn(num) * 5. - 55.

    def update_state(neu_state, t):
        not_in_ref = (t - neu_state[-2]) > ref
        neu_state[-5] = not_in_ref
        not_ref_idx = np.where(not_in_ref)[0]
        for idx in not_ref_idx:
            v = neu_state[0, idx]
            ge = neu_state[1, idx]
            gi = neu_state[2, idx]
            ge -= ge / taue * dt
            gi -= gi / taui * dt
            v += (ge * (Erev_exc - v) + gi * (Erev_inh - v) - (v - El) + I) / taum * dt
            neu_state[0, idx] = v
            neu_state[1, idx] = ge
            neu_state[2, idx] = gi
        # judge spike
        spike_idx = nn.judge_spike(neu_state, Vt, t)
        for idx in spike_idx:
            neu_state[0, idx] = Vr
            neu_state[-5, idx] = 0.

    return nn.Neurons(**locals())


exc_pre, exc_post, exc_acs = nn.connect.fixed_prob(
    num_exc, num_exc + num_inh, 0.02, include_self=False)
exc_anchors = np.zeros((2, num_exc + num_inh), dtype=np.int32)
exc_anchors[:, :num_exc] = exc_acs

inh_pre, inh_post, inh_anchors = nn.connect.fixed_prob(
    list(range(num_exc, num_exc + num_inh)),
    num_exc + num_inh, 0.02, include_self=False)


def Synapse(pre, post, delay=None):
    var2index = dict()

    num_pre, num_post, num = pre.num, post.num, len(exc_pre)
    delay_state = nn.init_delay_state(num_post=num_post * 2, delay=delay)

    def update_state(delay_st, delay_idx, pre_state):
        pre_spike = pre_state[-3]
        g = np.zeros(num_post * 2)
        for pre_id in range(num_pre):
            if pre_spike[pre_id] > 0.:
                if pre_id < num_exc:
                    idx = exc_anchors[:, pre_id]
                    exc_post_idx = exc_post[idx[0]: idx[1]]
                    for idx in exc_post_idx:
                        g[idx] += we
                else:
                    idx = inh_anchors[:, pre_id]
                    inh_post_idx = inh_post[idx[0]: idx[1]]
                    for idx in inh_post_idx:
                        g[idx + num_post] += wi
        delay_st[delay_idx] = g

    def output_synapse(delay_st, output_idx, post_state):
        syn_val = delay_st[output_idx]
        ge = syn_val[:num_post]
        gi = syn_val[num_post:]
        for idx in range(num_post):
            post_state[1, idx] += ge[idx] * post_state[-5, idx]
            post_state[2, idx] += gi[idx] * post_state[-5, idx]

    return nn.Synapses(**locals())


neurons = COBA(num_exc + num_inh)
syn = Synapse(neurons, neurons)
mon = nn.StateMonitor(neurons, ['spike'])
net = nn.Network(syn=syn, neu=neurons, mon=mon)

t0 = time.time()
net.run(5 * 1000., report=True)
print('Used time {} s.'.format(time.time() - t0))

index, time = nn.raster_plot(mon, net.run_time())
plt.plot(time, index, ',k')
plt.xlabel('Time (ms)')
plt.ylabel('Neuron index')
plt.show()
