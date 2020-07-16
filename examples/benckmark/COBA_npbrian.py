import time

import matplotlib.pyplot as plt
import numpy as np

import npbrain as nn

nn.profile.set_backend('numba')

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

    state = nn.initial_neu_state(3, num)
    state[0] = np.random.randn(num) * 5. - 55.

    def update_state(neu_state, t):
        # get neuron index not in refractory
        not_ref = (t - neu_state[-2]) > ref
        not_ref_idx = np.where(not_ref)[0]
        neu_state[-5] = not_ref
        # get neuron state
        V = neu_state[0][not_ref_idx]
        ge = neu_state[1][not_ref_idx]
        gi = neu_state[2][not_ref_idx]
        # calculate neuron state
        ge -= ge / taue * dt
        gi -= gi / taui * dt
        V += (ge * (Erev_exc - V) + gi * (Erev_inh - V) - (V - El) + I) / taum * dt
        neu_state[0][not_ref_idx] = V
        neu_state[1][not_ref_idx] = ge
        neu_state[2][not_ref_idx] = gi
        spike_idx = nn.judge_spike(neu_state, Vt, t)
        neu_state[0][spike_idx] = Vr

    return nn.Neurons(**locals())


exc_pre, exc_post, exc_acs = nn.conn.fixed_prob(num_exc, num_exc + num_inh, 0.02, include_self=False)
exc_anchors = np.zeros((2, num_exc + num_inh), dtype=np.int32)
exc_anchors[:, :num_exc] = exc_acs

inh_pre, inh_post, inh_anchors = nn.conn.fixed_prob(list(range(num_exc, num_exc + num_inh)),
                                                    num_exc + num_inh, 0.02, include_self=False)


def Synapse(pre, post, delay=None, name='nromal_synapse'):
    var2index = dict()
    num_pre = pre.num
    num_post = post.num

    num = len(exc_pre)
    state = nn.initial_syn_state(delay, num_pre, num_post * 2, num)

    def update_state(syn_state, var_index, t):
        spike = syn_state[0][-1]
        spike_idx = np.where(spike > 0.)[0]
        # get post-synaptic values
        g = np.zeros(num_post * 2)
        g2 = np.zeros(num_post)
        for i_ in spike_idx:
            if i_ < num_exc:
                exc_start, exc_end = exc_anchors[:, i_]
                exc_post_idx = exc_post[exc_start: exc_end]
                g[exc_post_idx] += we
            else:
                inh_start, inh_end = inh_anchors[:, i_]
                inh_post_idx = inh_post[inh_start: inh_end]
                g2[inh_post_idx] += wi
        g[num_post:] = g2
        nn.record_conductance(syn_state, var_index, g)

    def output_synapse(syn_state, var_index, post_neu_state):
        output_idx = var_index[-2]
        syn_val = syn_state[output_idx[0]][output_idx[1]]
        not_in_ref = post_neu_state[-5]
        post_neu_state[1] += syn_val[:num_post] * not_in_ref
        post_neu_state[2] += syn_val[num_post:] * not_in_ref

    return nn.Synapses(**locals())


neurons = COBA(num_exc + num_inh)
syn = Synapse(neurons, neurons)
mon = nn.StateMonitor(neurons, ['spike', 'spike_time'])
net = nn.Network(syn=syn, neu=neurons, mon=mon)

t0 = time.time()
net.run(10 * 1000., report=True)
print('Used time {} s.'.format(time.time() - t0))

index, time = nn.raster_plot(mon)
plt.plot(time, index, ',k')
plt.xlabel('Time (ms)')
plt.ylabel('Neuron index')
plt.show()
