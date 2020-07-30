import time

import matplotlib.pyplot as plt
import numpy as np

import npbrain as nn
nn.profile.set_backend('numba')
nn.profile.predefine_signature = False

dt = 0.1
nn.profile.set_dt(dt)
np.random.seed(12345)

# ------------------
# parameters
# ------------------
num_exc = 3200
num_inh = 800
monitor = 'spike'

unit = 1e6 * 1.
Cm = 200 / unit
gl = 10. / unit
g_na = 20. * 1000 / unit
g_kd = 6. * 1000 / unit

El = -60.
EK = -90.
ENa = 50.
VT = -63.
Vt = -20.
# Time constants
taue = 5.
taui = 10.
# Reversal potentials
Ee = 0.
Ei = -80.
# excitatory synaptic weight
we = 6. / unit
# inhibitory synaptic weight
wi = 67. / unit


def COBA_HH(geometry, name='COBA_HH'):
    var2index = dict(V=0, m=1, h=2, n=3, ge=4, gi=5)
    num, geometry = nn.format_geometry(geometry)

    # V, m, h, n, ge, gi
    state = nn.initial_neu_state(6, num)
    state[0] = El + (np.random.randn(num_exc + num_inh) * 5 - 5)
    state[4] = (np.random.randn(num_exc + num_inh) * 1.5 + 4) * 10. / unit
    state[5] = (np.random.randn(num_exc + num_inh) * 12 + 20) * 10. / unit

    def update_state(neu_state, t):
        # get neuron state
        V = neu_state[0]
        m = neu_state[1]
        h = neu_state[2]
        n = neu_state[3]
        ge = neu_state[4]
        gi = neu_state[5]

        # calculate neuron state
        m_alpha = 0.32 * (13 - V + VT) / (np.exp((13 - V + VT) / 4) - 1.)
        m_beta = 0.28 * (V - VT - 40) / (np.exp((V - VT - 40) / 5) - 1)
        h_alpha = 0.128 * np.exp((17 - V + VT) / 18)
        h_beta = 4. / (1 + np.exp((40 - V + VT) / 5))
        n_alpha = 0.032 * (15 - V + VT) / (np.exp((15 - V + VT) / 5) - 1.)
        n_beta = .5 * np.exp((10 - V + VT) / 40)

        # m channel
        fm = (m_alpha * (1 - m) - m_beta * m)
        dfm_dm = - m_alpha - m_beta
        m = m + (np.exp(dfm_dm * dt) - 1) / dfm_dm * fm
        neu_state[1] = m

        # h channel
        fh = (h_alpha * (1 - h) - h_beta * h)
        dfh_dh = - h_alpha - h_beta
        h = h + (np.exp(dfh_dh * dt) - 1) / dfh_dh * fh
        neu_state[2] = h

        # n channel
        fn = (n_alpha * (1 - n) - n_beta * n)
        dfn_dn = - n_alpha - h_beta
        n = n + (np.exp(dfn_dn * dt) - 1) / dfn_dn * fn
        neu_state[3] = n

        # ge
        fge = - ge / taue
        dfge_dge = - 1 / taue
        ge = ge + (np.exp(dfge_dge * dt) - 1) / dfge_dge * fge
        neu_state[4] = ge

        # gi
        fgi = - gi / taui
        dfgi_dgi = - 1 / taui
        gi = gi + (np.exp(dfgi_dgi * dt) - 1) / dfgi_dgi * fgi
        neu_state[5] = gi

        # V
        g_na_ = g_na * (m * m * m) * h
        g_kd_ = g_kd * (n * n * n * n)
        fv = (gl * (El - V) + ge * (Ee - V) + gi * (Ei - V) -
              g_na_ * (V - ENa) - g_kd_ * (V - EK)) / Cm
        dfv_dv = (-gl - ge - gi - g_na_ - g_kd_) / Cm
        V = V + (np.exp(dfv_dv * dt) - 1) / dfv_dv * fv
        neu_state[0] = V

        # spike
        nn.judge_spike(neu_state, Vt, t)

    return nn.Neurons(**locals())


exc_pre, exc_post, exc_acs = nn.connect.fixed_prob(
    num_exc, num_exc + num_inh, 0.02, include_self=False)
exc_anchors = np.zeros((2, num_exc + num_inh), dtype=np.int32)
exc_anchors[:, :num_exc] = exc_acs

inh_pre, inh_post, inh_anchors = nn.connect.fixed_prob(
    list(range(num_exc, num_exc + num_inh)), num_exc + num_inh, 0.02, include_self=False)


def Synapse(pre, post, delay=None):
    var2index = dict()
    num_pre = pre.num
    num_post = post.num

    num = len(exc_pre)
    state = nn.initial_syn_state(delay, num_post=num_post * 2, num_syn=num)

    def update_state(syn_state, t, delay_idx, pre_state, post_state):
        spike_idx = np.where(pre_state[-3] > 0)[0]
        g = np.zeros(num_post * 2)
        for i_ in spike_idx:
            if i_ < num_exc:
                idx = exc_anchors[:, i_]
                exc_post_idx = exc_post[idx[0]: idx[1]]
                for pi in exc_post_idx:
                    g[pi] += we
            else:
                idx = inh_anchors[:, i_]
                inh_post_idx = inh_post[idx[0]: idx[1]]
                for pi in inh_post_idx:
                    g[num_post + pi] += wi
        syn_state[1][delay_idx] = g

    def output_synapse(syn_state, output_idx, pre_state, post_state):
        syn_val = syn_state[1][output_idx]
        ge = syn_val[:num_post]
        gi = syn_val[num_post:]
        for idx in range(num_post):
            post_state[4, idx] += ge[idx]
            post_state[5, idx] += gi[idx]

    return nn.Synapses(**locals())


neurons = COBA_HH(num_exc + num_inh)
syn = Synapse(neurons, neurons)
if monitor == 'V':
    mon = nn.StateMonitor(neurons, ['V'])
else:
    mon = nn.StateMonitor(neurons, ['spike'])
net = nn.Network(syn=syn, neu=neurons, mon=mon)

t0 = time.time()
net.run(10 * 1000., report=True)
print('Used time {} s.'.format(time.time() - t0))

if monitor == 'V':
    nn.visualize.plot_potential(mon, net.run_time(), neuron_index=[1, 10, 100])
    plt.legend()
    plt.show()
else:
    index, time = nn.raster_plot(mon, net.run_time())
    plt.plot(time, index, ',k')
    plt.xlabel('Time (ms)')
    plt.ylabel('Neuron index')
    plt.show()
