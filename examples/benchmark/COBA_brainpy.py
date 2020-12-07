# -*- coding: utf-8 -*-
import time

import numpy as np

import brainpy as bp

dt = 0.05
bp.profile.set(backend='numba', dt=dt)

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

neu_ST = bp.types.NeuState(
    {'sp_t': -1e7,
     'V': 0.,
     'spike': 0.,
     'ge': 0.,
     'gi': 0.}
)

def neu_update(ST, _t_):
    ge = ST['ge']
    gi = ST['gi']
    ge -= ge / taue
    gi -= gi / taui
    ST['ge'] = ge
    ST['gi'] = gi

    if (_t_ - ST['sp_t']) > ref:
        V = ST['V']
        dvdt = (ge * (Erev_exc - V) + gi * (Erev_inh - V) + (El - V) + I) / taum
        V += dvdt * dt
        ST['V'] = V
        if V >= Vt:
            ST['V'] = Vr
            ST['spike'] = 1.
            ST['sp_t'] = _t_
        else:
            ST['spike'] = 0.
            ST['V'] = V
    else:
        ST['spike'] = 0.


neuron = bp.NeuType(name='COBA',
                    requires=dict(ST=neu_ST),
                    steps=neu_update,
                    mode='scalar')


def update1(ST, pre, post, pre2post):
    for pre_id in range(len(pre2post)):
        if pre['spike'][pre_id] > 0.:
            post_ids = pre2post[pre_id]
            post['ge'][post_ids] += we


exc_syn = bp.SynType('exc_syn',
                     steps=update1,
                     requires=dict(ST=bp.types.SynState([])))


def update2(ST, pre, post, pre2post):
    for pre_id in range(len(pre2post)):
        if pre['spike'][pre_id] > 0.:
            post_ids = pre2post[pre_id]
            post['gi'][post_ids] += wi


inh_syn = bp.SynType('inh_syn',
                     steps=update2,
                     requires=dict(ST=bp.types.SynState([])))

group = bp.NeuGroup(neuron,
                    geometry=num_exc + num_inh,
                    monitors=['spike'])
group.ST['V'] = np.random.randn(num_exc + num_inh) * 5. - 55.

exc_conn = bp.SynConn(exc_syn,
                      pre_group=group[:num_exc],
                      post_group=group,
                      conn=bp.connect.FixedProb(prob=0.02))

inh_conn = bp.SynConn(inh_syn,
                      pre_group=group[num_exc:],
                      post_group=group,
                      conn=bp.connect.FixedProb(prob=0.02))

net = bp.Network(group, exc_conn, inh_conn, mode='repeat')
t0 = time.time()
# net.run(3000., report=True)
net.run(1000., report=True)
net.run((1000., 2000.), report=True)
net.run((2000., 3000.), report=True)
print('Used time {} s.'.format(time.time() - t0))

bp.visualize.raster_plot(net.ts, group.mon.spike, show=True)
