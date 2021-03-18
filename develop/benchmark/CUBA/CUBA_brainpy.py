# -*- coding: utf-8 -*-

import time

import numpy as np

import brainpy as bp

dt = 0.1
bp.profile.set(jit=True, dt=dt)

num_exc = 3200
num_inh = 800
taum = 20
taue = 5
taui = 10
Vt = -50
Vr = -60
El = -49
we = 60 * 0.27 / 10  # excitatory synaptic weight (voltage)
wi = -20 * 4.5 / 10  # inhibitory synaptic weight
ref = 5.0

neu_ST = bp.types.NeuState(
    {'sp_t': -1e7,
     'V': 0.,
     'sp': 0.,
     'ge': 0.,
     'gi': 0.}
)


@bp.integrate
def int_ge(ge, t):
    return -ge / taue


@bp.integrate
def int_gi(gi, t):
    return -gi / taui


@bp.integrate
def int_V(V, t, ge, gi):
    return (ge + gi - (V - El)) / taum


def neu_update(ST, _t):
    ST['ge'] = int_ge(ST['ge'], _t)
    ST['gi'] = int_gi(ST['gi'], _t)

    if _t - ST['sp_t'] > ref:
        V = int_V(ST['V'], _t, ST['ge'], ST['gi'])
        if V >= Vt:
            ST['V'] = Vr
            ST['sp'] = 1.
            ST['sp_t'] = _t
        else:
            ST['V'] = V
            ST['sp'] = 0.
    else:
        ST['sp'] = 0.


neuron = bp.NeuType(name='CUBA', ST=neu_ST, steps=neu_update, mode='scalar')


def update1(pre, post, pre2post):
    for pre_id in range(len(pre2post)):
        if pre['sp'][pre_id] > 0.:
            post_ids = pre2post[pre_id]
            for i in post_ids:
                post['ge'][i] += we


exc_syn = bp.SynType('exc_syn', steps=update1, ST=bp.types.SynState())


def update2(pre, post, pre2post):
    for pre_id in range(len(pre2post)):
        if pre['sp'][pre_id] > 0.:
            post_ids = pre2post[pre_id]
            for i in post_ids:
                post['gi'][i] += wi


inh_syn = bp.SynType('inh_syn', steps=update2, ST=bp.types.SynState())

group = bp.NeuGroup(neuron,
                    size=num_exc + num_inh,
                    monitors=['sp'])
group.ST['V'] = Vr + np.random.rand(num_exc + num_inh) * (Vt - Vr)

exc_conn = bp.TwoEndConn(exc_syn,
                         pre=group[:num_exc],
                         post=group,
                         conn=bp.connect.FixedProb(prob=0.02))

inh_conn = bp.TwoEndConn(inh_syn,
                         pre=group[num_exc:],
                         post=group,
                         conn=bp.connect.FixedProb(prob=0.02))

net = bp.Network(group, exc_conn, inh_conn, mode='repeat')
t0 = time.time()
# net.run(5 * 1000., report_percent=1., report=True)
net.run(1250., report=True)
net.run((1250., 2500.), report=True)
net.run((2500., 3750.), report=True)
net.run((3750., 5000.), report=True)
print('Used time {} s.'.format(time.time() - t0))

bp.visualize.raster_plot(net.ts, group.mon.sp, show=True)
