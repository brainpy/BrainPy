# -*- coding: utf-8 -*-

"""
Implementation of E/I balance network.
"""


import brainpy as bp
import numpy as np
import matplotlib.pyplot as plt


bp.profile.set(jit=True,
               device='gpu',
               numerical_method='exponential',
               show_code=True)

num_exc = 500
num_inh = 500
prob = 0.1

# -------
# neuron
# -------


tau = 10.
V_rest = -52.
V_reset = -60.
V_threshld = -50.


@bp.integrate
def int_f(V, t, Isyn):
    return (-V + V_rest + Isyn) / tau


def update(ST, _t):
    V = int_f(ST['V'], _t, ST['inp'])
    if V >= V_threshld:
        ST['sp'] = 1.
        V = V_reset
    else:
        ST['sp'] = 0.
    ST['V'] = V
    ST['inp'] = 0.


neu = bp.NeuType(name='LIF',
                 ST=bp.types.NeuState({'V': 0, 'sp': 0., 'inp': 0.}),
                 steps=update,
                 mode='scalar')

# -------
# synapse
# -------


tau_decay = 2.
JE = 1 / np.sqrt(prob * num_exc)
JI = 1 / np.sqrt(prob * num_inh)

@bp.integrate
def ints(s, t):
    return - s / tau_decay


def update(ST, _t, pre):
    s = ints(ST['s'], _t)
    s += pre['sp']
    ST['s'] = s
    ST['g'] = ST['w'] * s


def output(ST, post):
    # post['inp'] += ST['g']
    post['inp'] = post['inp'] + ST['g']


syn = bp.SynType(name='alpha_synapse',
                 ST=bp.types.SynState(['s', 'g', 'w']),
                 steps=(update, output),
                 mode='scalar')

# -------
# network
# -------

group = bp.NeuGroup(neu,
                    geometry=num_exc + num_inh,
                    monitors=['sp'])
group.ST['V'] = np.random.random(num_exc + num_inh) * (V_threshld - V_rest) + V_rest
# group.set_schedule(['update', 'monitor'])

exc_conn = bp.SynConn(syn,
                      pre_group=group[:num_exc],
                      post_group=group,
                      conn=bp.connect.FixedProb(prob=prob))
exc_conn.ST['w'] = JE

inh_conn = bp.SynConn(syn,
                      pre_group=group[num_exc:],
                      post_group=group,
                      conn=bp.connect.FixedProb(prob=prob))
inh_conn.ST['w'] = -JI

net = bp.Network(group, exc_conn, inh_conn)
net.run(duration=500., inputs=[(group, 'ST.inp', 3.)], report=True)

# --------------
# visualization
# --------------

fig, gs = bp.visualize.get_figure(4, 1, 2, 10)

fig.add_subplot(gs[:3, 0])
bp.visualize.raster_plot(net.ts, group.mon.sp, xlim=(50, 450))

fig.add_subplot(gs[3, 0])
rates = bp.measure.firing_rate(group.mon.sp, 5.)
plt.plot(net.ts, rates)
plt.xlim(50, 450)
plt.show()

