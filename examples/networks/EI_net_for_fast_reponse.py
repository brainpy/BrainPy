# -*- coding: utf-8 -*-

""""
Implementation of the paper：

Tian, Gengshuo, et al. "Excitation-Inhibition Balanced
Neural Networks for Fast Signal Detection." Frontiers in
Computational Neuroscience 14 (2020): 79.

"""


import brainpy as bp
import brainpy.numpy as np

bp.profile.set(backend='numba',
               device='cpu',
               merge_steps=True,
               numerical_method='exponential')

num = 10000
num_inh = int(num * 0.2)
num_exc = num - num_inh
prob = 0.25

tau_E = 15.
tau_I = 10.
V_reset = 0.
V_threshold = 15.
f_E = 3.
f_I = 2.
mu_f = 0.1

tau_Es = 6.
tau_Is = 5.
JEE = 0.25
JEI = -1.
JIE = 0.4
JII = -1.


# -------
# neuron
# -------


def get_neu(tau):
    neu_ST = bp.types.NeuState(
        {'V': 0, 'sp': 0., 'inp': 0.},
    )

    @bp.integrate
    def int_f(V, t, Isyn):
        return (-V + Isyn) / tau

    def update(ST, _t_):
        V = int_f(ST['V'], _t_, ST['inp'])
        if V >= V_threshold:
            ST['sp'] = 1.
            V = V_reset
        else:
            ST['sp'] = 0.
        ST['V'] = V
        ST['inp'] = 0.

    return bp.NeuType(name='LIF',
                      requires=dict(ST=neu_ST),
                      steps=update,
                      vector_based=False)


# -------
# synapse
# -------


def get_syn(tau):
    syn_ST = bp.types.SynState(['s', 'g', 'w'])

    @bp.integrate
    def ints(s, t):
        return - s / tau

    def update(ST, _t_, pre, pre2syn):
        s = ints(ST['s'], _t_)
        for i in range(pre['sp'].shape[0]):
            if pre['sp'][i] > 0.:
                syn_ids = pre2syn[i]
                s[syn_ids] += 1.
        ST['s'] = s
        ST['g'] = ST['w'] * s

    def output(ST, post, post2syn):
        post_cond = np.zeros(len(post2syn), dtype=np.float_)
        for post_id, syn_ids in enumerate(post2syn):
            post_cond[post_id] = np.sum(ST['g'][syn_ids])
        post['inp'] += post_cond

    return bp.SynType(name='alpha_synapse',
                      requires=dict(ST=syn_ST),
                      steps=(update, output),
                      vector_based=True)


# -------
# network
# -------

E_neu = get_neu(tau_E)
E_group = bp.NeuGroup(E_neu, geometry=num_exc, monitors=['sp'])
E_group.ST['V'] = np.random.random(num) * (V_threshold - V_reset) + V_reset

I_neu = get_neu(tau_I)
I_group = bp.NeuGroup(I_neu, geometry=num_inh, monitors=['sp'])
I_group.ST['V'] = np.random.random(num_inh) * (V_threshold - V_reset) + V_reset

E_syn = get_syn(tau_Es)
IE_conn = bp.SynConn(E_syn, pre_group=E_group, post_group=I_group,
                     conn=bp.connect.FixedProb(prob=prob))
IE_conn.ST['w'] = JIE

EE_conn = bp.SynConn(E_syn, pre_group=E_group, post_group=E_group,
                     conn=bp.connect.FixedProb(prob=prob))
EE_conn.ST['w'] = JEE

I_syn = get_syn(tau_Is)
II_conn = bp.SynConn(I_syn, pre_group=I_group, post_group=I_group,
                     conn=bp.connect.FixedProb(prob=prob))
II_conn.ST['w'] = JII

EI_conn = bp.SynConn(I_syn, pre_group=I_group, post_group=E_group,
                     conn=bp.connect.FixedProb(prob=prob))
EI_conn.ST['w'] = JEI

net = bp.Network(E_group, I_group, IE_conn, EE_conn, II_conn, EI_conn)
net.run(duration=100.,
        inputs=[(E_group, 'ST.inp', f_E * np.sqrt(num) * mu_f),
                (I_group, 'ST.inp', f_I * np.sqrt(num) * mu_f)],
        report=True)

# --------------
# visualization
# --------------

fig, gs = bp.visualize.get_figure(5, 1, 2, 10)

bp.visualize.plot_raster(E_group.mon, net.ts,
                         ax=fig.add_subplot(gs[:4, 0]),
                         xlim=(0, 100),
                         show=False)

bp.visualize.plot_raster(I_group.mon, net.ts,
                         ax=fig.add_subplot(gs[4, 0]),
                         xlim=(0, 100),
                         show=True)
