# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt

import npbrain as nb
from npbrain import numpy as np


def AMPA1(g_max=0.10, E=0., tau_decay=2.0):
    """AMPA conductance-based synapse (type 1).

    .. math::

        I_{syn}&=\\bar{g}_{syn} s (V-E_{syn})

        \\frac{d s}{d t}&=-\\frac{s}{\\tau_{decay}}+\\sum_{k} \\delta(t-t_{j}^{k})

    Parameters
    ----------
    g_max : float
        Maximum conductance.
    E : float
        Reversal potential.
    tau_decay : float
        Tau for decay.
    """

    @nb.integrate
    def ints(s, t):
        return - s / tau_decay

    # requirements
    # ------------

    requires = {
        'ST': nb.types.SynState(['s', 'g'], help='AMPA synapse state.'),
        'pre': nb.types.NeuState(['sp'], help='Pre-synaptic neuron state must have "sp" item.'),
        'post': nb.types.NeuState(['V', 'inp'], help='Pre-synaptic neuron state must have "V" and "inp" item.'),
        'pre2syn': nb.types.ListConn(help='Pre-synaptic neuron index -> synapse index'),
        'post2syn': nb.types.ListConn(help='Post-synaptic neuron index -> synapse index'),
    }

    # model logic
    # -----------

    @nb.delay_push
    def update(ST, _t_, pre, pre2syn):
        s = ints(ST['s'], _t_)
        spike_idx = np.where(pre['sp'] > 0.)[0]
        for i in spike_idx:
            syn_idx = pre2syn[i]
            s[syn_idx] += 1.
        ST['s'] = s
        ST['g'] = g_max * s

    @nb.delay_pull
    def output(ST, post, post2syn):
        post_cond = np.zeros(len(post2syn), dtype=np.float_)
        for post_id, syn_ids in enumerate(post2syn):
            post_cond[post_id] = np.sum(ST['g'][syn_ids])
        post['inp'] -= post_cond * (post['V'] - E)

    return nb.SynType(name='AMPA1', requires=requires, steps=(update, output), vector_based=True)


def AMPA2(g_max=0.42, E=0., alpha=0.98, beta=0.18, T=0.5, T_duration=0.5):
    """AMPA conductance-based synapse (type 2).

    .. math::

        I_{syn}&=\\bar{g}_{syn} s (V-E_{syn})

        \\frac{ds}{dt} &=\\alpha[T](1-s)-\\beta s

    Parameters
    ----------
    g_max : float
        Maximum conductance.
    E : float
        Reversal potential.
    alpha
    beta
    T
    T_duration
    """

    @nb.integrate
    def int_s(s, t, TT):
        return alpha * TT * (1 - s) - beta * s

    requires = dict(
        ST=nb.types.SynState({'s': 0., 'sp_t': -1e7, 'g': 0.},
                             help='AMPA synapse state.\n'
                                  '"s": Synaptic state.\n'
                                  '"sp_t": Pre-synaptic neuron spike time.'),
        pre=nb.types.NeuState(['sp'], help='Pre-synaptic neuron state must have "sp" item.'),
        post=nb.types.NeuState(['V', 'inp'], help='Pre-synaptic neuron state must have "V" and "inp" item.'),
        pre2syn=nb.types.ListConn(help='Pre-synaptic neuron index -> synapse index'),
        post2syn=nb.types.ListConn(help='Post-synaptic neuron index -> synapse index'),
    )

    @nb.delay_push
    def update(ST, _t_, pre, pre2syn):
        for i in np.where(pre['sp'] > 0.)[0]:
            syn_idx = pre2syn[i]
            ST['sp_t'][syn_idx] = _t_
        TT = ((_t_ - ST['sp_t']) < T_duration) * T
        s = np.clip(int_s(ST['s'], _t_, TT), 0., 1.)
        ST['s'] = s
        ST['g'] = g_max * s

    @nb.delay_pull
    def output(ST, post, post2syn):
        post_cond = np.zeros(len(post2syn), dtype=np.float_)
        for post_id, syn_ids in enumerate(post2syn):
            post_cond[post_id] = np.sum(ST['g'][syn_ids])
        post['inp'] -= post_cond * (post['V'] - E)

    return nb.SynType(name='AMPA', requires=requires, steps=(update, output), vector_based=True)


def run_ampa_group(cls, duration=650.):
    ampa = nb.SynConn(create_func=cls, num=1, monitors=['s'], delay=10.)
    ampa.pre = nb.types.NeuState(['sp'])(1)
    ampa.post = nb.types.NeuState(['V', 'inp'])(1)
    ampa.pre2syn = ampa.requires['pre2syn'].make_copy([[0]])
    ampa.post2syn = ampa.requires['post2syn'].make_copy([[0]])
    # ampa.set_schedule(['input', 'update', 'monitor'])

    net = nb.Network(ampa)
    Iext = nb.inputs.spike_current([10, 110, 210, 310, 410], nb.profile._dt, 1., duration=duration)
    net.run(duration, inputs=(ampa, 'pre.sp', Iext, '='), report=True)

    fig, gs = nb.visualize.get_figure(1, 1, 5, 10)
    fig.add_subplot(gs[0, 0])
    plt.plot(net.ts, ampa.mon.s[:, 0], label='s')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    nb.profile.set(backend='numba', )

    run_ampa_group(AMPA1)
    run_ampa_group(AMPA2)
