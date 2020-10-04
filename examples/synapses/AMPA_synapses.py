# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import npbrain as npb
from npbrain import _numpy as np

# npb.profile.show_codgen = True


def define_ampa1(g_max=0.10, E=0., tau_decay=2.0):
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

    attrs = dict(
        ST=npb.types.SynState(['s'],
                              help='AMPA synapse state.'),
        pre=npb.types.NeuState(['sp'],
                               help='Pre-synaptic neuron state must have "sp" item.'),
        post=npb.types.NeuState(['V', 'inp'],
                                help='Pre-synaptic neuron state must have "V" and "inp" item.'),
    )

    @npb.integrate(method='euler')
    def ints(s, t):
        return - s / tau_decay

    def update(ST, t, pre, pre2syn):
        s = ints(ST['s'], t)
        spike_idx = np.where(pre['sp'] > 0.)[0]
        for i in spike_idx:
            syn_idx = pre2syn[i]
            s[syn_idx] += 1.
        ST['s'] = s
        ST.push_cond(s)

    def output(ST, post, post2syn):
        g = ST.pull_cond()
        g_val = npb.post_cond_by_post2syn(g, post2syn)
        post_val = - g_max * g_val * (post['V'] - E)
        post['inp'] += post_val

    return {'attrs': attrs, 'step_func': (update, output)}


AMPA1 = npb.SynType(name='AMPA_type1', create_func=define_ampa1, group_based=True)


def define_ampa2(g_max=0.42, E=0., alpha=0.98, beta=0.18, T=0.5, T_duration=0.5):
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

    attrs = dict(
        ST=npb.types.SynState({'s': 0., 'sp_t': -1e7},
                              help='AMPA synapse state.\n'
                                   '"s": Synaptic state.\n'
                                   '"sp_t": Pre-synaptic neuron spike time.'),
        pre=npb.types.NeuState(['sp'],
                               help='Pre-synaptic neuron state must have "sp" item.'),
        post=npb.types.NeuState(['V', 'inp'],
                                help='Pre-synaptic neuron state must have "V" and "inp" item.'),
    )

    @npb.integrate(method='euler')
    def int_s(s, t, TT):
        return alpha * TT * (1 - s) - beta * s

    def update(ST, t, pre, pre2syn):
        spike_idx = np.where(pre['sp'] > 0.)[0]
        for i in spike_idx:
            syn_idx = pre2syn[i]
            ST['sp_t'][syn_idx] = t
        TT = ((t - ST['sp_t']) < T_duration) * T
        s = np.clip(int_s(ST['s'], t, TT), 0., 1.)
        ST['s'] = s
        ST.push_cond(s)

    def output(ST, post, post2syn):
        g = ST.pull_cond()
        g_val = npb.post_cond_by_post2syn(g, post2syn)
        post_val = - g_max * g_val * (post['V'] - E)
        post['inp'] += post_val

    return {'attrs': attrs, 'step_func': (update, output)}


AMPA2 = npb.SynType(name='AMPA_type2', create_func=define_ampa2, group_based=True)


def run_ampa(cls, duration=650.):
    ampa = npb.SynConn(model=cls, num=1, monitors=['s'], delay=10.)
    ampa.pre = npb.types.NeuState(['sp'])(1)
    ampa.post = npb.types.NeuState(['V', 'inp'])(1)
    ampa.pre2syn = [[0]]
    ampa.post2syn = [[0]]
    ampa.set_schedule(['input', 'update', 'monitor'])

    net = npb.Network(ampa)
    Iext = npbrain.inputs.spike_current([10, 110, 210, 310, 410], npb.profile.dt, 1., duration=duration)
    net.run(duration, inputs=(ampa, 'pre.sp', Iext, '='), report=True)

    fig, gs = npb.visualize.get_figure(1, 1, 5, 10)
    fig.add_subplot(gs[0, 0])
    plt.plot(net.ts, ampa.mon.s[:, 0], label='s')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    # run_ampa(AMPA1)
    run_ampa(AMPA2)
