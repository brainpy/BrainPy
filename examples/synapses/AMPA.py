# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt

import brainpy as bp
from brainpy import numpy as np
from examples.neurons.LIF_model import define_LIF


def define_AMPA1_scalar(g_max=0.10, E=0., tau_decay=2.0):
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

    requires = dict(
        ST=bp.types.SynState(['s'], help='AMPA synapse state.'),
        pre=bp.types.NeuState(['sp'], help='Pre-synaptic neuron state must have "sp" item.'),
        post=bp.types.NeuState(['V', 'inp'], help='Pre-synaptic neuron state must have "V" and "inp" item.'),
    )

    @bp.integrate
    def ints(s, t):
        return - s / tau_decay

    def update(ST, _t_, pre):
        s = ints(ST['s'], _t_)
        s += pre['sp']
        ST['s'] = s

    @bp.delayed
    def output(ST, post):
        post_val = - g_max * ST['s'] * (post['V'] - E)
        post['inp'] += post_val

    return bp.SynType(name='AMPA',
                      requires=requires,
                      steps=(update, output),
                      mode='scalar')


def define_AMPA2_scalar(g_max=0.42, E=0., alpha=0.98, beta=0.18, T=0.5, T_duration=0.5):
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

    requires = {
        'ST': bp.types.SynState({'s': 0., 'sp_t': -1e7},
                                help=""" "s": Synaptic state.
                                    "sp_t": Pre-synaptic neuron spike time.
                                """),
        'pre': bp.types.NeuState(['sp'], help='Pre-synaptic neuron state must have "sp" item.'),
        'post': bp.types.NeuState(['V', 'inp'], help='Pre-synaptic neuron state must have "V" and "inp" item.'),
    }

    @bp.integrate
    def int_s(s, t, TT):
        return alpha * TT * (1 - s) - beta * s

    def update(ST, _t_, pre):
        if pre['sp'] > 0.:
            ST['sp_t'] = _t_
        TT = ((_t_ - ST['sp_t']) < T_duration) * T
        s = np.clip(int_s(ST['s'], _t_, TT), 0., 1.)
        ST['s'] = s

    @bp.delayed
    def output(ST, post):
        post_val = - g_max * ST['s'] * (post['V'] - E)
        post['inp'] += post_val

    return bp.SynType(name='AMPA',
                      requires=requires,
                      steps=(update, output),
                      mode='scalar')


def define_AMPA1_vector(g_max=0.10, E=0., tau_decay=2.0):
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

    @bp.integrate
    def ints(s, t):
        return - s / tau_decay

    # requirements
    # ------------

    requires = {
        'ST': bp.types.SynState(['s', 'g'], help='AMPA synapse state.'),
        'pre': bp.types.NeuState(['sp'], help='Pre-synaptic neuron state must have "sp" item.'),
        'post': bp.types.NeuState(['V', 'inp'], help='Pre-synaptic neuron state must have "V" and "inp" item.'),
        'pre2syn': bp.types.ListConn(help='Pre-synaptic neuron index -> synapse index'),
        'post2syn': bp.types.ListConn(help='Post-synaptic neuron index -> synapse index'),
    }

    # model logic
    # -----------

    def update(ST, _t_, pre, pre2syn):
        s = ints(ST['s'], _t_)
        spike_idx = np.where(pre['sp'] > 0.)[0]
        for i in spike_idx:
            syn_idx = pre2syn[i]
            s[syn_idx] += 1.
        ST['s'] = s
        ST['g'] = g_max * s

    @bp.delayed
    def output(ST, post, post2syn):
        post_cond = np.zeros(len(post2syn), dtype=np.float_)
        for post_id, syn_ids in enumerate(post2syn):
            post_cond[post_id] = np.sum(ST['g'][syn_ids])
        post['inp'] -= post_cond * (post['V'] - E)

    return bp.SynType(name='AMPA1',
                      requires=requires,
                      steps=(update, output),
                      mode='vector')


def define_AMPA2_vector(g_max=0.42, E=0., alpha=0.98, beta=0.18, T=0.5, T_duration=0.5):
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

    @bp.integrate
    def int_s(s, t, TT):
        return alpha * TT * (1 - s) - beta * s

    requires = dict(
        ST=bp.types.SynState({'s': 0., 'sp_t': -1e7, 'g': 0.},
                             help='AMPA synapse state.\n'
                                  '"s": Synaptic state.\n'
                                  '"sp_t": Pre-synaptic neuron spike time.'),
        pre=bp.types.NeuState(['sp'], help='Pre-synaptic neuron state must have "sp" item.'),
        post=bp.types.NeuState(['V', 'inp'], help='Pre-synaptic neuron state must have "V" and "inp" item.'),
        pre2syn=bp.types.ListConn(help='Pre-synaptic neuron index -> synapse index'),
        post2syn=bp.types.ListConn(help='Post-synaptic neuron index -> synapse index'),
    )

    def update(ST, _t_, pre, pre2syn):
        for i in np.where(pre['sp'] > 0.)[0]:
            syn_idx = pre2syn[i]
            ST['sp_t'][syn_idx] = _t_
        TT = ((_t_ - ST['sp_t']) < T_duration) * T
        s = np.clip(int_s(ST['s'], _t_, TT), 0., 1.)
        ST['s'] = s
        ST['g'] = g_max * s

    @bp.delayed
    def output(ST, post, post2syn):
        post_cond = np.zeros(len(post2syn), dtype=np.float_)
        for post_id, syn_ids in enumerate(post2syn):
            post_cond[post_id] = np.sum(ST['g'][syn_ids])
        post['inp'] -= post_cond * (post['V'] - E)

    return bp.SynType(name='AMPA',
                      requires=requires,
                      steps=(update, output),
                      mode='vector')


def define_AMPA1_matrix(g_max=0.10, E=0., tau_decay=2.0):
    @bp.integrate
    def ints(s, t):
        return - s / tau_decay

    # requirements
    # ------------

    requires = {
        'ST': bp.types.SynState(['s', 'g']),
        'pre': bp.types.NeuState(['sp']),
        'post': bp.types.NeuState(['V', 'inp']),
        'conn_mat': bp.types.MatConn(),
    }

    # model logic
    # -----------

    def update(ST, _t_, pre, conn_mat):
        s = ints(ST['s'], _t_)
        s += pre['sp'].reshape((-1, 1)) * conn_mat
        ST['s'] = s
        ST['g'] = g_max * s

    @bp.delayed
    def output(ST, post):
        post_cond = np.sum(ST['g'], axis=0)
        post['inp'] -= post_cond * (post['V'] - E)

    return bp.SynType(name='AMPA1',
                      requires=requires,
                      steps=(update, output),
                      mode='matrix')


def define_AMPA2_matrix(g_max=0.42, E=0., alpha=0.98, beta=0.18, T=0.5, T_duration=0.5):
    requires = dict(
        ST=bp.types.SynState({'s': 0., 'sp_t': -1e7, 'g': 0.}),
        pre=bp.types.NeuState(['sp']),
        post=bp.types.NeuState(['V', 'inp']),
        conn_mat=bp.types.MatConn(),
    )

    @bp.integrate
    def int_s(s, t, TT):
        return alpha * TT * (1 - s) - beta * s

    def update(ST, _t_, pre, conn_mat):
        
        spike_idxs = np.where(pre['sp'] > 0.)[0]
        ST['sp_t'][spike_idxs] = _t_
        TT = ((_t_ - ST['sp_t']) < T_duration) * T
        s = np.clip(int_s(ST['s'], _t_, TT), 0., 1.)
        ST['s'] = s
        ST['g'] = g_max * s

    @bp.delayed
    def output(ST, post):
        post_cond = np.sum(ST['g'], axis=0)
        post['input'] -= post_cond * (post['V'] - E)

    return bp.SynType(name='AMPA',
                      requires=requires,
                      steps=(update, output),
                      mode='matrix')


def run_ampa_single(define, duration=350.):
    LIF = define_LIF()
    pre = bp.NeuGroup(LIF, 2)
    post = bp.NeuGroup(LIF, 3)
    cls = define()
    ampa = bp.SynConn(model=cls, pre_group=pre, post_group=post, conn=bp.connect.All2All(),
                      monitors=['s'], delay=10.)
    ampa.runner.set_schedule(['input', 'update', 'output', 'monitor'])

    net = bp.Network(pre, ampa, post)
    Iext = bp.inputs.spike_current([10, 110, 210, 310, 410], bp.profile._dt, 1., duration=duration)
    net.run(duration, inputs=(ampa, 'pre.sp', Iext, '='), report=True)

    fig, gs = bp.visualize.get_figure(1, 1, 5, 6)
    fig.add_subplot(gs[0, 0])
    plt.plot(net.ts, ampa.mon.s[:, 0], label='s')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    bp.profile.set(backend='numba', merge_steps=True, dt=0.1, show_code=True)

    run_ampa_single(define_AMPA1_scalar)
    run_ampa_single(define_AMPA2_scalar)
