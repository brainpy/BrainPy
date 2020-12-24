# -*- coding: utf-8 -*-

import brainpy as bp
import numpy as np
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
        pre=bp.types.NeuState(['spike'], help='Pre-synaptic neuron state must have "sp" item.'),
        post=bp.types.NeuState(['V', 'input'], help='Pre-synaptic neuron state must have "V" and "inp" item.'),
    )

    @bp.integrate
    def ints(s, t):
        return - s / tau_decay

    def update(ST, _t_, pre):
        s = ints(ST['s'], _t_)
        s += pre['spike']
        ST['s'] = s

    @bp.delayed
    def output(ST, post):
        post_val = - g_max * ST['s'] * (post['V'] - E)
        post['input'] += post_val

    return bp.SynType(name='AMPA',
                      ST=bp.types.SynState(['s']),
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
        'pre': bp.types.NeuState(['spike'], help='Pre-synaptic neuron state must have "sp" item.'),
        'post': bp.types.NeuState(['V', 'input'], help='Pre-synaptic neuron state must have "V" and "inp" item.'),
    }

    @bp.integrate
    def int_s(s, t, TT):
        return alpha * TT * (1 - s) - beta * s

    def update(ST, _t_, pre):
        if pre['spike'] > 0.:
            ST['sp_t'] = _t_
        TT = ((_t_ - ST['sp_t']) < T_duration) * T
        s = np.clip(int_s(ST['s'], _t_, TT), 0., 1.)
        ST['s'] = s

    @bp.delayed
    def output(ST, post):
        post_val = - g_max * ST['s'] * (post['V'] - E)
        post['input'] += post_val

    return bp.SynType(name='AMPA',
                      ST= bp.types.SynState({'s': 0., 'sp_t': -1e7}),
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
        'pre': bp.types.NeuState(['spike'], help='Pre-synaptic neuron state must have "sp" item.'),
        'post': bp.types.NeuState(['V', 'input'], help='Pre-synaptic neuron state must have "V" and "inp" item.'),
        'pre2syn': bp.types.ListConn(help='Pre-synaptic neuron index -> synapse index'),
        'post2syn': bp.types.ListConn(help='Post-synaptic neuron index -> synapse index'),
    }

    # model logic
    # -----------

    def update(ST, _t_, pre, pre2syn):
        s = ints(ST['s'], _t_)
        spike_idx = np.where(pre['spike'] > 0.)[0]
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
        post['input'] -= post_cond * (post['V'] - E)

    return bp.SynType(name='AMPA1',
                      ST=bp.types.SynState(['s', 'g']),
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
        pre=bp.types.NeuState(['spike'], help='Pre-synaptic neuron state must have "sp" item.'),
        post=bp.types.NeuState(['V', 'input'], help='Pre-synaptic neuron state must have "V" and "inp" item.'),
        pre2syn=bp.types.ListConn(help='Pre-synaptic neuron index -> synapse index'),
        post2syn=bp.types.ListConn(help='Post-synaptic neuron index -> synapse index'),
    )

    def update(ST, _t_, pre, pre2syn):
        for i in np.where(pre['spike'] > 0.)[0]:
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
        post['input'] -= post_cond * (post['V'] - E)

    return bp.SynType(name='AMPA',
                      ST=bp.types.SynState({'s': 0., 'sp_t': -1e7, 'g': 0.}),
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
        'pre': bp.types.NeuState(['spike']),
        'post': bp.types.NeuState(['V', 'input']),
        'conn_mat': bp.types.MatConn(),
    }

    # model logic
    # -----------

    def update(ST, _t_, pre, conn_mat):
        s = ints(ST['s'], _t_)
        s += pre['spike'].reshape((-1, 1)) * conn_mat
        ST['s'] = s
        ST['g'] = g_max * s

    @bp.delayed
    def output(ST, post):
        post_cond = np.sum(ST['g'], axis=0)
        post['input'] -= post_cond * (post['V'] - E)

    return bp.SynType(name='AMPA1',
                      ST=bp.types.SynState(['s', 'g']),
                      requires=requires,
                      steps=(update, output),
                      mode='matrix')


def define_AMPA2_matrix(g_max=0.42, E=0., alpha=0.98, beta=0.18, T=0.5, T_duration=0.5):
    requires = dict(
        pre=bp.types.NeuState(['spike']),
        post=bp.types.NeuState(['V', 'input']),
        conn_mat=bp.types.MatConn(),
    )

    @bp.integrate
    def int_s(s, t, TT):
        return alpha * TT * (1 - s) - beta * s

    def update(ST, _t_, pre, conn_mat):
        
        spike_idxs = np.where(pre['spike'] > 0.)[0]
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
                      ST=bp.types.SynState({'s': 0., 'sp_t': -1e7, 'g': 0.}),
                      requires=requires,
                      steps=(update, output),
                      mode='matrix')


def run_ampa_single(define, duration=350.):
    LIF = define_LIF()
    pre = bp.NeuGroup(LIF, 2)
    post = bp.NeuGroup(LIF, 3)
    cls = define()
    ampa = bp.SynConn(model=cls,
                      pre_group=pre,
                      post_group=post,
                      conn=bp.connect.All2All(),
                      monitors=['s'],
                      delay=10.)
    ampa.set_schedule(['input', 'update', 'output', 'monitor'])

    net = bp.Network(pre, ampa, post)
    Iext = bp.inputs.spike_current([10, 110, 210, 310, 410], bp.profile._dt, 1., duration=duration)
    net.run(duration, inputs=(ampa, 'pre.spike', Iext, '='), report=True)

    fig, gs = bp.visualize.get_figure(1, 1, 5, 6)
    fig.add_subplot(gs[0, 0])
    bp.visualize.line_plot(net.ts, ampa.mon.s, legend='s', show=True)


if __name__ == '__main__':
    bp.profile.set(jit=False, dt=0.1)

    # run_ampa_single(define_AMPA1_scalar)
    # run_ampa_single(define_AMPA2_scalar)
    # run_ampa_single(define_AMPA1_vector)
    # run_ampa_single(define_AMPA2_vector)
    # run_ampa_single(define_AMPA1_matrix)
    run_ampa_single(define_AMPA2_matrix)
