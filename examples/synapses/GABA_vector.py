# -*- coding: utf-8 -*-

import brainpy as bp
import brainpy.numpy as np


def GABAa1(g_max=0.4, reversal_potential=-80., tau_decay=6.):
    requires = dict(
        ST=bp.types.SynState(['s', 'g']),
        pre=bp.types.NeuState(['sp']),
        pre2syn=bp.types.ListConn(),
    )

    @bp.integrate
    def int_s(s, t):
        return - s / tau_decay

    def update(ST, pre, pre2syn):
        s = int_s(ST['s'], 0.)
        for pre_id in np.where(pre['sp'] > 0.)[0]:
            syn_ids = pre2syn[pre_id]
            s[syn_ids] += 1
        ST['s'] = s
        ST['g'] = g_max * s

    @bp.delayed
    def output(ST, post, post2syn):
        post_cond = np.zeros(len(post2syn), dtype=np.float_)
        for post_id, syn_ids in enumerate(post2syn):
            post_cond[post_id] = np.sum(ST['g'][syn_ids])
        post['inp'] -= post_cond * (post['V'] - reversal_potential)

    return bp.SynType(name='GABAa1',
                      requires=requires,
                      steps=(update, output),
                      vector_based=True)


def GABAa2(g_max=0.04, E=-80., alpha=0.53, beta=0.18, T=1., T_duration=1.):
    requires = dict(
        ST=bp.types.SynState({'s': 0., 'sp_t': -1e7, 'g': 0.}),
        pre=bp.types.NeuState(['sp']),
        pre2syn=bp.types.ListConn(),
        post2syn=bp.types.ListConn(),
    )

    @bp.integrate
    def int_s(s, t, TT):
        return alpha * TT * (1 - s) - beta * s

    def update(ST, pre, pre2syn, _t_):
        for pre_id in np.where(pre['sp'] > 0.)[0]:
            syn_ids = pre2syn[pre_id]
            ST['sp_t'][syn_ids] = _t_
        TT = ((_t_ - ST['sp_t']) < T_duration) * T
        s = int_s(ST['s'], _t_, TT)
        ST['s'] = s
        ST['g'] = g_max * s

    @bp.delayed
    def output(ST, post, post2syn):
        post_cond = np.zeros(len(post2syn), dtype=np.float_)
        for post_id, syn_ids in enumerate(post2syn):
            post_cond[post_id] = np.sum(ST['g'][syn_ids])
        post['inp'] -= post_cond * (post['V'] - E)

    return bp.SynType(name='GABAa2',
                      requires=requires,
                      steps=(update, output),
                      vector_based=True)


def GABAb1(g_max=0.02, E=-95., k1=0.18, k2=0.034, k3=0.09, k4=0.0012, T=0.5, T_duration=0.3):
    """GABAb conductance-based synapse (type 1).

    .. math::

        &\\frac{d[R]}{dt} = k_3 [T](1-[R])- k_4 [R]

        &\\frac{d[G]}{dt} = k_1 [R]- k_2 [G]

        I_{GABA_{B}} &=\\overline{g}_{GABA_{B}} (\\frac{[G]^{4}} {[G]^{4}+100}) (V-E_{GABA_{B}})


    - [G] is the concentration of activated G protein.
    - [R] is the fraction of activated receptor.
    - [T] is the transmitter concentration.

    Parameters
    ----------
    g_max : float
    E : float
    k1 : float
    k2 : float
    k3 : float
    k4 : float
    T : float
    T_duration : float
    """

    requires = dict(
        ST=bp.types.SynState({'R': 0., 'G': 0., 'sp_t': -1e7, 'g': 0.}),
        pre=bp.types.NeuState(['sp']),
        post=bp.types.NeuState(['V', 'inp']),
        pre2syn=bp.types.ListConn(),
        post2syn=bp.types.ListConn(),
    )

    @bp.integrate
    def int_R(R, t, TT):
        return k3 * TT * (1 - R) - k4 * R

    @bp.integrate
    def int_G(G, t, R):
        return k1 * R - k2 * G

    def update(ST, _t_, pre, pre2syn):
        for pre_id in np.where(pre['sp'] > 0.)[0]:
            syn_ids = pre2syn[pre_id]
            ST['sp_t'][syn_ids] = _t_
        TT = ((_t_ - ST['sp_t']) < T_duration) * T
        R = int_R(ST['R'], _t_, TT)
        G = int_G(ST['G'], _t_, R)
        ST['R'] = R
        ST['G'] = G
        ST['g'] = g_max * G ** 4 / (G ** 4 + 100)

    @bp.delayed
    def output(ST, post, post2syn):
        post_cond = np.zeros(len(post2syn), dtype=np.float_)
        for post_id, syn_ids in enumerate(post2syn):
            post_cond[post_id] = np.sum(ST['g'][syn_ids])
        post['inp'] -= post_cond * (post['V'] - E)

    return bp.SynType(name='GABAb1',
                      requires=requires,
                      steps=(update, output),
                      vector_based=True)


def GABAb2(g_max=0.02, E=-95., k1=0.66, k2=0.02, k3=0.0053, k4=0.017,
           k5=8.3e-5, k6=7.9e-3, T=0.5, T_duration=0.5):
    """GABAb conductance-based synapse (type 2).

    .. math::

        &\\frac{d[D]}{dt}=K_{4}[R]-K_{3}[D]

        &\\frac{d[R]}{dt}=K_{1}[T](1-[R]-[D])-K_{2}[R]+K_{3}[D]

        &\\frac{d[G]}{dt}=K_{5}[R]-K_{6}[G]

        I_{GABA_{B}}&=\\bar{g}_{GABA_{B}} \\frac{[G]^{n}}{[G]^{n}+K_{d}}(V-E_{GABA_{B}})

    where [R] and [D] are, respectively, the fraction of activated
    and desensitized receptor, [G] (in μM) the concentration of activated G-protein.

    Parameters
    ----------
    g_max
    E
    k1
    k2
    k3
    k4
    k5
    k6
    T
    T_duration
    """
    requires = dict(
        ST=bp.types.SynState({'D': 0., 'R': 0., 'G': 0., 'sp_t': -1e7, 'g': 0.}),
        pre=bp.types.NeuState(['sp']),
        post=bp.types.NeuState(['V', 'inp']),
        pre2syn=bp.types.ListConn(),
        post2syn=bp.types.ListConn(),
    )

    @bp.integrate
    def int_D(D, t, R):
        return k4 * R - k3 * D

    @bp.integrate
    def int_R(R, t, TT, D):
        return k1 * TT * (1 - R - D) - k2 * R + k3 * D

    @bp.integrate
    def int_G(G, t, R):
        return k5 * R - k6 * G

    def update(ST, _t_, pre, pre2syn):
        # calculate synaptic state
        for pre_id in np.where(pre['sp'] > 0.)[0]:
            syn_ids = pre2syn[pre_id]
            ST['sp_t'][syn_ids] = _t_
        TT = ((_t_ - ST['sp_t']) < T_duration) * T
        D = int_D(ST['D'], _t_, ST['R'])
        R = int_R(ST['R'], _t_, TT, D)
        G = int_G(ST['G'], _t_, R)
        ST['D'] = D
        ST['R'] = R
        ST['G'] = G
        ST['g'] = g_max * (G ** 4 / (G ** 4 + 100))

    @bp.delayed
    def output(ST, post, post2syn):
        post_cond = np.zeros(len(post2syn), dtype=np.float_)
        for post_id, syn_ids in enumerate(post2syn):
            post_cond[post_id] = np.sum(ST['g'][syn_ids])
        post['inp'] -= post_cond * (post['V'] - E)

    return bp.SynType(name='GABAb2',
                      requires=requires,
                      steps=(update, output),
                      vector_based=True)
