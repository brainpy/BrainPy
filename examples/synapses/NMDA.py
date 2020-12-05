# -*- coding: utf-8 -*-

import brainpy as bp
import brainpy.numpy as np


def get_NMDA_scalar(g_max=0.15, E=0, alpha=0.062, beta=3.75, cc_Mg=1.2, tau_decay=100., a=0.5, tau_rise=2.):
    """NMDA conductance-based synapse.

    .. math::

        I_{syn}&=\\bar{g}_{syn} s (V-E_{syn})

        g(_t_) &=\\bar{g} \\cdot (V-E_{syn}) \\cdot g_{\\infty}
        \\cdot \\sum_j s_j(_t_) \\quad (3)

        g_{\\infty}(V,[{Mg}^{2+}]_{o}) & =(1+{e}^{-\\alpha V}
        [{Mg}^{2+}]_{o}/\\beta)^{-1}  \\quad (4)

        \\frac{d s_{j}(_t_)}{dt} & =-\\frac{s_{j}(_t_)}
        {\\tau_{decay}}+a x_{j}(_t_)(1-s_{j}(_t_))  \\quad (5)

        \\frac{d x_{j}(_t_)}{dt} & =-\\frac{x_{j}(_t_)}{\\tau_{rise}}+
        \\sum_{k} \\delta(_t_-t_{j}^{k})  \\quad (6)

    where the decay time of NMDA currents is taken to be :math:`\\tau_{decay}` =100 ms,
    :math:`a= 0.5 ms^{-1}`, and :math:`\\tau_{rise}` =2 ms (Hestrin et al., 1990;
    Spruston et al., 1995).

    Parameters
    ----------
    g_max : float
        The maximum conductance.
    E : float
        The reversal potential.
    alpha : float
    beta : float
    cc_Mg : float
    tau_decay : float
        The time constant of decay.
    tau_rise : float
        The time constant of rise.
    a : float
    """
    requires = dict(
        ST=bp.types.SynState(['x', 's']),
        pre=bp.types.NeuState(['sp']),
        post=bp.types.NeuState(['V', 'inp']),
        pre2syn=bp.types.ListConn(),
        post2syn=bp.types.ListConn(),
    )

    @bp.integrate
    def int_x(x, t):
        return -x / tau_rise

    @bp.integrate
    def int_s(s, t, x):
        return -s / tau_decay + a * x * (1 - s)

    def update(ST, _t_, pre):
        x = int_x(ST['x'], _t_)
        x += pre['sp']
        s = int_s(ST['s'], _t_, x)
        ST['x'] = x
        ST['s'] = s

    @bp.delayed
    def output(ST, post):
        g = g_max * ST['s'] * (post['V'] - E)
        g_inf = 1 + cc_Mg / beta * np.exp(-alpha * post['V'])
        post['inp'] -= g * g_inf

    return bp.SynType(name='NMDA',
                      requires=requires,
                      steps=(update, output),
                      mode='scalar')


def get_NMDA_vector(g_max=0.15, E=0, alpha=0.062, beta=3.75, cc_Mg=1.2, tau_decay=100., a=0.5, tau_rise=2.):
    """NMDA conductance-based synapse.

    .. math::

        I_{syn}&=\\bar{g}_{syn} s (V-E_{syn})

        g(_t_) &=\\bar{g} \\cdot (V-E_{syn}) \\cdot g_{\\infty}
        \\cdot \\sum_j s_j(_t_) \\quad (3)

        g_{\\infty}(V,[{Mg}^{2+}]_{o}) & =(1+{e}^{-\\alpha V}
        [{Mg}^{2+}]_{o}/\\beta)^{-1}  \\quad (4)

        \\frac{d s_{j}(_t_)}{dt} & =-\\frac{s_{j}(_t_)}
        {\\tau_{decay}}+a x_{j}(_t_)(1-s_{j}(_t_))  \\quad (5)

        \\frac{d x_{j}(_t_)}{dt} & =-\\frac{x_{j}(_t_)}{\\tau_{rise}}+
        \\sum_{k} \\delta(_t_-t_{j}^{k})  \\quad (6)

    where the decay time of NMDA currents is taken to be :math:`\\tau_{decay}` =100 ms,
    :math:`a= 0.5 ms^{-1}`, and :math:`\\tau_{rise}` =2 ms (Hestrin et al., 1990;
    Spruston et al., 1995).

    Parameters
    ----------
    g_max : float
        The maximum conductance.
    E : float
        The reversal potential.
    alpha : float
    beta : float
    cc_Mg : float
    tau_decay : float
        The time constant of decay.
    tau_rise : float
        The time constant of rise.
    a : float
    """
    requires = dict(
        ST=bp.types.SynState(['x', 's', 'g']),
        pre=bp.types.NeuState(['sp']),
        post=bp.types.NeuState(['V', 'inp']),
        pre2syn=bp.types.ListConn(),
        post2syn=bp.types.ListConn(),
    )

    @bp.integrate
    def int_x(x, t):
        return -x / tau_rise

    @bp.integrate
    def int_s(s, t, x):
        return -s / tau_decay + a * x * (1 - s)

    def update(ST, _t_, pre, pre2syn):
        for pre_id in np.where(pre['sp'] > 0.)[0]:
            syn_ids = pre2syn[pre_id]
            ST['x'][syn_ids] += 1.
        x = int_x(ST['x'], _t_)
        s = int_s(ST['s'], _t_, x)
        ST['x'] = x
        ST['s'] = s
        ST['g'] = g_max * s

    @bp.delayed
    def output(ST, post, post2syn):
        post_cond = np.zeros(len(post2syn), dtype=np.float_)
        for post_id, syn_ids in enumerate(post2syn):
            post_cond[post_id] = np.sum(ST['g'][syn_ids])
        g = post_cond * (post['V'] - E)
        g_inf = 1 + cc_Mg / beta * np.exp(-alpha * post['V'])
        post['inp'] -= g * g_inf

    return bp.SynType(name='NMDA',
                      requires=requires,
                      steps=(update, output),
                      mode='vector')
