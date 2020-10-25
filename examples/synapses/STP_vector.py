# -*- coding: utf-8 -*-

import brainpy as bp
import brainpy.numpy as np


def STP(U=0.15, tau_f=1500., tau_d=200.):
    """Short-term plasticity proposed by Tsodyks and Markram (Tsodyks 98) [1]_.

    The model is given by

    .. math::

        \\frac{du}{dt} = -\\frac{u}{\\tau_f}+U(1-u^-)\\delta(t-t_{sp})

        \\frac{dx}{dt} = \\frac{1-x}{\\tau_d}-u^+x^-\\delta(t-t_{sp})

    where :math:`t_{sp}` denotes the spike time and :math:`U` is the increment
    of :math:`u` produced by a spike.

    The synaptic current generated at the synapse by the spike arriving
    at :math:`t_{sp}` is then given by

    .. math::

        \\Delta I(t_{sp}) = Au^+x^-

    where :math:`A` denotes the response amplitude that would be produced
    by total release of all the neurotransmitter (:math:`u=x=1`), called
    absolute synaptic efficacy of the connections.

    Parameters
    ----------
    tau_d : float
        Time constant of short-term depression.
    tau_f : float
        Time constant of short-term facilitation .
    U : float
        The increment of :math:`u` produced by a spike.
    x0 : float
        Initial value of :math:`x`.
    u0 : float
        Initial value of :math:`u`.

    References
    ----------

    .. [1] Tsodyks, Misha, Klaus Pawelzik, and Henry Markram. "Neural networks
           with dynamic synapses." Neural computation 10.4 (1998): 821-835.
    """
    requires = dict(
        ST=bp.types.SynState({'u': 0., 'x': 1., 'w': 1., 'g': 0.}),
        pre=bp.types.NeuState(['sp']),
        post=bp.types.NeuState(['V', 'inp']),
        pre2syn=bp.types.ListConn(),
        post2syn=bp.types.ListConn(),
    )

    @bp.integrate(method='exponential')
    def int_u(u, t):
        return - u / tau_f

    @bp.integrate(method='exponential')
    def int_x(x, t):
        return (1 - x) / tau_d

    def update(ST, pre, pre2syn):
        u = int_u(ST['u'], 0)
        x = int_x(ST['x'], 0)
        for pre_id in np.where(pre['sp'] > 0.)[0]:
            syn_ids = pre2syn[pre_id]
            u_syn = u[syn_ids] + U * (1 - ST['u'][syn_ids])
            u[syn_ids] = u_syn
            x[syn_ids] -= u_syn * ST['x'][syn_ids]
        ST['u'] = np.clip(u, 0., 1.)
        ST['x'] = np.clip(x, 0., 1.)
        ST['g'] = ST['w'] * ST['u'] * ST['x']

    @bp.delayed
    def output(ST, post, post2syn):
        post_cond = np.zeros(len(post2syn), dtype=np.float_)
        for post_id, syn_ids in enumerate(post2syn):
            post_cond[post_id] = np.sum(ST['g'][syn_ids])
        post['inp'] += post_cond

    return bp.SynType(name='STP',
                      requires=requires,
                      steps=(update, output),
                      vector_based=True)
