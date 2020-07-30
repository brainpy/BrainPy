# -*- coding: utf-8 -*-

import numpy as np

from npbrain.core import integrate
from npbrain.core.synapse import *

__all__ = [
    'NMDA',
]


def NMDA(pre, post, connection, delay=None, g_max=0.15, E=0, alpha=0.062, beta=3.75,
         cc_Mg=1.2, tau_decay=100., a=0.5, tau_rise=2., name='NMDA'):
    """NMDA conductance-based synapse.

    .. math::

        I_{syn}&=\\bar{g}_{syn} s (V-E_{syn})

        g(t) &=\\bar{g} \\cdot (V-E_{syn}) \\cdot g_{\\infty}
        \\cdot \\sum_j s_j(t) \\quad (3)

        g_{\\infty}(V,[{Mg}^{2+}]_{o}) & =(1+{e}^{-\\alpha V}
        [{Mg}^{2+}]_{o}/\\beta)^{-1}  \\quad (4)

        \\frac{d s_{j}(t)}{dt} & =-\\frac{s_{j}(t)}
        {\\tau_{decay}}+a x_{j}(t)(1-s_{j}(t))  \\quad (5)

        \\frac{d x_{j}(t)}{dt} & =-\\frac{x_{j}(t)}{\\tau_{rise}}+
        \\sum_{k} \\delta(t-t_{j}^{k})  \\quad (6)

    where the decay time of NMDA currents is taken to be :math:`\\tau_{decay}` =100 ms,
    :math:`a= 0.5 ms^{-1}`, and :math:`\\tau_{rise}` =2 ms (Hestrin et al., 1990;
    Spruston et al., 1995).

    Parameters
    ----------
    pre : Neurons
        The pre-synaptic neuron group.
    post : Neurons
        The post-synaptic neuron group.
    connection : tuple
        The connectivity.
    delay : None, float
        The delay length.
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
    name : str
        The name of synapse.

    Returns
    -------
    synapse : Synapses
        The constructed AMPA synapses.
    """

    var2index = {'x': (0, 0), 's': (0, 1)}

    pre_indexes, post_indexes, pre_anchors = connection
    num = len(pre_indexes)
    num_pre, num_post = pre.num, post.num

    state = initial_syn_state(delay, num_post=num_post, num_syn=num, num_syn_var=2)

    @integrate(signature='f8[:](f8[:], f8)')
    def int_x(x, t):
        return -x / tau_rise

    @integrate(signature='f8[:](f8[:], f8, f8[:])')
    def int_s(s, t, x):
        return -s / tau_decay + a * x * (1 - s)

    def update_state(syn_state, t, delay_idx, pre_state, post_state):
        # get synapse state
        pre_spike = pre_state[-3]
        x = syn_state[0][0]
        s = syn_state[0][1]
        # calculate synaptic state
        spike_idx = np.where(pre_spike > 0.)[0]
        for i in spike_idx:
            idx = pre_anchors[:, i]
            x[idx[0]: idx[1]] += 1.
        x = int_x(x, t)
        s = int_s(s, t, x)
        syn_state[0][0] = x
        syn_state[0][1] = s
        # get post-synaptic values
        g = np.zeros(num_post)
        for i in range(num_pre):
            idx = pre_anchors[:, i]
            post_idx = post_indexes[idx[0]: idx[1]]
            g[post_idx] += s[idx[0]: idx[1]]
        syn_state[1][delay_idx] = g

    if hasattr(post, 'ref') and getattr(post, 'ref') > 0.:

        def output_synapse(syn_state, output_idx, pre_state, post_state):
            g_val = syn_state[1][output_idx]
            post_v = post_state[0]
            g = - g_max * g_val * (post_v - E)
            g_inf = 1 + cc_Mg / beta * np.exp(-alpha * post_v)
            post_state[-1] += g * g_inf * post_state[-5]

    else:

        def output_synapse(syn_state, output_idx, pre_state, post_state):
            g_val = syn_state[1][output_idx]
            post_v = post_state[0]
            g = - g_max * g_val * (post_v - E)
            g_inf = 1 + cc_Mg / beta * np.exp(-alpha * post_v)
            post_state[-1] += g * g_inf

    return Synapses(**locals())
