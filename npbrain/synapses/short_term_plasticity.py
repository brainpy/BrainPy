# -*- coding: utf-8 -*-

from .. import _numpy as np

from npbrain.core import integrate
from npbrain.core.synapse import *
from npbrain.utils.helper import get_clip

__all__ = [
    'STP',
]


def STP(pre, post, weights, connection, U=0.15, tau_f=1500., tau_d=200.,
        u0=0.0, x0=1.0, delay=None, name='short_term_plasticity'):
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
    pre : Neurons
        The pre-synaptic neuron group.
    post : Neurons
        The post-synaptic neuron group.
    weights : dict, np.ndarray, int, float
        The weighted coefficients of synapses.
    connection : tuple
        The connection.
    delay : None, float
        The delay time length.
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
    name : str
        The name of synapse.

    Returns
    -------
    synapse : Synapses
        The constructed electrical synapses.

    References
    ----------

    .. [1] Tsodyks, Misha, Klaus Pawelzik, and Henry Markram. "Neural networks
           with dynamic synapses." Neural computation 10.4 (1998): 821-835.
    """
    var2index = {'u': 0, 'x': 1}

    pre_ids, post_ids, anchors = connection
    num, num_pre, num_post = len(pre_ids), pre.num, post.num
    state = init_syn_state(num_syn=num, variables=[('u', u0), ('x', x0)])
    delay_state = init_delay_state(num_post=num_post, delay=delay)

    clip = get_clip()

    # weights
    if np.size(weights) == 1:
        weights = np.ones(num) * weights
    assert np.size(weights) == num, 'Unknown weights shape: {}'.format(weights.shape)

    @integrate(signature='{f}[:]({f}[:], {f})')
    def int_u(u, t):
        return - u / tau_f

    @integrate(signature='{f}[:]({f}[:], {f})')
    def int_x(x, t):
        return (1 - x) / tau_d

    def update_state(syn_st, t, delay_st, delay_idx, pre_state):
        # get synapse state
        u_old = syn_st[0]
        x_old = syn_st[1]
        pre_spike = pre_state[-3]
        # calculate synaptic state
        u_new = int_u(u_old, t)
        x_new = int_x(x_old, t)
        for i in range(num_pre):
            if pre_spike[i] > 0.:
                se = anchors[:, i]
                u_new[se[0]: se[1]] += U * (1 - u_old[se[0]: se[1]])
                x_new[se[0]: se[1]] -= u_new[se[0]: se[1]] * x_old[se[0]: se[1]]
        u_new = clip(u_new, 0., 1.)
        x_new = clip(x_new, 0., 1.)
        syn_st[0] = u_new
        syn_st[1] = x_new
        # get post-synaptic values
        g = np.zeros(num_post)
        for i in range(num_pre):
            se = anchors[:, i]
            weight = weights[se[0]: se[1]]
            u = u_new[se[0]: se[1]]
            x = x_new[se[0]: se[1]]
            post_idx = post_ids[se[0]: se[1]]
            g[post_idx] += u * x * weight
        delay_st[delay_idx] = g

    if hasattr(post, 'ref') and getattr(post, 'ref') > 0.:

        def output_synapse(delay_st, output_idx, post_state):
            syn_val = delay_st[output_idx]
            for idx in range(num_post):
                val = syn_val[idx] * post_state[-5, idx]
                post_state[-1, idx] += val

    else:

        def output_synapse(delay_st, output_idx, post_state):
            syn_val = delay_st[output_idx]
            post_state[-1] += syn_val

    return Synapses(**locals())
