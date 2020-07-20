# -*- coding: utf-8 -*-

import numpy as np

from npbrain.core import integrate
from npbrain.core.synapse import *
from npbrain.utils.helper import clip

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
    connection : dict, str, callable
        The connection method.
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
    num_pre = pre.num
    num_post = post.num
    var2index = {'u': (2, 0), 'x': (2, 1)}

    pre_ids, post_ids, anchors = connection
    num = len(pre_ids)

    # The first (num_syn, ) shape variable is "u"
    # The second (num_syn, ) shape variable is "x"
    state = initial_syn_state(delay, num_pre, num_post, num, num_syn_shape_var=2)
    state[2][0] = np.ones(num) * u0
    state[2][1] = np.ones(num) * x0

    @integrate
    def int_u(u, t):
        return - u / tau_f

    @integrate
    def int_x(x, t):
        return (1 - x) / tau_d

    def update_state(syn_state, t, var_index):
        # get synapse state
        u_old = syn_state[2][0]
        x_old = syn_state[2][1]
        pre_spike = syn_state[0][0]
        # calculate synaptic state
        spike_idx = np.where(pre_spike > 0.)[0]
        u_new = int_u(u_old, t)
        x_new = int_x(x_old, t)
        for i in spike_idx:
            idx = anchors[:, i]
            u_new[idx[0]: idx[1]] += U * (1 - u_old[idx[0]: idx[1]])
            x_new[idx[0]: idx[1]] -= u_new[idx[0]: idx[1]] * x_old[idx[0]: idx[1]]
        u_new = clip(u_new, 0., 1.)
        x_new = clip(x_new, 0., 1.)
        syn_state[2][0] = u_new
        syn_state[2][1] = x_new
        # get post-synaptic values
        g = np.zeros(num_post)
        for i in range(num_pre):
            idx = anchors[:, i]
            post_idx = post_ids[idx[0]: idx[1]]
            g[post_idx] += u_new[idx[0]: idx[1]] * x_new[idx[0]: idx[1]]
        record_conductance(syn_state, var_index, g)

    def output_synapse(syn_state, var_index, post_neu_state):
        output_idx = var_index[-2]
        syn_val = syn_state[output_idx[0]][output_idx[1]]
        post_neu_state[-1] += syn_val * weights

    return Synapses(**locals())
