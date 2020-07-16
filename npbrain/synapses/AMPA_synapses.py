# -*- coding: utf-8 -*-

import numpy as np

from npbrain.core import integrate
from npbrain.core.synapse import *
from npbrain.utils.helper import clip

__all__ = [
    'AMPA1',
    'AMPA2',
]


def AMPA1(pre, post, connection, g_max=0.10, E=0., tau_decay=2.0, delay=None, name='AMPA_ChType1'):
    """AMPA conductance-based synapse (type 1).

    .. math::

        I_{syn}&=\\bar{g}_{syn} s (V-E_{syn})

        \\frac{d s}{d t}&=-\\frac{s}{\\tau_{decay}}+\\sum_{k} \\delta(t-t_{j}^{k})

    Parameters
    ----------
    pre : Neurons
        The pre-synaptic neuron group.
    post : Neurons
        The post-synaptic neuron group.
    connection : dict, str, callable
        The connection method.
    g_max
    E
    tau_decay
    delay
    name

    Returns
    -------
    synapse : Synapses
        The constructed AMPA synapses.
    """
    num_pre = pre.num
    num_post = post.num
    var2index = {'s': (2, 0)}

    pre_ids, post_ids, anchors = connection
    num = len(pre_ids)
    state = initial_syn_state(delay, num_pre, num_post, num, num_syn_shape_var=1)

    @integrate
    def int_f(s, t):
        return - s / tau_decay

    def update_state(syn_state, t, var_index):
        # get synaptic state
        spike_idx = np.where(syn_state[0][0] > 0.)[0]
        # calculate synaptic state
        s = int_f(syn_state[2][0], t)
        for i in spike_idx:
            idx = anchors[:, i]
            s[idx[0]: idx[1]] += 1
        syn_state[2][0] = s
        # get post-synaptic values
        g = np.zeros(num_post)
        for i in range(num_pre):
            idx = anchors[:, i]
            post_idx = post_ids[idx[0]: idx[1]]
            g[post_idx] += s[idx[0]: idx[1]]
        record_conductance(syn_state, var_index, g)

    def output_synapse(syn_state, var_index, post_neu_state):
        output_idx = var_index[-2]
        g_val = syn_state[output_idx[0]][output_idx[1]]
        post_val = - g_max * g_val * (post_neu_state[0] - E)
        post_neu_state[-1] += post_val

    return Synapses(**locals())


def AMPA2(pre, post, connection, g_max=0.42, E=0., alpha=0.98, beta=0.18,
          T=0.5, T_duration=0.5, delay=None, name='AMPA_ChType2'):
    """AMPA conductance-based synapse (type 2).

    .. math::

        I_{syn}&=\\bar{g}_{syn} s (V-E_{syn})

        \\frac{ds}{dt} &=\\alpha[T](1-s)-\\beta s

    Parameters
    ----------
    pre : Neurons
        The pre-synaptic neuron group.
    post : Neurons
        The post-synaptic neuron group.
    connection : dict, str, callable
        The connection method.
    g_max
    E
    alpha
    beta
    T
    T_duration
    delay
    name

    Returns
    -------
    synapse : Synapses
        The constructed AMPA synapses.
    """
    num_pre = pre.num
    num_post = post.num
    var2index = {'s': (2, 0), 'pre_spike_time': (2, 1)}

    pre_ids, post_ids, anchors = connection
    num = len(pre_ids)

    # The first (num_syn, ) variable is ``s``
    # The second (num_syn, ) variable if the last_spike time
    state = initial_syn_state(delay, num_pre, num_post, num, num_syn_shape_var=2)
    state[2][1] = -np.inf

    @integrate
    def int_s(s, t, TT):
        return alpha * TT * (1 - s) - beta * s

    def update_state(syn_state, t, var_index):
        # get synaptic state
        spike = syn_state[0][0]
        s = syn_state[2][0]
        last_spike = syn_state[2][1]
        # calculate synaptic state
        spike_idx = np.where(spike > 0.)[0]
        for i in spike_idx:
            idx = anchors[:, i]
            last_spike[idx[0]: idx[1]] = t
        TT = ((t - last_spike) < T_duration).astype(np.float64) * T
        s = clip(int_s(s, t, TT), 0., 1.)
        syn_state[2][0] = s
        syn_state[2][1] = last_spike
        # get post-synaptic values
        g = np.zeros(num_post)
        for i in range(num_pre):
            idx = anchors[:, i]
            post_idx = post_ids[idx[0]: idx[1]]
            g[post_idx] += s[idx[0]: idx[1]]
        record_conductance(syn_state, var_index, g)

    def output_synapse(syn_state, var_index, post_neu_state):
        output_idx = var_index[-2]
        g_val = syn_state[output_idx[0]][output_idx[1]]
        post_val = - g_max * g_val * (post_neu_state[0] - E)
        post_neu_state[-1] += post_val

    return Synapses(**locals())
