# -*- coding: utf-8 -*-

import numpy as np

from npbrain.core.synapse import *

__all__ = [
    'GapJunction',
    'GapJunction_LIF',
]


def GapJunction(pre, post, weights, connection, delay=None, name='gap_junction'):
    """Gap junction, or, electrical synapse.

    Parameters
    ----------
    pre : Neurons
        The pre-synaptic neuron group.
    post : Neurons
        The post-synaptic neuron group.
    weights : dict, np.ndarray, int, float
        The weighted coefficients of synapses.
    connection : tuple
        The connectivity.
    delay : None, float
        The delay length.
    name : str
        The name of synapse.

    Returns
    -------
    synapse : Synapses
        The constructed electrical synapses.
    """

    num_pre = pre.num
    num_post = post.num
    var2index = {'pre_V': (0, 0), 'post_V': (1, -1)}

    pre_ids, post_ids, anchors = connection
    num = len(pre_ids)

    # The first last (num_pre, ) shape variable is "pre-neuron spike"
    # The second last (num_pre, ) shape variable is "pre-neuron potential"
    # The first last (num_post, ) shape variable is "post-neuron potential"
    state = initial_syn_state(delay, num_pre, num_post, num,
                              num_pre_shape_var=1, num_post_shape_var=1)

    # weights
    if np.size(weights) == 1:
        weights = np.ones(num) * weights
    elif np.size(weights) == num:
        weights = weights
    else:
        raise ValueError('Unknown weights shape.')

    def update_state(syn_state, t, delay_idx):
        # get synapse state
        pre_v = syn_state[0][-2]
        post_v = syn_state[1][-1]
        # get gap junction value
        g = np.zeros(num_post)
        for i_ in range(num_pre):
            idx = anchors[:, i_]
            post_idx = post_ids[idx[0]: idx[1]]
            g[post_idx] += weights[idx[0]: idx[1]] * (pre_v[i_] - post_v[post_idx])
        syn_state[1][delay_idx] = g

    if hasattr(post, 'ref') and getattr(post, 'ref') > 0.:

        def output_synapse(syn_state, output_idx, post_neu_state):
            g_val = syn_state[1][output_idx]
            for idx in range(num_post):
                post_neu_state[-1, idx] += g_val[idx] * post_neu_state[-5, idx]
    else:

        def output_synapse(syn_state, output_idx, post_neu_state):
            g_val = syn_state[1][output_idx]
            # post-neuron inputs
            post_neu_state[-1] += g_val

    def collect_spike(syn_state, pre_neu_state, post_neu_state):
        # spike
        syn_state[0][-1] = pre_neu_state[-3]
        # membrane potential of pre-synaptic neuron group
        syn_state[0][-2] = pre_neu_state[0]
        # membrane potential of post-synaptic neuron group
        syn_state[1][-1] = post_neu_state[0]

    return Synapses(**locals())


def GapJunction_LIF(pre, post, weights, connection, k_spikelet=0.1, delay=None, name='gap_junction'):
    """Gap junction, or, electrical synapse for LIF neuron model.

    Parameters
    ----------
    pre : Neurons
        The pre-synaptic neuron group.
    post : Neurons
        The post-synaptic neuron group.
    weights : dict, np.ndarray, int, float
        The weighted coefficients of synapses.
    connection : tuple
        The connectivity.
    k_spikelet : float
        The spikelet factor.
    delay : None, float
        The delay length.
    name : str
        The name of synapse.

    Returns
    -------
    synapse : Synapses
        The constructed electrical synapses.
    """
    num_pre = pre.num
    num_post = post.num
    var2index = {'pre_V': (0, 0), 'post_V': (1, -1)}
    k = k_spikelet * weights

    pre_ids, post_ids, anchors = connection
    num = len(pre_ids)

    # The first last (num_pre, ) shape variable is "pre-neuron spike"
    # The second last (num_pre, ) shape variable is "pre-neuron potential"
    # The first last (num_post * 2, ) shape variable is "post-neuron potential"
    # Other (num_post * 2, ) shape variables are corresponding for delays
    state = initial_syn_state(delay, num_pre, num_post * 2, num,
                              num_pre_shape_var=1, num_post_shape_var=1)

    # weights
    if np.size(weights) == 1:
        weights = np.ones(num) * weights
    elif np.size(weights) == num:
        weights = weights
    else:
        raise ValueError('Unknown weights shape.')

    def update_state(syn_state, t, delay_idx):
        # get synapse state
        spike = syn_state[0][-1]
        pre_v = syn_state[0][-2]
        post_v = syn_state[1][-1]
        # calculate synapse state
        spike_idx = np.where(spike > 0.)[0]
        g = np.zeros(num_post * 2)
        # get spikelet
        g1 = np.zeros(num_post)
        for i_ in spike_idx:
            idx = anchors[:, i_]
            post_idx = post_ids[idx[0]: idx[1]]
            g1[post_idx] += k
        g[num_post:] = g1
        # get gap junction value
        g2 = np.zeros(num_post)
        for i_ in range(num_pre):
            idx = anchors[:, i_]
            post_idx = post_ids[idx[0]: idx[1]]
            g2[post_idx] += weights[idx[0]: idx[1]] * (pre_v[i_] - post_v[post_idx])
        g[:num_post] = g2
        syn_state[1][delay_idx] = g

    if hasattr(post, 'ref') and getattr(post, 'ref') > 0.:

        def output_synapse(syn_state, output_idx, post_neu_state):
            syn_val = syn_state[1][output_idx]
            for idx in range(num_post):
                # post-neuron inputs
                post_neu_state[-1, idx] += syn_val[:num_post] * post_neu_state[-5, idx]
                # post-neuron potential
                post_neu_state[0, idx] += syn_val[num_post:] * post_neu_state[-5, idx]

    else:

        def output_synapse(syn_state, output_idx, post_neu_state):
            syn_val = syn_state[1][output_idx]
            # post-neuron inputs
            post_neu_state[-1] += syn_val[:num_post]
            # post-neuron potential
            post_neu_state[0] += syn_val[num_post:]

    def collect_spike(syn_state, pre_neu_state, post_neu_state):
        # spike
        syn_state[0][-1] = pre_neu_state[-3]
        # membrane potential of pre-synaptic neuron group
        syn_state[0][-2] = pre_neu_state[0]
        # membrane potential of post-synaptic neuron group
        syn_state[1][-1, :num_post] = post_neu_state[0]

    return Synapses(**locals())
