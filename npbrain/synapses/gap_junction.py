# -*- coding: utf-8 -*-

from .. import _numpy as np

from npbrain.core.synapse_group import *


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

    pre_ids, post_ids, anchors = connection

    var2index = dict()
    num, num_pre, num_post = len(pre_ids), pre.num, post.num
    state = None
    delay_state = init_delay_state(delay=delay, num_post=num_post)

    # weights
    if np.size(weights) == 1:
        weights = np.ones(num) * weights
    assert np.size(weights) == num, 'Unknown weights shape: {}'.format(weights.shape)

    def update_state(delay_st, delay_idx, pre_state, post_state):
        # get synapse state
        pre_v = pre_state[0]
        post_v = post_state[0]
        # get gap junction value
        g = np.zeros(num_post)
        for i_ in range(num_pre):
            idx = anchors[:, i_]
            post_idx = post_ids[idx[0]: idx[1]]
            v_diff = pre_v[i_] - post_v[post_idx]
            g[post_idx] += weights[idx[0]: idx[1]] * v_diff
        delay_st[delay_idx] = g

    if hasattr(post, 'ref') and getattr(post, 'ref') > 0.:

        def output_synapse(delay_st, output_idx, post_state):
            g_val = delay_st[output_idx]
            post_state[-1] += g_val * post_state[-5]
    else:

        def output_synapse(delay_st, output_idx, post_state):
            g_val = delay_st[output_idx]
            post_state[-1] += g_val

    return Synapses(**locals())


def GapJunction_LIF(pre, post, weights, connection, k_spikelet=0.1, delay=None, name='LIF_gap_junction'):
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
    k = k_spikelet * weights

    var2index = dict()
    pre_ids, post_ids, anchors = connection
    num, num_pre, num_post = len(pre_ids), pre.num, post.num
    state = None
    delay_state = init_delay_state(num_post=num_post * 2, delay=delay)

    # weights
    if np.size(weights) == 1:
        weights = np.ones(num) * weights
    assert np.size(weights) == num, 'Unknown weights shape: {}'.format(weights.shape)

    def update_state(delay_st, delay_idx, pre_state, post_state):
        # get synapse state
        pre_spike = pre_state[-3]
        pre_v = pre_state[0]
        post_v = post_state[0]
        # calculate synapse state
        spike_idx = np.where(pre_spike > 0.)[0]
        # get spikelet
        g1 = np.zeros(num_post)
        for i_ in spike_idx:
            idx = anchors[:, i_]
            post_idx = post_ids[idx[0]: idx[1]]
            g1[post_idx] += k
        # get gap junction value
        g2 = np.zeros(num_post)
        for i_ in range(num_pre):
            idx = anchors[:, i_]
            post_idx = post_ids[idx[0]: idx[1]]
            v_diff = pre_v[i_] - post_v[post_idx]
            g2[post_idx] += weights[idx[0]: idx[1]] * v_diff
        # record conductance
        g = np.zeros(num_post * 2)
        g[num_post:] = g1
        g[:num_post] = g2
        delay_st[delay_idx] = g

    if hasattr(post, 'ref') and getattr(post, 'ref') > 0.:

        def output_synapse(delay_st, output_idx, post_state):
            syn_val = delay_st[output_idx]
            # post-neuron inputs
            val_input = syn_val[:num_post]
            post_state[-1] += val_input * post_state[-5]
            # post-neuron potential
            val_potential = syn_val[num_post:]
            post_state[0] += val_potential * post_state[-5]

    else:

        def output_synapse(delay_st, output_idx, post_state):
            syn_val = delay_st[output_idx]
            # post-neuron inputs
            post_state[-1] += syn_val[:num_post]
            # post-neuron potential
            post_state[0] += syn_val[num_post:]

    return Synapses(**locals())
