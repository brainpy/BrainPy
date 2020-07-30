# -*- coding: utf-8 -*-

import numpy as np

from npbrain.core.synapse import *

__all__ = [
    'VoltageJumpSynapse',
]


def VoltageJumpSynapse(pre, post, weights, connection, delay=None, var='V', name='VoltageJumpSynapse'):
    """Voltage jump synapses.

    .. math::

        I_{syn} = \sum J \delta(t-t_j)

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
    delay : int, float, None
        The delay period (ms).
    name : str
        The name of the synapse.
    var : str
        The variable of the post-synapse going to connect.

    Returns
    -------
    synapse : Synapses
        The constructed ordinary synapses.
    """
    var2index = dict()

    pre_ids, post_ids, anchors = connection
    num = len(pre_ids)
    num_pre, num_post = pre.num, post.num
    state = initial_syn_state(delay, num_syn=num, num_post=num_post)

    try:
        post_varid = post.var2index[var]
    except KeyError:
        raise KeyError("Post synapse doesn't has variable '{}'.".format(var))

    # weights
    if np.size(weights) == 1:
        weights = np.ones(num) * weights
    assert np.size(weights) == num, 'Unknown weights shape: {}'.format(weights.shape)

    def update_state(syn_state, t, delay_idx, pre_state, post_state):
        # get synapse state
        pre_spike = pre_state[-3]
        spike_idx = np.where(pre_spike > 0)[0]
        # get post-synaptic values
        g = np.zeros(num_post)
        for i_ in spike_idx:
            idx = anchors[:, i_]
            post_idx = post_ids[idx[0]: idx[1]]
            g[post_idx] += weights[idx[0]: idx[1]]
        syn_state[1][delay_idx] = g

    if hasattr(post, 'ref') and getattr(post, 'ref') > 0.:

        def output_synapse(syn_state, output_idx, pre_state, post_state):
            g_val = syn_state[1][output_idx]
            for idx in range(num_post):
                val = g_val[idx] * post_state[-5, idx]
                post_state[post_varid, idx] += val
    else:

        def output_synapse(syn_state, output_idx, pre_state, post_state):
            g_val = syn_state[1][output_idx]
            post_state[post_varid] += g_val

    return Synapses(**locals())
