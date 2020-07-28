# -*- coding: utf-8 -*-

import numpy as np

from npbrain.core.synapse import *

__all__ = [
    'VoltageJumpSynapse',
]


def VoltageJumpSynapse(pre, post, weights, connection, delay=None, name='VoltageJumpSynapse'):
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
    name : str
        The name of the synapse.

    Returns
    -------
    synapse : Synapses
        The constructed ordinary synapses.
    """
    num_pre = pre.num
    num_post = post.num
    var2index = dict()

    pre_ids, post_ids, anchors = connection
    num = len(pre_ids)
    state = initial_syn_state(delay, num_pre, num_post, num)

    def update_state(syn_state, t, delay_idx):
        # get synapse state
        spike = syn_state[0][-1]
        spike_idx = np.where(spike > 0)[0]
        # get post-synaptic values
        g = np.zeros(num_post)
        for i_ in spike_idx:
            idx = anchors[:, i_]
            post_idx = post_ids[idx[0]: idx[1]]
            g[post_idx] += weights
        syn_state[1][delay_idx] = g

    if hasattr(post, 'ref') and getattr(post, 'ref') > 0.:
        def output_synapse(syn_state, output_idx, post_neu_state):
            g_val = syn_state[1][output_idx]
            for idx in range(num_post):
                post_neu_state[0, idx] += g_val[idx] * post_neu_state[-5, idx]
    else:
        def output_synapse(syn_state, output_idx, post_neu_state):
            g_val = syn_state[1][output_idx]
            post_neu_state[0] += g_val

    return Synapses(**locals())
