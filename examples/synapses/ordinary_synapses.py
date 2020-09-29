# -*- coding: utf-8 -*-

from npbrain import _numpy as np

from npbrain.core.synapse_connection import *

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
    weights : dict, bnp.ndarray, int, float
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
    num, num_pre, num_post = len(pre_ids), pre.num, post.num
    delay_state = init_delay_state(num_post=num_post, delay=delay)

    try:
        post_varid = post.var2index[var]
    except KeyError:
        raise KeyError("Post synapse doesn't has variable '{}'.".format(var))

    # weights
    if np.size(weights) == 1:
        weights = np.ones(num) * weights
    assert np.size(weights) == num, 'Unknown weights shape: {}'.format(weights.shape)

    def update_state(delay_st, delay_idx, pre_state):
        # get synapse state
        pre_spike = pre_state[-3]
        spike_idx = np.where(pre_spike > 0)[0]
        # get post-synaptic values
        g = np.zeros(num_post)
        for i_ in spike_idx:
            idx = anchors[:, i_]
            post_idx = post_ids[idx[0]: idx[1]]
            g[post_idx] += weights[idx[0]: idx[1]]
        delay_st[delay_idx] = g

    if hasattr(post, 'ref') and getattr(post, 'ref') > 0.:

        def output_synapse(delay_st, out_idx, post_state):
            g_val = delay_st[out_idx]
            for idx in range(num_post):
                val = g_val[idx] * post_state[-5, idx]
                post_state[post_varid, idx] += val
    else:

        def output_synapse(delay_st, out_idx, post_state):
            g_val = delay_st[1][out_idx]
            post_state[post_varid] += g_val

    return Synapses(**locals())
