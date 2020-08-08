# -*- coding: utf-8 -*-

import numpy as np
import npbrain as nn


def GapJunction(pre, post, weights, connection, delay=None):
    var2index = dict()

    pre_ids, post_ids, anchors = connection
    num = len(pre_ids)
    num_pre = pre.num
    num_post = post.num

    state = nn.init_syn_state(delay, num_pre, num_post, num)

    def update_state(syn_state, t, delay_idx, pre_state, post_state):
        # get synapse state
        pre_v = pre_state[0]
        post_v = post_state[0]
        # get gap junction value
        g = np.zeros(num_post)
        for i_ in range(num_pre):
            idx = anchors[:, i_]
            post_idx = post_ids[idx[0]: idx[1]]
            g[post_idx] += weights * (pre_v[i_] - post_v[post_idx])
        syn_state[1][delay_idx] = g

    def output_synapse(syn_state, output_idx, pre_state, post_state):
        g_val = syn_state[1][output_idx]
        for idx in range(num_post):
            post_state[-1, idx] += g_val[idx] * post_state[-5, idx]

    return nn.Synapses(**locals())


def GJ_LIF(pre, post, weights, connection, k_spikelet=0.1, delay=None):
    num_pre = pre.num
    num_post = post.num
    var2index = dict()
    k = k_spikelet * weights

    pre_ids, post_ids, anchors = connection
    num = len(pre_ids)

    state = nn.init_syn_state(delay,
                              num_pre=num_pre, num_post=num_post * 2, num_syn=num)

    def update_state(syn_state, t, delay_idx, pre_state, post_state):
        # get synapse state
        pre_spike = pre_state[-3]
        pre_v = pre_state[0]
        post_v = post_state[0]

        # get spikelet value
        g1 = np.zeros(num_post)
        spike_idx = np.where(pre_spike > 0.)[0]
        for i_ in spike_idx:
            se = anchors[:, i_]
            post_idx = post_ids[se[0]: se[1]]
            g1[post_idx] += k

        # get gap junction value
        g2 = np.zeros(num_post)
        for i_ in range(num_pre):
            se = anchors[:, i_]
            post_idx = post_ids[se[0]: se[1]]
            g2[post_idx] += weights * (pre_v[i_] - post_v[post_idx])

        # record conductance
        g = np.zeros(num_post * 2)
        g[num_post:] = g1
        g[:num_post] = g2
        syn_state[1][delay_idx] = g

    def output_synapse(syn_state, output_idx, pre_state, post_state):
        syn_val = syn_state[1][output_idx]
        val_input = syn_val[:num_post]
        val_potential = syn_val[num_post:]
        post_state[-1] += val_input * post_state[-5]  # add to Isyn
        post_state[0] += val_potential * post_state[-5]  # add to V

    return nn.Synapses(**locals())



