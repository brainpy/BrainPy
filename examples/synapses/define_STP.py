# -*- coding: utf-8 -*-

import numpy as np
import npbrain as nn


def STP(pre, post, weights, connection, U=0.15, tau_f=1500.,
        tau_d=200., u0=0.0, x0=1.0, delay=None, ):
    num_pre, num_post = pre.num, post.num
    var2index = {'u': (2, 0), 'x': (2, 1)}

    pre_ids, post_ids, anchors = connection
    num = len(pre_ids)

    state = nn.initial_syn_state(delay, num_pre, num_post, num, num_syn_shape_var=2)
    state[2][0] = np.ones(num) * u0
    state[2][1] = np.ones(num) * x0

    @nn.integrate
    def int_u(u, t):
        return - u / tau_f

    @nn.integrate
    def int_x(x, t):
        return (1 - x) / tau_d

    @nn.syn_delay
    def update_state(syn_state, t):
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
        u_new = nn.clip(u_new, 0., 1.)
        x_new = nn.clip(x_new, 0., 1.)
        syn_state[2][0] = u_new
        syn_state[2][1] = x_new
        # get post-synaptic values
        g = np.zeros(num_post)
        for i in range(num_pre):
            idx = anchors[:, i]
            post_idx = post_ids[idx[0]: idx[1]]
            g[post_idx] += u_new[idx[0]: idx[1]] * x_new[idx[0]: idx[1]]
        return g

    def output_synapse(syn_state, var_index, post_neu_state):
        output_idx = var_index[-2]
        syn_val = syn_state[output_idx[0]][output_idx[1]]
        post_neu_state[-1] += syn_val * weights

    return nn.Synapses(**locals())

