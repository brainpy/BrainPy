# -*- coding: utf-8 -*-

import numpy as np

from npbrain.core import integrate
from npbrain.core.synapse import *
from npbrain.utils.helper import get_clip

__all__ = [
    'GABAa1',
    'GABAa2',
    'GABAb1',
    'GABAb2',
]


def GABAa1(pre, post, connection, g_max=0.4, E=-80., tau_decay=6., delay=None, name='GABAa_ChType1'):
    """GABAa conductance-based synapse (type 1).

    .. math::

        I_{syn}&=\\bar{g}_{syn} s (V-E_{syn})

        \\frac{ds}{dt}&=-\\frac{s}{\\tau_{decay}}+\\sum_{k} \\delta(t-t_{j}^{k})

    Parameters
    ----------
    pre : Neurons
        The pre-synaptic neuron group.
    post : Neurons
        The post-synaptic neuron group.
    connection : tuple
        The connectivity.
    g_max
    E
    tau_decay
    delay
    name

    Returns
    -------
    synapse : Synapses
        The constructed GABAa synapses.
    """
    var2index = {'s': (0, 0)}

    pre_indexes, post_indexes, pre_anchors = connection
    
    num = len(pre_indexes)
    num_pre, num_post = pre.num, post.num
    state = initial_syn_state(delay, num_post=num_post, num_syn=num, num_syn_var=1)

    @integrate(signature='f8[:](f8[:], f8)')
    def int_s(s, t):
        return - s / tau_decay

    def update_state(syn_state, t, delay_idx, pre_state, post_state):
        # get synaptic state
        s = syn_state[0][0]
        s = int_s(s, t)
        pre_spike = pre_state[-3]
        # calculate synaptic state
        spike_idx = np.where(pre_spike > 0.)[0]
        for i in spike_idx:
            idx = pre_anchors[:, i]
            s[idx[0]: idx[1]] += 1
        syn_state[0][0] = s
        # get post-synaptic values
        g = np.zeros(num_post)
        for i in range(num_pre):
            idx = pre_anchors[:, i]
            post_idx = post_indexes[idx[0]: idx[1]]
            g[post_idx] += s[idx[0]: idx[1]]
        syn_state[1][delay_idx] = g

    if hasattr(post, 'ref') and getattr(post, 'ref') > 0.:

        def output_synapse(syn_state, output_idx, pre_state, post_state):
            g_val = syn_state[1][output_idx]
            post_val = - g_max * g_val * (post_state[0] - E)
            post_state[-1] += post_val * post_state[-5]

    else:

        def output_synapse(syn_state, output_idx, pre_state, post_state):
            g_val = syn_state[1][output_idx]
            post_val = - g_max * g_val * (post_state[0] - E)
            post_state[-1] += post_val

    return Synapses(**locals())


def GABAa2(pre, post, connection, g_max=0.04, E=-80., alpha=0.53, beta=0.18,
           T=1., T_duration=1., delay=None, name='GABAa_ChType2'):
    """GABAa conductance-based synapse (type 2).

    .. math::

        I_{syn} &=\\bar{g}_{syn} s (V-E_{syn})

        \\frac{ds}{dt} &=\\alpha[T](1-s)-\\beta s

    Parameters
    ----------
    pre : Neurons
        The pre-synaptic neuron group.
    post : Neurons
        The post-synaptic neuron group.
    connection : tuple
        The connectivity.
    g_max
    E
    tau_decay
    alpha
    beta
    T : float
        transmitter concentration, mM
    T_duration : float
    delay
    name

    Returns
    -------
    synapse : Synapses
        The constructed GABAa synapses.
    """
    var2index = {'s': (0, 0), 'syn_spike_time': (0, 1)}

    pre_indexes, post_indexes, pre_anchors = connection
    num = len(pre_indexes)
    num_pre, num_post = pre.num, post.num
    state = initial_syn_state(delay, num_post=num_post, num_syn=num, num_syn_var=2)
    state[0][1] = -np.inf

    clip = get_clip()

    @integrate(signature='f8[:](f8[:], f8, f8[:])')
    def int_s(s, t, TT):
        return alpha * TT * (1 - s) - beta * s

    def update_state(syn_state, t, delay_idx, pre_state, post_state):
        # get synaptic state
        pre_spike = pre_state[-3]
        s = syn_state[0][0]
        last_spike = syn_state[0][1]
        # calculate synaptic state
        spike_idx = np.where(pre_spike > 0.)[0]
        for i in spike_idx:
            idx = pre_anchors[:, i]
            last_spike[idx[0]: idx[1]] = t
        TT = ((t - last_spike) < T_duration).astype(np.float64) * T
        s = int_s(s, t, TT)
        s = clip(s, 0., 1.)
        syn_state[0][0] = s
        syn_state[0][1] = last_spike
        # get post-synaptic values
        g = np.zeros(num_post)
        for i in range(num_pre):
            idx = pre_anchors[:, i]
            post_idx = post_indexes[idx[0]: idx[1]]
            g[post_idx] += s[idx[0]: idx[1]]
        syn_state[1][delay_idx] = g

    if hasattr(post, 'ref') and getattr(post, 'ref') > 0.:

        def output_synapse(syn_state, output_idx, pre_state, post_state):
            g_val = syn_state[1][output_idx]
            post_val = - g_max * g_val * (post_state[0] - E)
            post_state[-1] += post_val * post_state[-5]

    else:

        def output_synapse(syn_state, output_idx, pre_state, post_state):
            g_val = syn_state[1][output_idx]
            post_val = - g_max * g_val * (post_state[0] - E)
            post_state[-1] += post_val

    return Synapses(**locals())


def GABAb1(pre, post, connection, g_max=0.02, E=-95., k1=0.18, k2=0.034, k3=0.09,
           k4=0.0012, T=0.5, T_duration=0.3, delay=None, name='GABAb_ChType1'):
    """GABAb conductance-based synapse (type 1).

    .. math::

        &\\frac{d[R]}{dt} = k_3 [T](1-[R])- k_4 [R]

        &\\frac{d[G]}{dt} = k_1 [R]- k_2 [G]

        I_{GABA_{B}} &=\\overline{g}_{GABA_{B}} (\\frac{[G]^{4}} {[G]^{4}+100}) (V-E_{GABA_{B}})


    - [G] is the concentration of activated G protein.
    - [R] is the fraction of activated receptor.
    - [T] is the transmitter concentration.

    Parameters
    ----------
    pre : Neurons
        The pre-synaptic neuron group.
    post : Neurons
        The post-synaptic neuron group.
    connection : tuple
        The connectivity.
    g_max
    E
    k1
    k2
    k3
    k4
    T
    T_duration
    delay
    name

    Returns
    -------
    synapse : Synapses
        The constructed GABAb synapses.
    """
    var2index = {'R': (0, 0), 'G': (0, 1), 'syn_spike_time': (0, 2)}

    pre_indexes, post_indexes, pre_anchors = connection

    num = len(pre_indexes)
    num_pre, num_post = pre.num, post.num
    state = initial_syn_state(delay, num_post=num_post, num_syn=num, num_syn_var=3)
    state[0][2] = -np.inf

    clip = get_clip()

    @integrate(signature='f8[:](f8[:], f8, f8[:])')
    def int_R(R, t, TT):
        return k3 * TT * (1 - R) - k4 * R

    @integrate(signature='f8[:](f8[:], f8, f8[:])')
    def int_G(G, t, R):
        return k1 * R - k2 * G

    def update_state(syn_state, t, delay_idx, pre_state, post_state):
        # get synaptic state
        pre_spike = pre_state[-3]
        R = syn_state[0][0]
        G = syn_state[0][1]
        last_spike = syn_state[0][2]
        # calculate synaptic state
        spike_idx = np.where(pre_spike > 0.)[0]
        for i in spike_idx:
            idx = pre_anchors[:, i]
            last_spike[idx[0]: idx[1]] = t
        TT = ((t - last_spike) < T_duration).astype(np.float64) * T
        R = clip(int_R(R, t, TT), 0., 1.)
        G = int_G(G, t, R)
        syn_state[0][0] = R
        syn_state[0][1] = G
        syn_state[0][2] = last_spike
        # get post-synaptic values
        G = g_max * (G ** 4 / (G ** 4 + 100))
        g = np.zeros(num_post)
        for i in range(num_pre):
            idx = pre_anchors[:, i]
            post_idx = post_indexes[idx[0]: idx[1]]
            g[post_idx] += G[idx[0]: idx[1]]
        syn_state[1][delay_idx] = g

    if hasattr(post, 'ref') and getattr(post, 'ref') > 0.:

        def output_synapse(syn_state, output_idx, pre_state, post_state):
            g_val = syn_state[1][output_idx]
            post_val = - g_val * (post_state[0] - E)
            post_state[-1] += post_val * post_state[-5]

    else:

        def output_synapse(syn_state, output_idx, pre_state, post_state):
            g_val = syn_state[1][output_idx]
            post_val = - g_val * (post_state[0] - E)
            post_state[-1] += post_val

    return Synapses(**locals())


def GABAb2(pre, post, connection, g_max=0.02, E=-95., k1=0.66, k2=0.02, k3=0.0053, k4=0.017,
           k5=8.3e-5, k6=7.9e-3, T=0.5, T_duration=0.5, delay=None, name='GABAb_ChType2'):
    """GABAb conductance-based synapse (type 2).

    .. math::

        &\\frac{d[D]}{dt}=K_{4}[R]-K_{3}[D]

        &\\frac{d[R]}{dt}=K_{1}[T](1-[R]-[D])-K_{2}[R]+K_{3}[D]

        &\\frac{d[G]}{dt}=K_{5}[R]-K_{6}[G]

        I_{GABA_{B}}&=\\bar{g}_{GABA_{B}} \\frac{[G]^{n}}{[G]^{n}+K_{d}}(V-E_{GABA_{B}})

    where [R] and [D] are, respectively, the fraction of activated
    and desensitized receptor, [G] (in μM) the concentration of activated G-protein.

    Parameters
    ----------
    pre : Neurons
        The pre-synaptic neuron group.
    post : Neurons
        The post-synaptic neuron group.
    connection : tuple
        The connectivity.
    g_max
    E
    k1
    k2
    k3
    k4
    k5
    k6
    T
    T_duration
    delay
    name

    Returns
    -------
    synapse : Synapses
        The constructed GABAb synapses.
    """
    var2index = {'D': (0, 0), 'R': (0, 1), 'G': (0, 2), 'syn_spike_time': (0, 3)}

    pre_indexes, post_indexes, pre_anchors = connection

    num = len(pre_indexes)
    num_pre, num_post = pre.num, post.num
    state = initial_syn_state(delay, num_post=num_post, num_syn=num, num_syn_var=4)
    state[0][3] = -np.inf

    clip = get_clip()

    @integrate(signature='f8[:](f8[:], f8, f8[:])')
    def int_D(D, t, R):
        return k4 * R - k3 * D

    @integrate(signature='f8[:](f8[:], f8, f8[:], f8[:])')
    def int_R(R, t, TT, D):
        return k1 * TT * (1 - R - D) - k2 * R + k3 * D

    @integrate(signature='f8[:](f8[:], f8, f8[:])')
    def int_G(G, t, R):
        return k5 * R - k6 * G

    def update_state(syn_state, t, delay_idx, pre_state, post_state):
        # calculate synaptic state
        pre_spike = pre_state[-3]
        D = syn_state[0][0]
        R = syn_state[0][1]
        G = syn_state[0][2]
        last_spike = syn_state[0][3]
        spike_idx = np.where(pre_spike > 0.)[0]
        for i in spike_idx:
            idx = pre_anchors[:, i]
            last_spike[idx[0]: idx[1]] = t
        TT = ((t - last_spike) < T_duration).astype(np.float64) * T
        D = int_D(D, t, R)
        R = int_R(R, t, TT, D)
        G = int_G(G, t, R)
        syn_state[0][0] = D
        syn_state[0][1] = R
        syn_state[0][2] = G
        syn_state[0][3] = last_spike
        # get post-synaptic values
        g = np.zeros(num_post)
        for i in range(num_pre):
            idx = pre_anchors[:, i]
            post_idx = post_indexes[idx[0]: idx[1]]
            g[post_idx] += G[idx[0]: idx[1]]
        g = g_max * (g ** 4 / (g ** 4 + 100))
        syn_state[1][delay_idx] = g

    if hasattr(post, 'ref') and getattr(post, 'ref') > 0.:

        def output_synapse(syn_state, output_idx, pre_state, post_state):
            g_val = syn_state[1][output_idx]
            post_val = - g_val * (post_state[0] - E)
            post_state[-1] += post_val * post_state[-5]

    else:

        def output_synapse(syn_state, output_idx, pre_state, post_state):
            g_val = syn_state[1][output_idx]
            post_val = - g_val * (post_state[0] - E)
            post_state[-1] += post_val

    return Synapses(**locals())
