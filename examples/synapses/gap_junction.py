# -*- coding: utf-8 -*-

import npbrain as nb
import npbrain.numpy as np


def gap_junction_vector():
    requires = dict(
        ST=nb.types.SynState(
            ['s', 'w'],
            help='''
            Gap junction state.

            s : conductance for post-synaptic neuron.
            w : gap junction conductance.
            '''
        ),
        pre=nb.types.NeuState(['V']),
        post=nb.types.NeuState(['V', 'inp']),
        post2syn=nb.types.ListConn(help='post-to-synapse connection.'),
        pre_ids=nb.types.Array(dim=1, help='Pre-synaptic neuron indices.'),
    )

    def update(ST, pre, post, post2syn, pre_ids):
        num_post = len(post2syn)
        post_cond = np.zeros(num_post, np.float_)
        for post_id in range(num_post):
            for syn_id in post2syn[post_id]:
                pre_id = pre_ids[syn_id]
                post_cond[post_id] = ST['w'][syn_id] * (pre['V'][pre_id] - post['V'][post_id])
        post['inp'] += post_cond

    return dict(requires=requires, steps=update)


GapJunction_vector = nb.SynType('GapJunction', create_func=gap_junction_vector, vector_based=True)


def gap_junction_single():
    requires = dict(
        ST=nb.types.SynState(
            ['w'],
            help='''
            Gap junction state.
            
            w : gap junction conductance.
            '''
        ),
        pre=nb.types.NeuState(['V']),
        post=nb.types.NeuState(['V', 'inp']),
    )

    def update(ST, pre, post):
        post['inp'] += ST['w'] * (pre['V'] - post['V'])

    return dict(requires=requires, steps=update)


GapJunction_single = nb.SynType('GapJunction', create_func=gap_junction_single, vector_based=False)


def gap_junction_lif_vector(spikelet=0.1):
    requires = dict(
        ST=nb.types.SynState(
            ['s', 'w'],
            help='''
                Gap junction state.

                s : conductance for post-synaptic neuron.
                w : gap junction conductance. It
                '''
        ),
        pre=nb.types.NeuState(['V', 'sp']),
        post=nb.types.NeuState(['V', 'inp']),
        post2syn=nb.types.ListConn(help='post-to-synapse connection.'),
        pre_ids=nb.types.Array(dim=1, help='Pre-synaptic neuron indices.'),
    )

    def update(ST, pre, post, post2syn, pre_ids):
        num_post = len(post2syn)
        post_cond = np.zeros(num_post, np.float_)
        post_spikelet = np.zeros(num_post, np.float_)
        for post_id in range(num_post):
            for syn_id in post2syn[post_id]:
                pre_id = pre_ids[syn_id]
                post_cond[post_id] = ST['w'] * (pre['V'][pre_id] - post['V'][post_id])
                post_spikelet[post_id] = ST['w'][syn_id] * spikelet * pre['sp']
        post['inp'] += post_cond
        post['V'] += post_spikelet

    return dict(requires=requires, steps=update)


GapJunction_LIF = nb.SynType('GapJunctin_for_LIF', create_func=gap_junction_lif_vector, vector_based=True)


