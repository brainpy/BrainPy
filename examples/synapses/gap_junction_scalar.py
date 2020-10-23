# -*- coding: utf-8 -*-

import brainpy as nb


def GapJunction():
    requires = dict(
        ST=nb.types.SynState(['w'], help='w : gap junction conductance.'),
        pre=nb.types.NeuState(['V']),
        post=nb.types.NeuState(['V', 'inp']),
    )

    def update(ST, pre, post):
        post['inp'] += ST['w'] * (pre['V'] - post['V'])

    return nb.SynType(name='GapJunction', requires=requires, steps=update, vector_based=False)


def LIFGapJunction(spikelet=0.1):
    requires = dict(
        ST=nb.types.SynState(
            ['w', 'spikelet'],
            help='''Gap junction state.

                s : conductance for post-synaptic neuron.
                w : gap junction conductance. 
                '''
        ),
        pre=nb.types.NeuState(['V', 'sp']),
        post=nb.types.NeuState(['V', 'inp']),
        post2syn=nb.types.ListConn(help='post-to-synapse connection.'),
        pre_ids=nb.types.Array(dim=1, help='Pre-synaptic neuron indices.'),
    )

    def update(ST, pre, post):
        # gap junction sub-threshold
        post['inp'] += ST['w'] * (pre['V'] - post['V'])
        # gap junction supra-threshold
        ST['spikelet'] = ST['w'] * spikelet * pre['sp']

    @nb.delayed
    def output(ST, post):
        post['V'] += ST['spikelet']

    return nb.SynType(name='gap_junction_for_lif', requires=requires,
                      steps=(update, output), vector_based=False)
