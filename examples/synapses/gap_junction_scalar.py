# -*- coding: utf-8 -*-

import npbrain as nb


def gap_junction():
    requires = dict(
        ST=nb.types.SynState(['w'], help='w : gap junction conductance.'),
        pre=nb.types.NeuState(['V']),
        post=nb.types.NeuState(['V', 'inp']),
    )

    def update(ST, pre, post):
        post['inp'] += ST['w'] * (pre['V'] - post['V'])

    return dict(requires=requires, steps=update)


GapJunction_scalar = nb.SynType('GapJunction', create_func=gap_junction, vector_based=False)


def gap_junction_lif_scalar(spikelet=0.1):
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

    @nb.delay_push
    def update(ST, pre, post):
        # gap junction sub-threshold
        post['inp'] += ST['w'] * (pre['V'] - post['V'])
        # gap junction supra-threshold
        ST['spikelet'] = ST['w'] * spikelet * pre['sp']

    @nb.delay_pull
    def output(ST, post):
        post['V'] += ST['spikelet']

    return dict(requires=requires, steps=update)


GapJunction_LIF = nb.SynType('GapJunctin_for_LIF', create_func=gap_junction_lif_scalar, vector_based=True)


