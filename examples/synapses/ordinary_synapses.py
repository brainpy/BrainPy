# -*- coding: utf-8 -*-

import brainpy as bp
import brainpy.numpy as np


def get_VoltageJumpSynapse(post_has_refractory=False):
    """Voltage jump synapses with post-synaptic neuron refractory.

    .. math::

        I_{syn} = \sum J \delta(t-t_j)
    """

    requires = dict(
        ST=bp.types.SynState(['s']),
        pre=bp.types.NeuState(['sp']),
        pre2post=bp.types.ListConn(),
    )

    if post_has_refractory:
        requires['post'] = bp.types.NeuState(['V', 'refractory'])

        @bp.delayed
        def output(ST, post):
            post['V'] += ST['s'] * (1. - post['refractory'])

    else:
        requires['post'] = bp.types.NeuState(['V'])

        @bp.delayed
        def output(ST, post):
            post['V'] += ST['s']

    def update(ST, pre, post, pre2post):
        num_post = post['V'].shape[0]
        s = np.zeros_like(num_post, dtype=np.float_)
        for pre_id in range(pre['sp'].shape[0]):
            if pre['sp'][pre_id] > 0.:
                post_ids = pre2post[pre_id]
                s[post_ids] = 1.
        ST['s'] = s

    return bp.SynType(name='VoltageJumpSynapse',
                      requires=requires,
                      steps=(update, output),
                      mode='vector')
