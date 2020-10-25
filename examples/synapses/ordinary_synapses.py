# -*- coding: utf-8 -*-

import brainpy as bp
import brainpy.numpy as np


def VoltageJumpSynapse():
    """Voltage jump synapses.

    .. math::

        I_{syn} = \sum J \delta(t-t_j)
    """

    requires = dict(
        ST=bp.types.SynState(['s']),
        pre=bp.types.NeuState(['sp']),
        post=bp.types.NeuState(['V', 'not_ref']),
        pre2post=bp.types.ListConn(),
    )

    def update(ST, pre, pre2post):
        s = np.zeros_like(ST['s'], dtype=np.float_)
        for pre_id in np.where(pre['sp'] > 0)[0]:
            post_ids = pre2post[pre_id]
            s[post_ids] = 1
        ST['s'] = s

    @bp.delayed
    def output(ST, post):
        post['V'] += ST['s'] * post['not_ref']

    return bp.SynType(name='VoltageJumpSynapse',
                      requires=requires,
                      steps=(update, output),
                      vector_based=True)
