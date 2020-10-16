# -*- coding: utf-8 -*-

import npbrain . numpy as np
import npbrain as nb


def VoltageJumpSynapse():
    """Voltage jump synapses.

    .. math::

        I_{syn} = \sum J \delta(t-t_j)
    """

    requires = dict(
        ST=nb.types.SynState(['s']),
        pre=nb.types.NeuState(['sp']),
        post=nb.types.NeuState(['V', 'not_ref']),
        pre2post=nb.types.ListConn(),
    )

    def update(ST, pre, pre2post):
        s = np.zeros_like(ST['s'], dtype=np.float_)
        for pre_id in np.where(pre['sp'] > 0)[0]:
            post_ids = pre2post[pre_id]
            s[post_ids] = 1
        ST['s'] = s

    @nb.delayed
    def output(ST, post):
        post['V'] += ST['s'] * post['not_ref']

    return nb.SynType(name='VoltageJumpSynapse', requires=requires,
                      steps=(update, output), vector_based=True)
