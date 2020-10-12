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
        ST.push(s)

    def output(ST, post):
        post_cond = ST.pull()
        post['V'] += post_cond * post['not_ref']

    return dict(requires=requires, steps=(update, output))
