# -*- coding: utf-8 -*-

import numpy as np
import npbrain as nn

nn.profile.set_backend('numba')
nn.profile.set_dt(dt=0.1)


def model_run(sigmaext):
    lif = nn.LIF(5000, Vr=10, Vth=20, tau=20, ref=2, noise=sigmaext * np.sqrt(20))
    conn = nn.connect.fixed_prob(lif.num, lif.num, prob=0.2, include_self=False)
    syn = nn.VoltageJumpSynapse(lif, lif, weights=-0.1, delay=2, connection=conn)
    mon = nn.SpikeMonitor(lif)

    net = nn.Network(syn=syn, lif=lif, mon=mon)
    net.run(duration=100., inputs=[lif, 25], report=True)


if __name__ == '__main__':
    all_params = [{'sigmaext': a} for a in np.arange(0.5, 1.5, 0.1)]
    nn.run.process_pool(model_run, all_params, 5)

