# -*- coding: utf-8 -*-


import numpy as np

import npbrain as nn

npbrain.profile.set_backend('numpy')
npbrain.profile.set_dt(dt=0.1)

Vr = 10
theta = 20
tau = 20
delta = 2
taurefr = 2
duration = 100
C = 1000
N = 5000
sparseness = float(C) / N
J = .1
muext = 25


def model_run(sigmaext, lock):
    lif = nn.LIF(N, Vr=Vr, Vth=theta, tau=tau, ref=taurefr,
                 noise=sigmaext * np.sqrt(tau))
    mon = nn.SpikeMonitor(lif)
    conn = nn.connect.fixed_prob(lif.num, lif.num, sparseness, False)
    syn = nn.VoltageJumpSynapse(lif, lif, -J, connection=conn, delay=delta)

    net = nn.Network(syn=syn, lif=lif, mon=mon)

    net.run(duration, inputs=[lif, muext], report=True, repeat=True)
    # nn.visualization.plot_raster(mon, show=True)

    lock.acquire()
    with open('results.txt', 'a') as fout:
        fout.write("sigmaext = {:.3f}\n".format(sigmaext))
    lock.release()


if __name__ == '__main__':
    all_params = [{'sigmaext': a} for a in np.arange(0.5, 1.5, 0.1)]
    nn.run.process_pool_lock(model_run, all_params, 5)

