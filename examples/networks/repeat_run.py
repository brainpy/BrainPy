"""
Dynamics of a network of sparsely connected inhibitory current-based
integrate-and-fire neurons. Individual neurons fire irregularly at
low rate but the network is in an oscillatory global activity regime
where neurons are weakly synchronized.
Reference:
    "Fast Global Oscillations in Networks of Integrate-and-Fire
    Neurons with Low Firing Rates"
    Nicolas Brunel & Vincent Hakim
    Neural Computation 11, 1621-1671 (1999)
"""

import numpy as np

import npbrain as nn
nn.profile.set_backend('numpy')


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
sigmaext = 1
dt = 0.1
npbrain.utils.profile.set_dt(dt)

lif = nn.LIF(N, dt=dt, V_reset=Vr, Vth=theta, tau=tau, ref=taurefr,
             noise=sigmaext * np.sqrt(tau))
mon = nn.SpikeMonitor(lif)
syn = nn.VoltageJumpSynapse(lif, lif, -J,
                            {'method': 'fixed_prob',
                             'prob': sparseness,
                             'include_self': False}, delay=delta)

net = nn.Network(syn=syn, lif=lif, mon=mon)
for _ in range(4):
    net.run(duration, inputs=[lif, muext], report=True, repeat=True)
    nn.vis.plot_raster(mon, show=True)
