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

nn.profile.set_backend('numba')
nn.profile.set_dt(dt=0.1)

Vr = 10
theta = 20
tau = 20
delta = 2
taurefr = 2
duration = 1000
C = 1000
N = 5000
sparseness = float(C) / N
J = .1
muext = 25
sigmaext = 1.


lif = nn.LIF(N, Vr=Vr, Vth=theta, tau=tau, ref=taurefr,
             noise=sigmaext * np.sqrt(tau))
conn = nn.connect.fixed_prob(lif.num, lif.num, sparseness, False)
syn = nn.VoltageJumpSynapse(lif, lif, -J, delay=delta, connection=conn)
mon = nn.SpikeMonitor(lif)

net = nn.Network(syn=syn, lif=lif, mon=mon)
net.run(duration, inputs=[lif, muext], report=True)

nn.visualize.plot_raster(mon, show=True)
