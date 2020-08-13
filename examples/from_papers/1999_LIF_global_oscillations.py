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
nn.profile.set_dt(0.01)
nn.profile.debug = True

lif = nn.LIF(5000, Vr=10, Vth=20, tau=20, ref=2, noise=np.sqrt(20))
conn = nn.connect.fixed_prob(lif.num, lif.num, prob=0.2, include_self=False)
syn = nn.VoltageJumpSynapse(lif, lif, weights=-0.1, delay=2, connection=conn)
mon = nn.SpikeMonitor(lif)

net = nn.Network(syn=syn, lif=lif, mon=mon)
net.run(duration=100., inputs=[lif, 25], report=True)

nn.visualize.plot_raster(mon, show=True)
