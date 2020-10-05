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
import npbrain as nb

nb.profile.set_backend('numba')
nb.profile.set_dt(0.01)

lif = nb.LIF(5000, Vr=10, Vth=20, tau=20, ref=2, noise=np.sqrt(20))
conn = nb.connect.fixed_prob(lif.num, lif.num, prob=0.2, include_self=False)
syn = nb.VoltageJumpSynapse(lif, lif, weights=-0.1, delay=2, connection=conn)
mon = nb.SpikeMonitor(lif)

net = nb.Network(syn=syn, lif=lif, mon=mon)
net.run(duration=100., inputs=[lif, 25], report=True)

nb.visualize.plot_raster(mon, show=True)
