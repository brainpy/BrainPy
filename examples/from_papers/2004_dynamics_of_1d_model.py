'''
Phase locking in leaky integrate-and-fire model
-----------------------------------------------
Fig. 2A from:
Brette R (2004). Dynamics of one-dimensional spiking neuron models.
J Math Biol 48(1): 38-56.

This shows the phase-locking structure of a LIF driven by a sinusoidal
current. When the current crosses the threshold (a<3), the model
almost always phase locks (in a measure-theoretical sense).

Faster than the version in ``brian2``.
'''

import matplotlib.pyplot as plt
import numpy as np

import npbrain as nn

nn.profile.set_backend('numpy')
N = 2000
dt = 0.1
tau = 100
V_reset = 0.
Vth = 1.
freq = 1 / tau
inputs = np.linspace(2., 4., N)
nn.profile.set_dt(dt)


def LIF(geometry, **kwargs):
    var2index = dict(V=0)
    num, geometry = nn.format_geometry(geometry)

    state = nn.initial_neu_state(1, num)
    state[0] = V_reset

    @nn.integrate
    def int_f(V, t, Isyn):
        return (-V + Isyn + 2 * np.sin(2 * np.pi * t / tau)) / tau

    def update_state(neu_state, t):
        V_new = int_f(neu_state[0], t, neu_state[-1])
        neu_state[0] = V_new
        spike_idx = nn.judge_spike(neu_state, Vth, t)
        neu_state[0][spike_idx] = V_reset

    return nn.Neurons(**locals())


lif = LIF(N)
mon = nn.SpikeMonitor(lif)
net = nn.Network(lif=lif, mon=mon)

net.run(10e3, report=True, inputs=[lif, inputs])

idx, time = np.array(mon.index), np.array(mon.time)
idx_selected = np.where(time >= 5000)[0]
idx, time = idx[idx_selected], time[idx_selected]
fig, gs = nn.visualize.get_figure(1, 1, 5, 7)
fig.add_subplot(gs[0, 0])
plt.plot((time % tau) / tau, inputs[idx], ',')
plt.xlabel('Spike phase')
plt.ylabel('Parameter a')
plt.show()
