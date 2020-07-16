# ---
# jupyter:
#   jupytext:
#     formats: py:light,ipynb
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.5.1
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# # Gap junction model between LIF neurons

from brian2 import *
from brian2 import __version__
print(__version__)

seed(12345)
prefs.codegen.target = "numpy"

import matplotlib.patches as patches
import npbrain.all as nn

gj_w, k_spikelet = 1., 0.5
size = (10, 10)
Vr = 0.
Vth = 10.
tau = 10 * ms
Iext = 12.
noise = 1.
duration = 500

eqs = '''
    dV/dt = (-V + Vr + Igap + Iext + sqrt(1*ms) * noise * xi) / tau : 1
    Igap : 1 # gap junction current
'''
neurons = NeuronGroup(size[0] * size[1], eqs, threshold='V>Vth', reset='V=Vr', method='euler')
neurons.V = 'rand() * (Vth - Vr) + Vr'

S = Synapses(source=neurons,
             target=neurons,
             model='''w : 1 # gap junction conductance
                      Igap_post = w * (V_pre - V_post) : 1 (summed)''',
             on_pre='V_post += w * {}'.format(k_spikelet))
pre_index, post_index, _ = nn.conn.grid_four(size[0], size[1])
S.connect(i=pre_index, j=post_index)
S.w = gj_w

mon_st = StateMonitor(neurons, 'V', record=True)
mon_sp = SpikeMonitor(neurons, record=True)
run(duration * ms)

neuron_indexes = [1]
spike_trains = mon_sp.spike_trains()

fig, gs = nn.vis.get_figure(1, 1, 6, 14)
ax = fig.add_subplot(gs[0, 0])
ax.plot([0, duration], [Vth, Vth], 'k', label='threshold')
for i in neuron_indexes:
    ax.plot(mon_st.t / ms, mon_st.V[i], label='N{}-potential'.format(i))
    spikes = spike_trains[1] / ms
    ax.plot(spikes, np.ones_like(spikes) * Vth, '.r', markersize=10, label='N{}-spikes'.format(i))
    ax.add_patch(patches.Rectangle((133, 9.8), 8, 0.4, linewidth=1, edgecolor='r', facecolor='none'))
    ax.add_patch(patches.Rectangle((170, 9.8), 8, 0.4, linewidth=1, edgecolor='r', facecolor='none'))
    ax.add_patch(patches.Rectangle((293, 9.8), 8, 0.4, linewidth=1, edgecolor='r', facecolor='none'))
ax.set_xlabel('Time (ms)')
ax.set_ylabel('Potential (mV)')
ax.legend(loc='lower center', fontsize=14)
xlim(99, 351)
ylim(-0.5, 12.)
show()
