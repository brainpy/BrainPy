import time

import numpy
from nest import *

numpy.random.seed(98765)
SetKernelStatus({"resolution": 0.1})
# nb_threads = 4
# SetKernelStatus({"local_num_threads": int(nb_threads)})

simtime = 5 * 1000.0  # [ms] Simulation time
NE = 3200  # number of exc. neurons
NI = 800  # number of inh. neurons

SetDefaults("iaf_cond_exp", {
    "C_m": 200.,
    "g_L": 10.,
    "tau_syn_ex": 5.,
    "tau_syn_in": 10.,
    "E_ex": 0.,
    "E_in": -80.,
    "t_ref": 5.,
    "E_L": -60.,
    "V_th": -50.,
    "I_e": 200.,
    "V_reset": -60.,
    "V_m": -60.
})

nodes_ex = Create("iaf_cond_exp", NE)
nodes_in = Create("iaf_cond_exp", NI)
nodes = nodes_ex + nodes_in

# Initialize the membrane potentials
v = -55.0 + 5.0 * numpy.random.normal(size=NE + NI)
for i, node in enumerate(nodes):
    SetStatus([node], {"V_m": v[i]})

# Create the synapses
w_exc = 6.
w_inh = -67.
SetDefaults("static_synapse", {"delay": 0.1})
CopyModel("static_synapse", "excitatory", {"weight": w_exc})
CopyModel("static_synapse", "inhibitory", {"weight": w_inh})



Connect(nodes_ex, nodes,{'rule': 'pairwise_bernoulli', 'p': 0.02}, syn_spec="excitatory")
Connect(nodes_in, nodes,{'rule': 'pairwise_bernoulli', 'p': 0.02}, syn_spec="inhibitory")

# Spike detectors
SetDefaults("spike_detector", {"withtime": True,
                               "withgid": True,
                               "to_file": False})
espikes = Create("spike_detector")
ispikes = Create("spike_detector")
Connect(nodes_ex, espikes, 'all_to_all')
Connect(nodes_in, ispikes, 'all_to_all')

tstart = time.time()
Simulate(simtime)
print('Done in', time.time() - tstart)

events_ex = GetStatus(espikes, "n_events")[0]
events_in = GetStatus(ispikes, "n_events")[0]
print('Total spikes:', events_ex + events_in)

nest.raster_plot.from_device(espikes, hist=True)
nest.raster_plot.show()
