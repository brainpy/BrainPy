from nest import *
import nest.raster_plot
import time, pickle
import numpy

# ###########################################
# Configuration
# ###########################################
numpy.random.seed(98765)
SetKernelStatus({"resolution": 0.1})
if len(sys.argv) > 1:
    nb_threads = sys.argv[1]
else:
    nb_threads = 1
SetKernelStatus({"local_num_threads": int(nb_threads)})

# ###########################################
# Network parameters
# ###########################################
simtime = 10000.0  # [ms] Simulation time
NE = 3200  # number of exc. neurons
NI = 800  # number of inh. neurons

# ###########################################
# Neuron model
# ###########################################
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

# ###########################################
# Population
# ###########################################
nodes_ex = Create("iaf_cond_exp", NE)
nodes_in = Create("iaf_cond_exp", NI)
nodes = nodes_ex + nodes_in

# Initialize the membrane potentials
v = -55.0 + 5.0 * numpy.random.normal(size=NE + NI)
for i, node in enumerate(nodes):
    SetStatus([node], {"V_m": v[i]})

# ###########################################
# Projections
# ###########################################
# Create the synapses
w_exc = 6.
w_inh = -67.
SetDefaults("static_synapse", {"delay": 0.1})
CopyModel("static_synapse", "excitatory",
          {"weight": w_exc})
CopyModel("static_synapse", "inhibitory",
          {"weight": w_inh})

# Create the projections
mate = pickle.load(open('exc.data', 'rb'))
mati = pickle.load(open('inh.data', 'rb'))
for i in range(NE):
    post = list(mate.rows[i])
    Connect([nodes_ex[i]], [nodes[p] for p in post], 'all_to_all', syn_spec="excitatory")
for i in range(NI):
    post = list(mati.rows[i])
    Connect([nodes_in[i]], [nodes[p] for p in post], 'all_to_all', syn_spec="inhibitory")

# Connect(nodes_ex, nodes,{'rule': 'pairwise_bernoulli', 'p': 0.02}, syn_spec="excitatory")
# Connect(nodes_in, nodes,{'rule': 'pairwise_bernoulli', 'p': 0.02}, syn_spec="inhibitory")

# Spike detectors
SetDefaults("spike_detector", {"withtime": True,
                               "withgid": True,
                               "to_file": False})
espikes = Create("spike_detector")
ispikes = Create("spike_detector")
Connect(nodes_ex, espikes, 'all_to_all')
Connect(nodes_in, ispikes, 'all_to_all')

# ###########################################
# Simulation
# ###########################################
tstart = time.time()
Simulate(simtime)
print('Done in', time.time() - tstart)

# ###########################################
# Data analysis
# ###########################################
events_ex = GetStatus(espikes, "n_events")[0]
events_in = GetStatus(ispikes, "n_events")[0]
print('Total spikes:', events_ex + events_in)

# nest.raster_plot.from_device(espikes, hist=True)
# nest.raster_plot.show()
