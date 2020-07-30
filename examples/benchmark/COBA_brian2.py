from brian2 import *

import npbrain as nn

defaultclock.dt = 0.05 * ms
set_device('cpp_standalone', directory='brian2_COBA')
# prefs.codegen.target = "cython"

# ###########################################
# Network parameters
# ###########################################
taum = 20 * ms
taue = 5 * ms
taui = 10 * ms
Vt = -50 * mV
Vr = -60 * mV
El = -60 * mV
Erev_exc = 0. * mV
Erev_inh = -80. * mV
I = 20. * mvolt
num_exc = 3200
num_inh = 800

# ###########################################
# Neuron model
# ###########################################
eqs = '''
dv/dt  = (ge*(Erev_exc-v)+gi*(Erev_inh-v)-(v-El) + I)*(1./taum) : volt (unless refractory)
dge/dt = -ge/taue : 1 
dgi/dt = -gi/taui : 1 
'''
net = Network()

# ###########################################
# Population
# ###########################################
P = NeuronGroup(num_exc + num_inh, eqs, threshold='v>Vt', reset='v = Vr',
                refractory=5 * ms, method='euler')

# ###########################################
# Projections
# ###########################################

exc_pre, exc_post, exc_acs = nn.connect.fixed_prob(
    num_exc, num_exc + num_inh, 0.02, include_self=False)
exc_anchors = np.zeros((2, num_exc + num_inh), dtype=np.int32)
exc_anchors[:, :num_exc] = exc_acs

inh_pre, inh_post, inh_anchors = nn.connect.fixed_prob(
    list(range(num_exc, num_exc + num_inh)),
    num_exc + num_inh, 0.02, include_self=False)

we = 0.6  # excitatory synaptic weight (voltage)
wi = 6.7  # inhibitory synaptic weight
Ce = Synapses(P, P, on_pre='ge += we')
Ci = Synapses(P, P, on_pre='gi += wi')

# ###########################################
# initialization
# ###########################################

P.v = (np.random.randn(num_exc + num_inh) * 5. - 55.) * mvolt

Ce.connect(i=exc_pre, j=exc_post)
Ci.connect(i=inh_pre, j=inh_post)

# ###########################################
# Simulation
# ###########################################
s_mon = SpikeMonitor(P)

net.add(s_mon, P, Ce, Ci)

# Run for 0 second in order to measure compilation time
t1 = time.time()
net.run(5. * second, report='text')
t2 = time.time()
print('Done in', t2 - t1)

# ###########################################
# Data analysis
# ###########################################
plot(s_mon.t / ms, s_mon.i, ',k')
xlabel('Time (ms)')
ylabel('Neuron index')
show()
