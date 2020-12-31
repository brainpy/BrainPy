from brian2 import *

defaultclock.dt = 0.05 * ms
set_device('cpp_standalone', directory='brian2_COBA')
# prefs.codegen.target = "cython"

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

eqs = '''
dv/dt  = (ge*(Erev_exc-v)+gi*(Erev_inh-v)-(v-El) + I)*(1./taum) : volt (unless refractory)
dge/dt = -ge/taue : 1 
dgi/dt = -gi/taui : 1 
'''
net = Network()

P = NeuronGroup(num_exc + num_inh, eqs, threshold='v>Vt', reset='v = Vr',
                refractory=5 * ms, method='euler')


we = 0.6  # excitatory synaptic weight (voltage)
wi = 6.7  # inhibitory synaptic weight
Ce = Synapses(P[:3200], P, on_pre='ge += we')
Ci = Synapses(P[3200:], P, on_pre='gi += wi')


P.v = (np.random.randn(num_exc + num_inh) * 5. - 55.) * mvolt
Ce.connect(p=0.02)
Ci.connect(p=0.02)

s_mon = SpikeMonitor(P)


# Run for 0 second in order to measure compilation time
t1 = time.time()
run(5. * second, report='text')
t2 = time.time()
print('Done in', t2 - t1)

plot(s_mon.t / ms, s_mon.i, '.k')
xlabel('Time (ms)')
ylabel('Neuron index')
show()
