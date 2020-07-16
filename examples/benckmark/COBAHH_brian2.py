from brian2 import *

# set_device('cpp_standalone', directory='brian2_COBAHH')
prefs.codegen.target = "cython"
np.random.seed(12345)
defaultclock.dt = 0.1 * ms

# Parameters
# area = 0.02 * mmetre ** 2
# Cm=200 * pF
# gl = 10. * nsiemens
# g_na = 20 * usiemens
# g_kd = 6. * usiemens

monitor = 'spike'
area = 0.02
unit = 1e6
Cm = 200 / unit
gl = 10. / unit
g_na = 20 * 1000 / unit
g_kd = 6. * 1000 / unit

time_unit = 1 * ms
El = -60
EK = -90
ENa = 50
VT = -63
# Time constants
taue = 5 * ms
taui = 10 * ms
# Reversal potentials
Ee = 0
Ei = -80
# excitatory synaptic weight
we = 6 / unit
# inhibitory synaptic weight
wi = 67 / unit

# The model
eqs = Equations('''
    dv/dt = (gl*(El-v) + ge*(Ee-v) + gi*(Ei-v)-
             g_na*(m*m*m)*h*(v-ENa)-
             g_kd*(n*n*n*n)*(v-EK))/Cm/time_unit : 1
    dm/dt = (alpha_m*(1-m)-beta_m*m)/time_unit : 1
    dn/dt = (alpha_n*(1-n)-beta_n*n)/time_unit : 1
    dh/dt = (alpha_h*(1-h)-beta_h*h)/time_unit : 1
    dge/dt = -ge/taue : 1
    dgi/dt = -gi/taui : 1
    alpha_m = 0.32*(13-v+VT)/(exp((13-v+VT)/4)-1.) : 1
    beta_m = 0.28*(v-VT-40)/(exp((v-VT-40)/5)-1) : 1
    alpha_h = 0.128*exp((17-v+VT)/18) : 1
    beta_h = 4./(1+exp((40-v+VT)/5)) : 1
    alpha_n = 0.032*(15-v+VT)/(exp((15-v+VT)/5)-1.) : 1
    beta_n = .5*exp((10-v+VT)/40) : 1
''')

P = NeuronGroup(4000, model=eqs, threshold='v>-20', method='exponential_euler')
Pe = P[:3200]
Pi = P[3200:]
Ce = Synapses(Pe, P, on_pre='ge+=we')
Ci = Synapses(Pi, P, on_pre='gi+=wi')
Ce.connect(p=0.02)
Ci.connect(p=0.02)

# Initialization
P.v = 'El + (randn() * 5 - 5)'
P.ge = '(randn() * 1.5 + 4) * 10. / unit'
P.gi = '(randn() * 12 + 20) * 10. / unit'

# monitor
if monitor == 'V':
    trace = StateMonitor(P, 'v', record=True)
else:
    s_mon = SpikeMonitor(P)

# Record a few traces
t0 = time.time()
run(10 * second, report='text')
print('{}. Used time {} s.'.format(prefs.codegen.target, time.time() - t0))

if monitor == 'V':
    for i in [1, 10, 100]:
        plot(trace.t / ms, trace.v[i], label='N-{}'.format(i))
    xlabel('t (ms)')
    ylabel('v (mV)')
    legend()
    show()
else:
    plot(s_mon.t / ms, s_mon.i, ',k')
    xlabel('Time (ms)')
    ylabel('Neuron index')
    show()
