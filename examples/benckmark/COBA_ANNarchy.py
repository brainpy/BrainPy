from ANNarchy import *

# ###########################################
# Configuration
# ###########################################
setup(dt=0.05)

# ###########################################
# Network parameters
# ###########################################
NE = 3200        # Number of excitatory cells
NI = 800          # Number of inhibitory cells
duration = 10.0 * 1000.0 # Total time of the simulation

# ###########################################
# Neuron model
# ###########################################
COBA = Neuron(
    parameters="""
        El = -60.0  : population
        Vr = -60.0  : population
        Erev_exc = 0.0  : population
        Erev_inh = -80.0  : population
        Vt = -50.0   : population
        tau = 20.0   : population
        tau_exc = 5.0   : population
        tau_inh = 10.0  : population
        I = 20.0 : population
    """,
    equations="""
        tau * dv/dt = (El - v) + g_exc * (Erev_exc - v) + g_inh * (Erev_inh - v ) + I

        tau_exc * dg_exc/dt = - g_exc 
        tau_inh * dg_inh/dt = - g_inh 
    """,
    spike = """
        v > Vt
    """,
    reset = """
        v = Vr
    """,
    refractory = 5.0
)

# ###########################################
# Population
# ###########################################
P = Population(geometry=NE+NI, neuron=COBA)
Pe = P[:NE]
Pi = P[NE:]
P.v = Normal(-55.0, 5.0)

# ###########################################
# Projections
# ###########################################
we = 6. / 10. # excitatory synaptic weight (voltage)
wi = 67. / 10. # inhibitory synaptic weight

Ce = Projection(pre=Pe, post=P, target='exc')
Ce.connect_fixed_probability(weights=we, probability=0.02)
Ci = Projection(pre=Pi, post=P, target='inh')
Ci.connect_fixed_probability(weights=wi, probability=0.02)
m = Monitor(P, 'spike')

# ###########################################
# Simulation
# ###########################################
t1 = time.time()
compile(clean=True)
simulate(duration, measure_time=True)
t2 = time.time()
print('Done in', t2 - t1)

# ###########################################
# Data analysis
# ###########################################
t, n = m.raster_plot()
print(('Number of spikes:', len(t)))

from pylab import *
plot(t, n, '.')
xlabel('Time (ms)')
ylabel('Neuron index')
show()
