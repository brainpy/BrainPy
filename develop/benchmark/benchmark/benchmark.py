
#modify from https://github.com/BindsNET/bindsnet

import os
import nengo
import torch
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from time import time as t

import brian2 as b2
import nest
import ANNarchy
import bindsnet

import brainpy as bp
import bpmodels
from bpmodels.neurons import get_LIF
import pdb

#from experiments.benchmark import plot_benchmark
figure_path = os.path.abspath('.')
benchmark_path = os.path.abspath('.')
if not os.path.isdir(benchmark_path):
    os.makedirs(benchmark_path)

# "Warm up" the CPU.
torch.set_default_tensor_type("torch.FloatTensor")
x = torch.rand(1000)
del x

# BRIAN2 clock
defaultclock = 1.0 * b2.ms            


def get_simple(g_max=0.10, E=0., tau_decay=2.0, mode = 'matrix'):

    ST = bp.types.SynState(['s'])
    
    requires = {
        'pre': bp.types.NeuState(['spike']),
        'post': bp.types.NeuState(['V', 'input']),
        'conn_mat': bp.types.MatConn()
    }
    
    @bp.delayed
    def output(ST, pre, post, conn_mat):
        g = g_max * (pre['spike'].reshape((-1, 1)) * conn_mat)
        post['input'] -= g.sum(axis=0) * (post['V'] - E)

    return bp.SynType(name='simple_synapse',
                      ST=ST,
                      requires=requires,
                      steps=output,
                      mode = mode)

def BrainPy_cpu(n_neurons, time):
    '''
    UNIFY:
    poisson input: Y
    input freq:    15.
    dt:            1.0
    n_neuron:      Y
    conn:          all2all(n_neuron)
    neuron type:   LIF
    neuron params: N
    synapse type:  AMPA
    one thread:    Y
    '''
    t0 = t()
    dt = 1.0  # update variables per <dt> ms
    bp.profile.set(jit=True, device = 'cpu', dt=dt, merge_steps=True)
    
    t1 = t()
    
    LIF_neuron = get_LIF()
    sim_synapse = get_simple()  ###??? synapse type???
    pre_neu = bp.inputs.PoissonInput(geometry = (n_neurons,), freqs = 15.)
    post_neu = bp.NeuGroup(LIF_neuron, geometry = (n_neurons, ))
    post_neu.pars['tau'] = 10.
    post_neu.pars['V_th'] = -54.
    post_neu.pars['V_rest'] = -74.
    post_neu.pars['V_reset'] = -60.
    post_neu.ST['V'] = -60.
    syn = bp.SynConn(sim_synapse, pre_group = pre_neu, post_group = post_neu,
                     conn = bp.connect.All2All(), delay = 10.)
    syn.ST['s'] = np.random.rand(n_neurons, n_neurons)
    net = bp.Network(pre_neu, syn, post_neu)
    
    t2 = t()
    net.run(duration=time, inputs=[], report=True)
        
    return t() - t0, t() - t1, t() - t2


def BindsNET_cpu(n_neurons, time):
    '''
    UNIFY:
    poisson input: Y
    input freq:    ???
    dt:            ???
    n_neuron:      Y
    conn:          all2all(n_neuron^2)
    neuron type:   LIF
    neuron params: Y(most)
    synapse type:  Y (rand(0, 1)?)
    one thread:    Y?
    '''
    t0 = t()

    torch.set_default_tensor_type("torch.FloatTensor")
    torch.set_num_threads(1)

    t1 = t()

    network = bindsnet.network.Network()
    network.add_layer(bindsnet.network.nodes.Input(n=n_neurons), name="X")
    network.add_layer(bindsnet.network.nodes.LIFNodes(n=n_neurons, thresh = -54., rest = -74.,
                                                      reset = -60.), name="Y")
    network.add_connection(
        bindsnet.network.topology.Connection(source=network.layers["X"], target=network.layers["Y"]),
        source="X",
        target="Y",
    )

    data = {"X": bindsnet.encoding.poisson(datum=torch.rand(n_neurons), time=time)}
    
    t2 = t()
    network.run(inputs=data, time=time)

    return t() - t0, t() - t1, t() - t2


def BRIAN2(n_neurons, time):
    '''
    UNIFY:
    poisson input: Y
    input freq:    15.
    dt:            1.0
    n_neuron:      Y
    conn:          all2all
    neuron type:   LIF
    neuron params: N
    synapse type:  simple
    one thread:    Y
    '''
    t0 = t()

    b2.set_device('cpp_standalone', build_on_run=False)
    #b2.prefs.devices.cpp_standalone.openmp_threads = XX  #multi thread
    b2.defaultclock = 1.0 * b2.ms
    #b2.device.build()

    t1 = t()

    eqs_neurons = """
        dv/dt = (ge * (-60 * mV) + (-74 * mV) - v) / (10 * ms) : volt
        dge/dt = -ge / (5 * ms) : 1
    """

    input = b2.PoissonGroup(n_neurons, rates=15. * b2.Hz)
    neurons = b2.NeuronGroup(
        n_neurons,
        eqs_neurons,
        threshold="v > (-54 * mV)",
        reset="v = -60 * mV",
        method="exact",
    )
    S = b2.Synapses(input, neurons, """w: 1""")
    S.connect(p=1.0)
    S.w = "rand() * 0.01"  #???

    t2 = t()
    b2.run(time * b2.ms, report='stdout', report_period = 0.1 * b2.second)

    return t() - t0, t() - t1, t() - t2


def PyNEST(n_neurons, time):
    '''
    UNIFY:
    poisson input: Y
    input freq:    15.
    dt:            1.0
    n_neuron:      Y
    conn:          all2all
    neuron type:   LIF
    neuron params: ???
    synapse type:  simple
    one thread:    Y
    '''
    t0 = t()

    nest.ResetKernel()
    nest.SetKernelStatus({"local_num_threads": 1, "resolution": 1.0})

    t1 = t()

    r_ex = 15.0  # [Hz] rate of exc. neurons

    neuron = nest.Create("iaf_psc_alpha", n_neurons)
    noise = nest.Create("poisson_generator", n_neurons)

    nest.SetStatus(noise, [{"rate": r_ex}])
    nest.Connect(noise, neuron, conn_spec = 'all_to_all')

    t2 = t()
    nest.Simulate(time)

    return t() - t0, t() - t1, t() - t2


def ANNarchy_cpu(n_neurons, time):  #HH https://annarchy.readthedocs.io/en/latest/example/HodgkinHuxley.html?highlight=HH#Hodgkin-Huxley-neuron
    '''
    UNIFY:
    poisson input: Y
    input freq:    15.
    dt:            1.0
    n_neuron:      Y
    conn:          all2all
    neuron type:   LIF
    neuron params: N
    synapse type:  None--Increases the post-synaptic conductance from the synaptic efficency after each pre-synaptic spike?
    one thread:    Y
    '''
    t0 = t()
    ANNarchy.setup(paradigm="openmp", num_threads = 1, dt=1.0)
    ANNarchy.clear()

    t1 = t()

    IF = ANNarchy.Neuron(
        parameters="""
            tau_m = 10.0
            tau_e = 5.0
            vt = -54.0
            vr = -60.0
            El = -74.0
            Ee = 0.0
        """,
        equations="""
            tau_m * dv/dt = El - v + g_exc *  (Ee - vr) : init = -60.0
            tau_e * dg_exc/dt = - g_exc
        """,
        spike="""
            v > vt
        """,
        reset="""
            v = vr
        """,
    )

    Input = ANNarchy.PoissonPopulation(name="Input", geometry=n_neurons, rates=15.0)
    Output = ANNarchy.Population(name="Output", geometry=n_neurons, neuron=IF)
    proj = ANNarchy.Projection(pre=Input, post=Output, target="exc", synapse=None)
    proj.connect_all_to_all(weights=ANNarchy.Uniform(0.0, 1.0))

    ANNarchy.compile()
    t2 = t()
    ANNarchy.simulate(duration=time)

    return t() - t0, t() - t1, t() - t2
    
    
    
def Nengo(n_neurons, time):
    '''
    UNIFY:
    poisson input: N(LIF)
    input freq:    ???
    dt:            ???
    n_neuron:      Y
    conn:          ???
    neuron type:   LIF
    neuron params: ???
    synapse type:  ???
    one thread:    ???(cant find)+
    '''
    t0 = t()
    t1 = t()

    model = nengo.Network()
    with model:
        X = nengo.Ensemble(n_neurons, dimensions=n_neurons, neuron_type=nengo.LIF())
        Y = nengo.Ensemble(n_neurons, dimensions=n_neurons, neuron_type=nengo.LIF())
        nengo.Connection(X, Y, transform=np.random.rand(n_neurons, n_neurons))

    t2 = t()
    with nengo.Simulator(model) as sim:
        sim.run(time / 1000)  #dt?

    return t() - t0, t() - t1, t() - t2
    
    #have Izh but no HH

def write(start=100, stop=1000, step=100, time=1000, name = None, data = None):
    print(data)
    
    filename = "benchmark_LIF_" + name + f"_{start}_{stop}_{step}_{time}.csv"
    f = os.path.join(benchmark_path, filename)
    if os.path.isfile(f):
        os.remove(f)
    df = pd.DataFrame.from_dict(data)
    df.index = list(range(start, stop + step, step))

    print()
    print(df)
    print()

    df.to_csv(f)


def main(start=100000, stop=100001, step=100, time=1000):
    
    total_times = {
        "BindsNET_cpu": [],
        "BRIAN2": [],
        "PyNEST": [],
        "ANNarchy_cpu": [],
        'Nengo': [],
        'BrainPy_cpu': []
    }
    
    build_times = {
        "BindsNET_cpu": [],
        "BRIAN2": [],
        "PyNEST": [],
        "ANNarchy_cpu": [],
        'Nengo': [],
        'BrainPy_cpu': []
    }
    
    sim_times = {
        "BindsNET_cpu": [],
        "BRIAN2": [],
        "PyNEST": [],
        "ANNarchy_cpu": [],
        'Nengo': [],
        'BrainPy_cpu': []
    }

    for n_neurons in range(start, stop + step, step):
        print(f"\nRunning benchmark with {n_neurons} neurons.")
        for framework in sim_times.keys():
            if n_neurons > 5000 and framework == "ANNarchy_cpu" or \
               n_neurons > 2500 and framework == "PyNEST":
                total_times[framework].append(np.nan)
                build_times[framework].append(np.nan)
                sim_times[framework].append(np.nan)
                continue

            print(f"- {framework}:", end=" ")

            fn = globals()[framework]
            total, build, sim = fn(n_neurons=n_neurons, time=time)
            total_times[framework].append(total)
            build_times[framework].append(build)
            sim_times[framework].append(sim)

            print(f"(total: {total:.4f}; build: {build:.4f};sim: {sim:.4f})")
            
    #write(start = start, stop = stop, step = step, time = time, name = 'total', data = total_times)
    #write(start = start, stop = stop, step = step, time = time, name = 'build', data = build_times)
    write(start = start, stop = stop, step = step, time = time, name = 'sim', data = sim_times)

    


if __name__ == "__main__":
    # get params
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", type=int, default=100)
    parser.add_argument("--stop", type=int, default=1000)
    parser.add_argument("--step", type=int, default=100)
    parser.add_argument("--time", type=int, default=1000)
    args = parser.parse_args()

    main(
        start=args.start,
        stop=args.stop,
        step=args.step,
        time=args.time
    )
