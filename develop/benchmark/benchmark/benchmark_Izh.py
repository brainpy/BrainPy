
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
import brainpy.numpy as np
import bpmodels
from bpmodels.neurons import get_Izhikevich
from bpmodels.synapses import get_AMPA1
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
        
def get_simple(g_max=0.10, E=0., tau_decay=2.0, mode = 'vector'):

    requires = {
        'ST': bp.types.SynState(['s']),
        'pre': bp.types.NeuState(['spike']),
        'post': bp.types.NeuState(['V', 'input'])
    }

    requires['post2syn']=bp.types.ListConn()
        
    def update(ST, _t_):
        ST['s'] = ST['s']

    @bp.delayed
    def output(ST, post, post2syn):
        g = np.zeros(len(post2syn), dtype=np.float_)
        for post_id, syn_ids in enumerate(post2syn):
            g[post_id] = np.sum(g_max * ST['s'][syn_ids])
        post['input'] -= g * (post['V'] - E)

    return bp.SynType(name='simple_synapse',
                      requires=requires,
                      steps=output,
                      mode = mode)

def BrainPy_cpu(n_neurons, time):  #Izh yes
    t0 = t()
    dt = 1.  # update variables per <dt> ms
    bp.profile.set(backend="numba", dt=dt, merge_steps=True)
    
    t1 = t()
    
    LIF_neuron = get_Izhikevich()
    sim_synapse = get_simple()
    pre_neu = bp.inputs.PoissonInput(geometry = (n_neurons,), freqs = 15.)
    post_neu = bp.NeuGroup(LIF_neuron, geometry = (n_neurons, ))
    syn = bp.SynConn(sim_synapse, pre_group = pre_neu, post_group = post_neu,
                     conn = bp.connect.All2All(), delay = 10.)
    net = bp.Network(pre_neu, syn, post_neu)
    
    t2 = t()
    net.run(duration=time, inputs=[], report=False)
        
    return t() - t0, t() - t1, t() - t2


def BindsNET_cpu(n_neurons, time): #HH no
    t0 = t()

    torch.set_default_tensor_type("torch.FloatTensor")

    t1 = t()

    network = bindsnet.network.Network()
    network.add_layer(bindsnet.network.nodes.Input(n=n_neurons), name="X")
    network.add_layer(bindsnet.network.nodes.IzhikevichNodes(n=n_neurons), name="Y")
    network.add_connection(
        bindsnet.network.topology.Connection(source=network.layers["X"], target=network.layers["Y"]),
        source="X",
        target="Y",
    )

    data = {"X": bindsnet.encoding.poisson(datum=torch.rand(n_neurons), time=time)}
    t2 = t()
    network.run(inputs=data, time=time)

    return t() - t0, t() - t1, t() - t2


def BRIAN2(n_neurons, time):  #hh yes
    t0 = t()
    
    b2.set_device('cpp_standalone', directory='brian2_COBAHH', build_on_run=False)
    np.random.seed(42)
    b2.defaultclock.dt = 0.1 * b2.ms

	## Neurons
    taum = 10 * b2.ms
    Ee = 0 * b2.mV
    vt = -54 * b2.mV
    vr = -60 * b2.mV
    El = -74 * b2.mV
    taue = 5 * b2.ms

    t1 = t()

    # The model
    eqs = b2.Equations(
        '''
        dv/dt = (ge * (Ee-vr) + El - v) / taum : volt
        dge/dt = -ge / taue : 1
        '''
    )
    
    input = b2.PoissonGroup(n_neurons, rates=15 * b2.Hz)
    neurons = b2.NeuronGroup(n_neurons, model=eqs, threshold='v>vt', method='exponential_euler')

    S = b2.Synapses(input, neurons, """w: 1""")
    S.connect()
    S.w = "rand() * 0.01"

    t2 = t()
    b2.run(time * b2.ms)

    return t() - t0, t() - t1, t() - t2


def PyNEST(n_neurons, time):  #hh yes
    t0 = t()

    nest.ResetKernel()
    nest.SetKernelStatus({"local_num_threads": 1, "resolution": 1.0})  ##???check threads???

    t1 = t()

    r_ex = 60.0  # [Hz] rate of exc. neurons  #check input???

    neuron = nest.Create("izhikevich", n_neurons)  ##HH get
    noise = nest.Create("poisson_generator", n_neurons)

    nest.SetStatus(noise, [{"rate": r_ex}])
    nest.Connect(noise, neuron)

    t2 = t()
    nest.Simulate(time)

    return t() - t0, t() - t1, t() - t2


def ANNarchy_cpu(n_neurons, time):  #hh yes
    t0 = t()
    ANNarchy.setup(paradigm="openmp", dt=1.0)
    ANNarchy.clear()

    t1 = t()

    Izhikevich = ANNarchy.Neuron(
        parameters = """
            noise = 0.0
            a = 0.02
            b = 0.2
            c = -65.0
            d = 8.0
            v_thresh = 30.0
            i_offset = 0.0
        """, 
        equations = """
            I = g_exc - g_inh + noise * Normal(0.0, 1.0) + i_offset
            dv/dt = 0.04 * v^2 + 5.0 * v + 140.0 - u + I : init = -65.0
            du/dt = a * (b*v - u) : init= -13.0
        """,
        spike = "v > v_thresh",
        reset = "v = c; u += d",
        refractory = 0.0
    )


    Input = ANNarchy.PoissonPopulation(name="Input", geometry=n_neurons, rates=50.0)
    Output = ANNarchy.Population(name="Output", geometry=n_neurons, neuron=Izhikevich)
    proj = ANNarchy.Projection(pre=Input, post=Output, target="exc", synapse=None)
    proj.connect_all_to_all(weights=ANNarchy.Uniform(0.0, 1.0))

    t2 = t()  ##??? put here? or after compile?
    ANNarchy.compile()
    ANNarchy.simulate(duration=time)

    return t() - t0, t() - t1, t() - t2
    
    
    
def Nengo(n_neurons, time):  #HH no
    t0 = t()
    t1 = t()

    model = nengo.Network()
    #pdb.set_trace()
    with model:
        X = nengo.Ensemble(n_neurons, dimensions=n_neurons, neuron_type=nengo.Izhikevich())
        Y = nengo.Ensemble(n_neurons, dimensions=n_neurons, neuron_type=nengo.Izhikevich())
        nengo.Connection(X, Y, transform=np.random.rand(n_neurons, n_neurons))

    t2 = t()
    with nengo.Simulator(model) as sim:
        sim.run(time / 1000)

    return t() - t0, t() - t1, t() - t2


def write(start=100, stop=1000, step=100, time=1000, name = None, data = None):
    print(data)

    filename = "benchmark_Izh_" + name + f"_{start}_{stop}_{step}_{time}.csv"
    f = os.path.join(benchmark_path, filename)
    if os.path.isfile(f):
        os.remove(f)
    df = pd.DataFrame.from_dict(data)
    df.index = list(range(start, stop + step, step))

    print()
    print(df)
    print()

    df.to_csv(f)


def main(start=100, stop=1000, step=100, time=1000):
    
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
            if n_neurons > 5000 and framework == "ANNarchy_cpu":
                total_times[framework].append(np.nan)
                build_times[framework].append(np.nan)
                sim_times[framework].append(np.nan)
                continue
                
            if n_neurons > 2500 and framework == "PyNEST":
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
            
    write(start = start, stop = stop, step = step, time = time, name = 'total', data = total_times)
    write(start = start, stop = stop, step = step, time = time, name = 'build', data = build_times)
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
