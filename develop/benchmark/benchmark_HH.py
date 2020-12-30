# modify from https://github.com/BindsNET/bindsnet

import argparse
import os
import pdb
from time import time as t

import ANNarchy
import bindsnet
import brian2 as b2
import nengo
import nest
import numpy as np
import pandas as pd
import torch
from bpmodels.neurons import get_HH

import brainpy as bp

# from experiments.benchmark import plot_benchmark
figure_path = os.path.abspath('benchmark')
benchmark_path = os.path.abspath('benchmark')
if not os.path.isdir(benchmark_path):
    os.makedirs(benchmark_path)

# "Warm up" the CPU.
torch.set_default_tensor_type("torch.FloatTensor")
x = torch.rand(1000)
del x

# BRIAN2 clock
defaultclock = 1.0 * b2.ms


def get_simple(g_max=0.10, E=0., tau_decay=2.0, mode='matrix'):
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
                      mode=mode)


def BrainPy_cpu(n_neurons, time):  # HH yes
    t0 = t()
    dt = 1.0  # update variables per <dt> ms
    bp.profile.set(jit=True, device='cpu', dt=dt, merge_steps=True)

    t1 = t()

    LIF_neuron = get_HH()
    sim_synapse = get_simple()
    pre_neu = bp.inputs.PoissonInput(geometry=(n_neurons,), freqs=15.)
    post_neu = bp.NeuGroup(LIF_neuron, geometry=(n_neurons,))
    syn = bp.SynConn(sim_synapse, pre_group=pre_neu, post_group=post_neu,
                     conn=bp.connect.All2All(), delay=10.)
    net = bp.Network(pre_neu, syn, post_neu)

    t2 = t()
    net.run(duration=time, inputs=[], report=True)

    return t() - t0, t() - t1, t() - t2


def BindsNET_cpu(n_neurons, time):  # HH no
    t0 = t()

    torch.set_default_tensor_type("torch.FloatTensor")
    torch.set_num_threads(1)

    t1 = t()

    network = bindsnet.network.Network()
    network.add_layer(bindsnet.network.nodes.Input(n=n_neurons), name="X")
    network.add_layer(bindsnet.network.nodes.HHNodes(n=n_neurons), name="Y")
    network.add_connection(
        bindsnet.network.topology.Connection(source=network.layers["X"], target=network.layers["Y"]),
        source="X",
        target="Y",
    )

    data = {"X": bindsnet.encoding.poisson(datum=torch.rand(n_neurons), time=time)}
    t2 = t()
    network.run(inputs=data, time=time)

    return t() - t0, t() - t1, t() - t2


def BRIAN2(n_neurons, time):  # hh yes
    t0 = t()

    b2.set_device('cpp_standalone', build_on_run=False)
    np.random.seed(42)
    b2.defaultclock.dt = 1.0 * b2.ms

    monitor = 'spike'
    area = 0.02
    unit = 1e6
    Cm = 200. / unit
    gl = 10. / unit
    g_na = 20. * 1000 / unit
    g_kd = 6. * 1000 / unit

    time_unit = 1 * b2.ms
    El = -60.
    EK = -90.
    ENa = 50.
    VT = -63.
    # Time constants
    taue = 5. * b2.ms
    taui = 10. * b2.ms
    # Reversal potentials
    Ee = 0.
    Ei = -80.
    # excitatory synaptic weight
    we = 6. / unit
    # inhibitory synaptic weight
    wi = 67. / unit

    t1 = t()

    # The model
    eqs = b2.Equations('''
        dv/dt = (gl*(El-v) + ge*(Ee-v) + gi*(Ei-v)-
                 g_na*(m*m*m)*h*(v-ENa)-
                 g_kd*(n*n*n*n)*(v-EK))/Cm/time_unit : 1
        dm/dt = (alpha_m*(1-m)-beta_m*m)/time_unit : 1
        dn/dt = (alpha_n*(1-n)-beta_n*n)/time_unit : 1
        dh/dt = (alpha_h*(1-h)-beta_h*h)/time_unit : 1
        dge/dt = -ge/taue : 1
        dgi/dt = -gi/taui : 1
        alpha_m = 0.32*(13.-v+VT)/(exp((13.-v+VT)/4)-1.) : 1
        beta_m = 0.28*(v-VT-40)/(exp((v-VT-40)/5)-1) : 1
        alpha_h = 0.128*exp((17.-v+VT)/18) : 1
        beta_h = 4./(1.+exp((40-v+VT)/5)) : 1
        alpha_n = 0.032*(15-v+VT)/(exp((15-v+VT)/5)-1.) : 1
        beta_n = .5*exp((10-v+VT)/40) : 1
    ''')

    input = b2.PoissonGroup(n_neurons, rates=15 * b2.Hz)
    neurons = b2.NeuronGroup(n_neurons, model=eqs, threshold='v>-20', method='exponential_euler')
    # Initialization
    neurons.v = 'El + (randn() * 5 - 5)'
    neurons.ge = '(randn() * 1.5 + 4) * 10. / unit'
    neurons.gi = '(randn() * 12 + 20) * 10. / unit'

    S = b2.Synapses(input, neurons, """w: 1""")
    S.connect(p=1.0)
    S.w = "rand() * 0.01"

    t2 = t()
    b2.run(time * b2.ms)

    return t() - t0, t() - t1, t() - t2


def PyNEST(n_neurons, time):  # hh yes
    t0 = t()

    nest.ResetKernel()
    nest.SetKernelStatus({"local_num_threads": 1, "resolution": 1.0})  ##???check threads???

    t1 = t()

    r_ex = 60.0  # [Hz] rate of exc. neurons  #check input???

    neuron = nest.Create("hh_psc_alpha ", n_neurons)  ##HH get
    noise = nest.Create("poisson_generator", n_neurons)

    nest.SetStatus(noise, [{"rate": r_ex}])
    nest.Connect(noise, neuron)

    t2 = t()
    nest.Simulate(time)

    return t() - t0, t() - t1, t() - t2


def ANNarchy_cpu(n_neurons, time):  # hh yes
    t0 = t()
    ANNarchy.setup(paradigm="openmp", num_threads=1, dt=1.0)
    ANNarchy.clear()

    t1 = t()

    HH_cond_exp = ANNarchy.Neuron(
        parameters="""
            gbar_Na = 20.0
            gbar_K = 6.0
            gleak = 0.01
            cm = 0.2 
            v_offset = -63.0 
            e_rev_Na = 50.0
            e_rev_K = -90.0 
            e_rev_leak = -65.0
            e_rev_E = 0.0
            e_rev_I = -80.0 
            tau_syn_E = 0.2
            tau_syn_I = 2.0
            i_offset = 0.0
            v_thresh = 0.0
        """,
        equations="""
            # Previous membrane potential
            prev_v = v

            # Voltage-dependent rate constants
            an = 0.032 * (15.0 - v + v_offset) / (exp((15.0 - v + v_offset)/5.0) - 1.0)
            am = 0.32  * (13.0 - v + v_offset) / (exp((13.0 - v + v_offset)/4.0) - 1.0)
            ah = 0.128 * exp((17.0 - v + v_offset)/18.0) 

            bn = 0.5   * exp ((10.0 - v + v_offset)/40.0)
            bm = 0.28  * (v - v_offset - 40.0) / (exp((v - v_offset - 40.0)/5.0) - 1.0)
            bh = 4.0/(1.0 + exp (( 10.0 - v + v_offset )) )

            # Activation variables
            dn/dt = an * (1.0 - n) - bn * n : init = 0.0, exponential
            dm/dt = am * (1.0 - m) - bm * m : init = 0.0, exponential
            dh/dt = ah * (1.0 - h) - bh * h : init = 1.0, exponential

            # Membrane equation
            cm * dv/dt = gleak*(e_rev_leak -v) + gbar_K * n**4 * (e_rev_K - v) \
                         + gbar_Na * m**3 * h * (e_rev_Na - v) + g_exc * (e_rev_E - v) \
                         + g_inh * (e_rev_I - v) + i_offset: exponential, init=-65.0

            # Exponentially-decaying conductances
            tau_syn_E * dg_exc/dt = - g_exc : exponential
            tau_syn_I * dg_inh/dt = - g_inh : exponential
        """,
        spike="(v > v_thresh) and (prev_v <= v_thresh)",
        reset=""
    )

    Input = ANNarchy.PoissonPopulation(name="Input", geometry=n_neurons, rates=50.0)
    Output = ANNarchy.Population(name="Output", geometry=n_neurons, neuron=HH_cond_exp)
    proj = ANNarchy.Projection(pre=Input, post=Output, target="exc", synapse=None)
    proj.connect_all_to_all(weights=ANNarchy.Uniform(0.0, 1.0))

    ANNarchy.compile()
    t2 = t()
    ANNarchy.simulate(duration=time)

    return t() - t0, t() - t1, t() - t2


'''class HH(nengo.neurons.NeuronType):
    def __init__()'''


def Nengo(n_neurons, time):  # HH no
    t0 = t()
    t1 = t()

    model = nengo.Network()
    pdb.set_trace()
    with model:
        X = nengo.Ensemble(n_neurons, dimensions=n_neurons, neuron_type=nengo.LIF())
        Y = nengo.Ensemble(n_neurons, dimensions=n_neurons, neuron_type=nengo.LIF())
        nengo.Connection(X, Y, transform=np.random.rand(n_neurons, n_neurons))

    t2 = t()
    with nengo.Simulator(model) as sim:
        sim.run(time / 1000)

    return t() - t0, t() - t1, t() - t2


def write(start=100, stop=1000, step=100, time=1000, name=None, data=None):
    print(data)

    filename = "benchmark_HH_" + name + f"_{start}_{stop}_{step}_{time}.csv"
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
        # "BindsNET_cpu": [],
        "BRIAN2": [],
        "PyNEST": [],
        "ANNarchy_cpu": [],
        # 'Nengo': [],
        'BrainPy_cpu': []
    }

    build_times = {
        # "BindsNET_cpu": [],
        "BRIAN2": [],
        "PyNEST": [],
        "ANNarchy_cpu": [],
        # 'Nengo': [],
        'BrainPy_cpu': []
    }

    sim_times = {
        # "BindsNET_cpu": [],
        "BRIAN2": [],
        "PyNEST": [],
        "ANNarchy_cpu": [],
        # 'Nengo': [],
        'BrainPy_cpu': []
    }

    for n_neurons in range(start, stop + step, step):
        print(f"\nRunning benchmark with {n_neurons} neurons.")
        for framework in sim_times.keys():
            if (n_neurons > 5000 and framework == "ANNarchy_cpu") or \
                    (n_neurons > 2500 and framework == "PyNEST"):
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

    write(start=start, stop=stop, step=step, time=time, name='total', data=total_times)
    write(start=start, stop=stop, step=step, time=time, name='build', data=build_times)
    write(start=start, stop=stop, step=step, time=time, name='sim', data=sim_times)


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
