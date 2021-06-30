# -*- coding: utf-8 -*-

from ANNarchy import *
from brian2 import *
from nest import *
import brainpy as bp

import os
import json
import time
import numpy as np
import matplotlib.pyplot as plt

dt = 0.05
setup(dt=dt)


def run_brianpy(num_neu, duration, device='cpu'):
    bp.backend.set('numba', dt=dt)

    # Parameters
    num_inh = int(num_neu / 5)
    num_exc = num_neu - num_inh
    taum = 20
    taue = 5
    taui = 10
    Vt = -50
    Vr = -60
    El = -60
    Erev_exc = 0.
    Erev_inh = -80.
    I = 20.
    we = 0.6  # excitatory synaptic weight (voltage)
    wi = 6.7  # inhibitory synaptic weight
    ref = 5.0

    class LIF(bp.NeuGroup):
        target_backend = ['numpy', 'numba']

        def __init__(self, size, **kwargs):
            super(LIF, self).__init__(size=size, **kwargs)
            # variables
            self.V = bp.ops.zeros(size)
            self.spike = bp.ops.zeros(size)
            self.ge = bp.ops.zeros(size)
            self.gi = bp.ops.zeros(size)
            self.input = bp.ops.zeros(size)
            self.t_last_spike = bp.ops.ones(size) * -1e7

        @staticmethod
        @bp.odeint
        def int_g(ge, gi, t):
            dge = - ge / taue
            dgi = - gi / taui
            return dge, dgi

        @staticmethod
        @bp.odeint
        def int_V(V, t, ge, gi):
            dV = (ge * (Erev_exc - V) + gi * (Erev_inh - V) + El - V + I) / taum
            return dV

        def update(self, _t, _i):
            self.ge, self.gi = self.int_g(self.ge, self.gi, _t)
            for i in range(self.size[0]):
                self.spike[i] = 0.
                if (_t - self.t_last_spike[i]) > ref:
                    V = self.int_V(self.V[i], _t, self.ge[i], self.gi[i])
                    if V >= Vt:
                        self.V[i] = Vr
                        self.spike[i] = 1.
                        self.t_last_spike[i] = _t
                    else:
                        self.V[i] = V
                self.input[i] = I

    class ExcSyn(bp.TwoEndConn):
        target_backend = ['numpy', 'numba']

        def __init__(self, pre, post, conn, **kwargs):
            self.conn = conn(pre.size, post.size)
            self.pre2post = self.conn.requires('pre2post')
            super(ExcSyn, self).__init__(pre=pre, post=post, **kwargs)

        def update(self, _t, _i):
            for pre_id, spike in enumerate(self.pre.spike):
                if spike > 0:
                    for post_i in self.pre2post[pre_id]:
                        self.post.ge[post_i] += we

    class InhSyn(bp.TwoEndConn):
        target_backend = ['numpy', 'numba']

        def __init__(self, pre, post, conn, **kwargs):
            self.conn = conn(pre.size, post.size)
            self.pre2post = self.conn.requires('pre2post')
            super(InhSyn, self).__init__(pre=pre, post=post, **kwargs)

        def update(self, _t, _i):
            for pre_id, spike in enumerate(self.pre.spike):
                if spike > 0:
                    for post_i in self.pre2post[pre_id]:
                        self.post.gi[post_i] += wi

    E_group = LIF(num_exc, monitors=['spike'])
    E_group.V = np.random.randn(num_exc) * 5. - 55.
    I_group = LIF(num_inh, monitors=['spike'])
    I_group.V = np.random.randn(num_inh) * 5. - 55.
    E2E = ExcSyn(pre=E_group, post=E_group, conn=bp.connect.FixedProb(0.02))
    E2I = ExcSyn(pre=E_group, post=I_group, conn=bp.connect.FixedProb(0.02))
    I2E = InhSyn(pre=I_group, post=E_group, conn=bp.connect.FixedProb(0.02))
    I2I = InhSyn(pre=I_group, post=I_group, conn=bp.connect.FixedProb(0.02))

    net = bp.Network(E_group, I_group, E2E, E2I, I2E, I2I)

    t0 = time.time()
    net.run(duration)
    t = time.time() - t0
    print(f'BrainPy ({device}) used time {t} s.')
    return t


def run_annarchy(num_neu, duration, device='cpu'):
    NI = int(num_neu / 5)
    NE = num_neu - NI

    clear()

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
        spike="""
            v > Vt
        """,
        reset="""
            v = Vr
        """,
        refractory=5.0
    )

    # ###########################################
    # Population
    # ###########################################
    P = Population(geometry=NE + NI, neuron=COBA)
    Pe = P[:NE]
    Pi = P[NE:]
    P.v = Normal(-55.0, 5.0)

    # ###########################################
    # Projections
    # ###########################################
    we = 6. / 10.  # excitatory synaptic weight (voltage)
    wi = 67. / 10.  # inhibitory synaptic weight

    Ce = Projection(pre=Pe, post=P, target='exc')
    Ci = Projection(pre=Pi, post=P, target='inh')
    Ce.connect_fixed_probability(weights=we, probability=0.02)
    Ci.connect_fixed_probability(weights=wi, probability=0.02)

    t0 = time.time()
    compile()
    simulate(duration)
    t = time.time() - t0
    print(f'ANNarchy ({device}) used time {t} s.')
    return t


def run_brian2(num_neu, duration):
    num_inh = int(num_neu / 5)
    num_exc = num_neu - num_inh

    start_scope()
    device.reinit()
    device.activate()

    defaultclock.dt = dt * ms
    set_device('cpp_standalone', directory='brian2_COBA')
    # device.build()
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

    eqs = '''
    dv/dt  = (ge*(Erev_exc-v)+gi*(Erev_inh-v)-(v-El) + I)*(1./taum) : volt (unless refractory)
    dge/dt = -ge/taue : 1 
    dgi/dt = -gi/taui : 1 
    '''
    net = Network()

    # ###########################################
    # Population
    # ###########################################
    P = NeuronGroup(num_exc + num_inh,
                    model=eqs,
                    threshold='v>Vt', reset='v = Vr',
                    refractory=5 * ms, method='euler')
    net.add(P)

    # ###########################################
    # Projections
    # ###########################################

    we = 0.6  # excitatory synaptic weight (voltage)
    wi = 6.7  # inhibitory synaptic weight
    Ce = Synapses(P[:num_exc], P, on_pre='ge += we')
    Ci = Synapses(P[num_exc:], P, on_pre='gi += wi')
    net.add(Ce, Ci)

    P.v = (np.random.randn(num_exc + num_inh) * 5. - 55.) * mvolt
    Ce.connect(p=0.02)
    Ci.connect(p=0.02)

    t1 = time.time()
    net.run(duration * ms)
    t = time.time() - t1
    print(f'Brian2 used {t} s')
    return t


def run_pynest(num_neu, duration):
    NI = int(num_neu / 5)
    NE = num_neu - NI

    ResetKernel()
    SetKernelStatus({"resolution": dt})
    # nb_threads = 4
    # SetKernelStatus({"local_num_threads": int(nb_threads)})

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
    v = -55.0 + 5.0 * np.random.normal(size=NE + NI)
    for i, node in enumerate(nodes):
        SetStatus([node], {"V_m": v[i]})

    # Create the synapses
    w_exc = 6.
    w_inh = -67.
    SetDefaults("static_synapse", {"delay": 0.1})
    CopyModel("static_synapse", "excitatory", {"weight": w_exc})
    CopyModel("static_synapse", "inhibitory", {"weight": w_inh})

    conn_dict = {'rule': 'pairwise_bernoulli', 'p': 0.02}
    Connect(nodes_ex, nodes, conn_dict, syn_spec="excitatory")
    Connect(nodes_in, nodes, conn_dict, syn_spec="inhibitory")

    # Spike detectors
    SetDefaults("spike_detector", {"withtime": True,
                                   "withgid": True,
                                   "to_file": False})
    espikes = Create("spike_detector")
    ispikes = Create("spike_detector")
    Connect(nodes_ex, espikes, 'all_to_all')
    Connect(nodes_in, ispikes, 'all_to_all')

    t0 = time.time()
    Simulate(duration)
    t = time.time() - t0
    print(f'PyNest used {t} s')
    return t


def main(num_neurons, duration=1000, fn_output=None):
    final_results = {'setting': dict(num_neurons=num_neurons,
                                     duration=duration,
                                     dt=dt),
                     "BRIAN2": [],
                     "PyNEST": [],
                     "ANNarchy_cpu": [],
                     'BrainPy_cpu': []}

    for num_neu in num_neurons:
        print(f"Running benchmark with {num_neu} neurons.")

        if num_neu > 2500:
            final_results['PyNEST'].append(np.nan)
        else:
            t = run_pynest(num_neu, duration)
            final_results['PyNEST'].append(t)

        t = run_brianpy(num_neu, duration, device='cpu')
        final_results['BrainPy_cpu'].append(t)

        t = run_annarchy(num_neu, duration, device='cpu')
        final_results['ANNarchy_cpu'].append(t)

        t = run_brian2(num_neu, duration)
        final_results['BRIAN2'].append(t)

    if fn_output is not None:
        if not os.path.exists(os.path.dirname(fn_output)):
            os.makedirs(os.path.dirname(fn_output))
        with open(fn_output, 'w') as fout:
            json.dump(final_results, fout, indent=2)

    plt.plot(num_neurons, final_results["BRIAN2"], label="BRIAN2", linestyle="--", color="r")
    plt.plot(num_neurons, final_results["PyNEST"], label="PyNEST", linestyle="--", color="y")
    plt.plot(num_neurons, final_results["ANNarchy_cpu"], label="ANNarchy", linestyle="--", color="m")
    plt.plot(num_neurons, final_results["BrainPy_cpu"], label="BrainPy", linestyle="--", color="g")

    plt.title("Benchmark comparison of neural simulators")
    plt.xlabel("Number of input / output neurons")
    plt.ylabel("Simulation time (seconds)")
    plt.legend(loc=1, prop={"size": 5})
    xticks = [num_neurons[0], num_neurons[len(num_neurons) // 2], num_neurons[-1]]
    plt.xticks(xticks)
    plt.yscale("log")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main(list(range(500, 9001, 500)), 5000, 'results/COBA.json')
