# Copyright 2024 BrainX Ecosystem Limited. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

#
# Implementation of the paper:
#
# - Susin, Eduarda, and Alain Destexhe. “Integration, coincidence detection and resonance in networks
#   of spiking neurons expressing gamma oscillations and asynchronous states.”
#   PLoS computational biology 17.9 (2021): e1009416.
#
#  Asynchronous Network


import braintools
import brainunit as u
import matplotlib.pyplot as plt

import brainpy
import brainstate
from Susin_Destexhe_2021_gamma_oscillation import (
    get_inputs, visualize_simulation_results,
    RS_par, FS_par, Ch_par, AdEx
)


def simulate_adex_neuron(ax_v, ax_I, pars, title):
    with brainstate.environ.context(dt=0.1 * u.ms):
        # neuron
        adex = brainstate.nn.init_all_states(AdEx(1, **pars))

        def run_step(t, x):
            with brainstate.environ.context(t=t):
                adex.update(x)
                return adex.V.value

        # simulation
        duration = 1.5e3 * u.ms
        times = u.math.arange(0. * u.ms, duration, brainstate.environ.get_dt())
        inputs = get_inputs(0. * u.nA, 0.5 * u.nA, t_transition=50. * u.ms,
                            t_min_plato=500 * u.ms, t_max_plato=500 * u.ms,
                            t_gap=500 * u.ms, t_total=duration)
        vs = brainstate.transform.for_loop(run_step, times, inputs, pbar=brainstate.transform.ProgressBar(10))

        # visualization
        ax_v.plot(times.to_decimal(u.ms), vs.to_decimal(u.mV))
        ax_v.set_title(title)
        ax_v.set_ylabel('V (mV)')
        ax_v.set_xlim(0.4 * u.second / u.ms, 1.2 * u.second / u.ms)

        ax_I.plot(times.to_decimal(u.ms), inputs.to_decimal(u.nA))
        ax_I.set_ylabel('I (nA)')
        ax_I.set_xlabel('Time (ms)')
        ax_I.set_xlim(0.4 * u.second / u.ms, 1.2 * u.second / u.ms)


def simulate_adex_neurons():
    fig, gs = braintools.visualize.get_figure(2, 3, 4, 6)
    simulate_adex_neuron(fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[1, 0]), RS_par, 'Regular Spiking')
    simulate_adex_neuron(fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[1, 1]), FS_par, 'Fast Spiking')
    simulate_adex_neuron(fig.add_subplot(gs[0, 2]), fig.add_subplot(gs[1, 2]), Ch_par, 'Chattering')
    plt.show()


class AINet(brainstate.nn.Module):
    def __init__(self):
        super().__init__()

        self.num_exc = 20000
        self.num_inh = 5000
        self.exc_syn_tau = 5. * u.ms
        self.inh_syn_tau = 5. * u.ms
        self.exc_syn_weight = 1. * u.nS
        self.inh_syn_weight = 5. * u.nS
        self.delay = 1.5 * u.ms
        self.ext_weight = 1.0 * u.nS

        # neuronal populations
        RS_par_ = RS_par.copy()
        FS_par_ = FS_par.copy()
        RS_par_.update(Vth=-50 * u.mV, V_sp_th=-40 * u.mV)
        FS_par_.update(Vth=-50 * u.mV, V_sp_th=-40 * u.mV)
        self.fs_pop = AdEx(self.num_inh, tau_e=self.exc_syn_tau, tau_i=self.inh_syn_tau, **FS_par_)
        self.rs_pop = AdEx(self.num_exc, tau_e=self.exc_syn_tau, tau_i=self.inh_syn_tau, **RS_par_)
        self.ext_pop = brainpy.state.PoissonEncoder(self.num_exc)

        # Poisson inputs
        self.ext_to_FS = brainpy.state.DeltaProj(
            comm=brainstate.nn.EventFixedProb(self.num_exc, self.num_inh, 0.02, self.ext_weight),
            post=self.fs_pop,
            label='ge'
        )
        self.ext_to_RS = brainpy.state.DeltaProj(
            comm=brainstate.nn.EventFixedProb(self.num_exc, self.num_exc, 0.02, self.ext_weight),
            post=self.rs_pop,
            label='ge'
        )

        # synaptic projections
        self.RS_to_FS = brainpy.state.DeltaProj(
            self.rs_pop.prefetch('spike').delay.at(self.delay),
            comm=brainstate.nn.EventFixedProb(self.num_exc, self.num_inh, 0.02, self.exc_syn_weight),
            post=self.fs_pop,
            label='ge'
        )
        self.RS_to_RS = brainpy.state.DeltaProj(
            self.rs_pop.prefetch('spike').delay.at(self.delay),
            comm=brainstate.nn.EventFixedProb(self.num_exc, self.num_exc, 0.02, self.exc_syn_weight),
            post=self.rs_pop,
            label='ge'
        )
        self.FS_to_FS = brainpy.state.DeltaProj(
            self.fs_pop.prefetch('spike').delay.at(self.delay),
            comm=brainstate.nn.EventFixedProb(self.num_inh, self.num_inh, 0.02, self.inh_syn_weight),
            post=self.fs_pop,
            label='gi'
        )
        self.FS_to_RS = brainpy.state.DeltaProj(
            self.fs_pop.prefetch('spike').delay.at(self.delay),
            comm=brainstate.nn.EventFixedProb(self.num_inh, self.num_exc, 0.02, self.inh_syn_weight),
            post=self.rs_pop,
            label='gi'
        )

    def update(self, i, t, freq):
        with brainstate.environ.context(t=t, i=i):
            ext_spikes = self.ext_pop(freq)
            self.ext_to_FS(ext_spikes)
            self.ext_to_RS(ext_spikes)
            self.RS_to_RS()
            self.RS_to_FS()
            self.FS_to_FS()
            self.FS_to_RS()
            self.rs_pop()
            self.fs_pop()
            return {
                'FS.V0': self.fs_pop.V.value[0],
                'RS.V0': self.rs_pop.V.value[0],
                'FS.spike': self.fs_pop.spike.value,
                'RS.spike': self.rs_pop.spike.value
            }


def simulate_ai_net():
    with brainstate.environ.context(dt=0.1 * u.ms):
        # inputs
        duration = 2e3 * u.ms
        varied_rates = get_inputs(2. * u.Hz, 2. * u.Hz, 50. * u.ms, 150 * u.ms, 600 * u.ms, 1e3 * u.ms, duration)

        # network
        net = brainstate.nn.init_all_states(AINet())

        # simulation
        times = u.math.arange(0. * u.ms, duration, brainstate.environ.get_dt())
        indices = u.math.arange(0, len(times))
        returns = brainstate.transform.for_loop(net.update, indices, times, varied_rates, pbar=4000)

        # # spike raster plot
        # spikes = returns['FS.spike']
        # fig, gs = braintools.visualize.get_figure(1, 1, 4., 5.)
        # fig.add_subplot(gs[0, 0])
        # times2 = times.to_decimal(u.ms)
        # t_indices, n_indices = u.math.where(spikes)
        # plt.scatter(times2[t_indices], n_indices, s=1, c='k')
        # plt.xlabel('Time (ms)')
        # plt.ylabel('Neuron index')
        # plt.title('Spike raster plot')
        # plt.show()

        # visualization
        visualize_simulation_results(
            times=times,
            spikes={'FS': (returns['FS.spike'], 'inh'),
                    'RS': (returns['RS.spike'], 'exc')},
            example_potentials={'FS': returns['FS.V0'],
                                'RS': returns['RS.V0']},
            varied_rates=varied_rates
        )


if __name__ == '__main__':
    # simulate_adex_neurons()
    simulate_ai_net()
