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
# - Susin, Eduarda, and Alain Destexhe. “Integration, coincidence detection and resonance in networks of
#   spiking neurons expressing gamma oscillations and asynchronous states.” PLoS computational biology 17.9 (2021): e1009416.
#
# CHING Network for Generating Gamma Oscillation


import brainunit as u

import brainpy.state_based as brainpy
import brainstate
from Susin_Destexhe_2021_gamma_oscillation import (
    get_inputs, visualize_simulation_results, RS_par, FS_par, Ch_par, AdEx
)


class CHINGNet(brainstate.nn.DynamicsGroup):
    def __init__(self):
        super().__init__()

        self.num_rs = 19000
        self.num_fs = 5000
        self.num_ch = 1000
        self.exc_syn_tau = 5. * u.ms
        self.inh_syn_tau = 5. * u.ms
        self.exc_syn_weight = 1. * u.nS
        self.inh_syn_weight1 = 7. * u.nS
        self.inh_syn_weight2 = 5. * u.nS
        self.ext_weight1 = 1. * u.nS
        self.ext_weight2 = 0.75 * u.nS
        self.delay = 1.5 * u.ms

        # neuronal populations
        RS_par_ = RS_par.copy()
        FS_par_ = FS_par.copy()
        Ch_par_ = Ch_par.copy()
        RS_par_.update(Vth=-50 * u.mV, V_sp_th=-40 * u.mV)
        FS_par_.update(Vth=-50 * u.mV, V_sp_th=-40 * u.mV)
        Ch_par_.update(Vth=-50 * u.mV, V_sp_th=-40 * u.mV)
        self.rs_pop = AdEx(self.num_rs, tau_e=self.exc_syn_tau, tau_i=self.inh_syn_tau, **RS_par_)
        self.fs_pop = AdEx(self.num_fs, tau_e=self.exc_syn_tau, tau_i=self.inh_syn_tau, **FS_par_)
        self.ch_pop = AdEx(self.num_ch, tau_e=self.exc_syn_tau, tau_i=self.inh_syn_tau, **Ch_par_)
        self.ext_pop = brainpy.PoissonEncoder(self.num_rs)

        # Poisson inputs
        self.ext_to_FS = brainpy.DeltaProj(
            comm=brainstate.nn.EventFixedProb(self.num_rs, self.num_fs, 0.02, self.ext_weight2),
            post=self.fs_pop,
            label='ge',
        )
        self.ext_to_RS = brainpy.DeltaProj(
            comm=brainstate.nn.EventFixedProb(self.num_rs, self.num_rs, 0.02, self.ext_weight1),
            post=self.rs_pop,
            label='ge',
        )
        self.ext_to_CH = brainpy.DeltaProj(
            comm=brainstate.nn.EventFixedProb(self.num_rs, self.num_ch, 0.02, self.ext_weight1),
            post=self.ch_pop,
            label='ge',
        )

        # synaptic projections
        self.RS_to_FS = brainpy.DeltaProj(
            self.rs_pop.prefetch('spike').delay.at(self.delay),
            comm=brainstate.nn.EventFixedProb(self.num_rs, self.num_fs, 0.02, self.exc_syn_weight),
            post=self.fs_pop,
            label='ge',
        )
        self.RS_to_RS = brainpy.DeltaProj(
            self.rs_pop.prefetch('spike').delay.at(self.delay),
            comm=brainstate.nn.EventFixedProb(self.num_rs, self.num_rs, 0.02, self.exc_syn_weight),
            post=self.rs_pop,
            label='ge',
        )
        self.RS_to_Ch = brainpy.DeltaProj(
            self.rs_pop.prefetch('spike').delay.at(self.delay),
            comm=brainstate.nn.EventFixedProb(self.num_rs, self.num_ch, 0.02, self.exc_syn_weight),
            post=self.ch_pop,
            label='ge',
        )

        # inhibitory projections
        self.FS_to_RS = brainpy.DeltaProj(
            self.fs_pop.prefetch('spike').delay.at(self.delay),
            comm=brainstate.nn.EventFixedProb(self.num_fs, self.num_rs, 0.02, self.inh_syn_weight1),
            post=self.rs_pop,
            label='gi',
        )
        self.FS_to_FS = brainpy.DeltaProj(
            self.fs_pop.prefetch('spike').delay.at(self.delay),
            comm=brainstate.nn.EventFixedProb(self.num_fs, self.num_fs, 0.02, self.inh_syn_weight2),
            post=self.fs_pop,
            label='gi',
        )
        self.FS_to_Ch = brainpy.DeltaProj(
            self.fs_pop.prefetch('spike').delay.at(self.delay),
            comm=brainstate.nn.EventFixedProb(self.num_fs, self.num_ch, 0.02, self.inh_syn_weight1),
            post=self.ch_pop,
            label='gi',
        )

        # chatter cell projections
        self.Ch_to_RS = brainpy.DeltaProj(
            self.ch_pop.prefetch('spike').delay.at(self.delay),
            comm=brainstate.nn.EventFixedProb(self.num_ch, self.num_rs, 0.02, self.exc_syn_weight),
            post=self.rs_pop,
            label='ge',
        )
        self.Ch_to_FS = brainpy.DeltaProj(
            self.ch_pop.prefetch('spike').delay.at(self.delay),
            comm=brainstate.nn.EventFixedProb(self.num_ch, self.num_fs, 0.02, self.exc_syn_weight),
            post=self.fs_pop,
            label='ge',
        )
        self.Ch_to_Ch = brainpy.DeltaProj(
            self.ch_pop.prefetch('spike').delay.at(self.delay),
            comm=brainstate.nn.EventFixedProb(self.num_ch, self.num_ch, 0.02, self.exc_syn_weight),
            post=self.ch_pop,
            label='ge',
        )

    def update(self, i, t, freq):
        with brainstate.environ.context(i=i, t=t):
            ext_spikes = self.ext_pop(freq)
            self.ext_to_FS(ext_spikes)
            self.ext_to_RS(ext_spikes)
            self.ext_to_CH(ext_spikes)

            self.RS_to_FS()
            self.RS_to_RS()
            self.RS_to_Ch()

            self.FS_to_RS()
            self.FS_to_FS()
            self.FS_to_Ch()

            self.Ch_to_RS()
            self.Ch_to_FS()
            self.Ch_to_Ch()

            self.rs_pop()
            self.fs_pop()
            self.ch_pop()

            return {
                'FS.V0': self.fs_pop.V.value[0],
                'CH.V0': self.ch_pop.V.value[0],
                'RS.V0': self.rs_pop.V.value[0],
                'FS.spike': self.fs_pop.spike.value,
                'CH.spike': self.ch_pop.spike.value,
                'RS.spike': self.rs_pop.spike.value
            }


def simulate_ching_net():
    with brainstate.environ.context(dt=0.1 * u.ms):
        # inputs
        duration = 6e3 * u.ms
        varied_rates = get_inputs(1. * u.Hz, 2. * u.Hz, 50. * u.ms, 150 * u.ms, 600 * u.ms, 1e3 * u.ms, duration)

        # network
        net = brainstate.nn.init_all_states(CHINGNet())

        # simulation
        times = u.math.arange(0. * u.ms, duration, brainstate.environ.get_dt())
        indices = u.math.arange(0, len(times))
        returns = brainstate.transform.for_loop(net.update, indices, times, varied_rates,
                                                pbar=brainstate.transform.ProgressBar(100))

        # visualization
        visualize_simulation_results(
            times=times,
            spikes={'FS': (returns['FS.spike'], 'inh'),
                    'CH': (returns['CH.spike'], 'exc'),
                    'RS': (returns['RS.spike'], 'exc')},
            example_potentials={'FS': returns['FS.V0'],
                                'CH': returns['CH.V0'],
                                'RS': returns['RS.V0']},
            varied_rates=varied_rates,
            xlim=(2e3 * u.ms, 3.4e3 * u.ms),
            t_lfp_start=1e3 * u.ms,
            t_lfp_end=5e3 * u.ms
        )


simulate_ching_net()
