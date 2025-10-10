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
# ING Network for Generating Gamma Oscillation


import brainunit as u

import brainpy
import brainstate
from Susin_Destexhe_2021_gamma_oscillation import (
    get_inputs, visualize_simulation_results, RS_par, FS_par, AdEx
)


class INGNet(brainstate.nn.DynamicsGroup):
    def __init__(self):
        super().__init__()

        self.num_rs = 20000
        self.num_fs = 4000
        self.num_fs2 = 1000
        self.exc_syn_tau = 5. * u.ms
        self.inh_syn_tau = 5. * u.ms
        self.ext_weight = 0.9 * u.nS
        self.exc_syn_weight = 1. * u.nS
        self.inh_syn_weight = 5. * u.nS
        self.delay = 1.5 * u.ms

        # neuronal populations
        RS_par_ = RS_par.copy()
        FS_par_ = FS_par.copy()
        FS2_par_ = FS_par.copy()
        RS_par_.update(Vth=-50 * u.mV, V_sp_th=-40 * u.mV)
        FS_par_.update(Vth=-50 * u.mV, V_sp_th=-40 * u.mV)
        FS2_par_.update(Vth=-50 * u.mV, V_sp_th=-40 * u.mV)
        self.rs_pop = AdEx(self.num_rs, tau_e=self.exc_syn_tau, tau_i=self.inh_syn_tau, **RS_par_)
        self.fs_pop = AdEx(self.num_fs, tau_e=self.exc_syn_tau, tau_i=self.inh_syn_tau, **FS_par_)
        self.fs2_pop = AdEx(self.num_fs2, tau_e=self.exc_syn_tau, tau_i=self.inh_syn_tau, **FS2_par_)
        self.ext_pop = brainpy.state.PoissonEncoder(self.num_rs)

        # Poisson inputs
        self.ext_to_FS = brainpy.state.DeltaProj(
            comm=brainstate.nn.EventFixedProb(self.num_rs, self.num_fs, 0.02, self.ext_weight),
            post=self.fs_pop,
            label='ge'
        )
        self.ext_to_RS = brainpy.state.DeltaProj(
            comm=brainstate.nn.EventFixedProb(self.num_rs, self.num_rs, 0.02, self.ext_weight),
            post=self.rs_pop,
            label='ge'
        )
        self.ext_to_RS2 = brainpy.state.DeltaProj(
            comm=brainstate.nn.EventFixedProb(self.num_rs, self.num_fs2, 0.02, self.ext_weight),
            post=self.fs2_pop,
            label='ge'
        )

        # synaptic projections
        self.RS_to_FS = brainpy.state.DeltaProj(
            self.rs_pop.prefetch('spike').delay.at(self.delay),
            comm=brainstate.nn.EventFixedProb(self.num_rs, self.num_fs, 0.02, self.exc_syn_weight),
            post=self.fs_pop,
            label='ge'
        )
        self.RS_to_RS = brainpy.state.DeltaProj(
            self.rs_pop.prefetch('spike').delay.at(self.delay),
            comm=brainstate.nn.EventFixedProb(self.num_rs, self.num_rs, 0.02, self.exc_syn_weight),
            post=self.rs_pop,
            label='ge'
        )
        self.RS_to_FS2 = brainpy.state.DeltaProj(
            self.rs_pop.prefetch('spike').delay.at(self.delay),
            comm=brainstate.nn.EventFixedProb(self.num_rs, self.num_fs2, 0.15, self.exc_syn_weight),
            post=self.fs2_pop,
            label='ge'
        )

        self.FS_to_RS = brainpy.state.DeltaProj(
            self.fs_pop.prefetch('spike').delay.at(self.delay),
            comm=brainstate.nn.EventFixedProb(self.num_fs, self.num_rs, 0.02, self.inh_syn_weight),
            post=self.rs_pop,
            label='gi'
        )
        self.FS_to_FS = brainpy.state.DeltaProj(
            self.fs_pop.prefetch('spike').delay.at(self.delay),
            comm=brainstate.nn.EventFixedProb(self.num_fs, self.num_fs, 0.02, self.inh_syn_weight),
            post=self.fs_pop,
            label='gi'
        )
        self.FS_to_FS2 = brainpy.state.DeltaProj(
            self.fs_pop.prefetch('spike').delay.at(self.delay),
            comm=brainstate.nn.EventFixedProb(self.num_fs, self.num_fs2, 0.03, self.inh_syn_weight),
            post=self.fs2_pop,
            label='gi'
        )

        self.FS2_to_RS = brainpy.state.DeltaProj(
            self.fs2_pop.prefetch('spike').delay.at(self.delay),
            comm=brainstate.nn.EventFixedProb(self.num_fs2, self.num_rs, 0.15, self.exc_syn_weight),
            post=self.rs_pop,
            label='gi'
        )
        self.FS2_to_FS = brainpy.state.DeltaProj(
            self.fs2_pop.prefetch('spike').delay.at(self.delay),
            comm=brainstate.nn.EventFixedProb(self.num_fs2, self.num_fs, 0.15, self.exc_syn_weight),
            post=self.fs_pop,
            label='gi'
        )
        self.FS2_to_FS2 = brainpy.state.DeltaProj(
            self.fs2_pop.prefetch('spike').delay.at(self.delay),
            comm=brainstate.nn.EventFixedProb(self.num_fs2, self.num_fs2, 0.6, self.exc_syn_weight),
            post=self.fs2_pop,
            label='gi'
        )

    def update(self, i, t, freq):
        with brainstate.environ.context(t=t, i=i):
            ext_spikes = self.ext_pop(freq)
            self.ext_to_FS(ext_spikes)
            self.ext_to_RS(ext_spikes)
            self.ext_to_RS2(ext_spikes)

            self.RS_to_RS()
            self.RS_to_FS()
            self.RS_to_FS2()

            self.FS_to_RS()
            self.FS_to_FS()
            self.FS_to_FS2()

            self.FS2_to_RS()
            self.FS2_to_FS()
            self.FS2_to_FS2()

            self.rs_pop()
            self.fs_pop()
            self.fs2_pop()

            return {
                'FS.V0': self.fs_pop.V.value[0],
                'FS2.V0': self.fs2_pop.V.value[0],
                'RS.V0': self.rs_pop.V.value[0],
                'FS.spike': self.fs_pop.spike.value,
                'FS2.spike': self.fs2_pop.spike.value,
                'RS.spike': self.rs_pop.spike.value
            }


def simulate_ing_net():
    with brainstate.environ.context(dt=0.1 * u.ms):
        # inputs
        duration = 6e3 * u.ms
        varied_rates = get_inputs(2. * u.Hz, 3. * u.Hz, 50. * u.ms, 350 * u.ms, 600 * u.ms, 1e3 * u.ms, duration)

        # network
        net = brainstate.nn.init_all_states(INGNet())

        # simulation
        times = u.math.arange(0. * u.ms, duration, brainstate.environ.get_dt())
        indices = u.math.arange(0, len(times))
        returns = brainstate.transform.for_loop(net.update, indices, times, varied_rates,
                                                pbar=brainstate.transform.ProgressBar(100))

        # visualization
        visualize_simulation_results(
            times=times,
            spikes={'FS': (returns['FS.spike'], 'inh'),
                    'FS2': (returns['FS2.spike'], 'inh'),
                    'RS': (returns['RS.spike'], 'exc')},
            example_potentials={'FS': returns['FS.V0'],
                                'FS2': returns['FS2.V0'],
                                'RS': returns['RS.V0']},
            varied_rates=varied_rates,
            xlim=(2e3 * u.ms, 3.4e3 * u.ms),
            t_lfp_start=1e3 * u.ms,
            t_lfp_end=5e3 * u.ms
        )


simulate_ing_net()
