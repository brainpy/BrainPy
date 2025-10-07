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
# - Brette, R., Rudolph, M., Carnevale, T., Hines, M., Beeman, D., Bower, J. M., et al. (2007),
#   Simulation of networks of spiking neurons: a review of tools and strategies., J. Comput. Neurosci., 23, 3, 349–98
#
# which is based on the balanced network proposed by:
#
# - Vogels, T. P. and Abbott, L. F. (2005), Signal propagation and logic gating in networks of integrate-and-fire neurons., J. Neurosci., 25, 46, 10786–95
#


import brainunit as u
import matplotlib.pyplot as plt

import brainpy
import brainstate
import braintools


class EINet(brainstate.nn.Module):
    def __init__(self):
        super().__init__()
        self.n_exc = 3200
        self.n_inh = 800
        self.E = brainpy.LIFRef(
            self.n_exc,
            V_rest=-49. * u.mV, V_th=-50. * u.mV, V_reset=-60. * u.mV,
            tau=20. * u.ms, tau_ref=5. * u.ms,
            V_initializer=braintools.init.Normal(-55. * u.mV, 2. * u.mV)
        )
        self.I = brainpy.LIFRef(
            self.n_inh,
            V_rest=-49. * u.mV, V_th=-50. * u.mV, V_reset=-60. * u.mV,
            tau=20. * u.ms, tau_ref=5. * u.ms,
            V_initializer=braintools.init.Normal(-55. * u.mV, 2. * u.mV)
        )
        self.E2E = brainpy.AlignPostProj(
            self.E.prefetch('V'),
            lambda x: self.E.get_spike(x) != 0.,
            comm=brainstate.nn.EventFixedProb(self.n_exc, self.n_exc, conn_num=0.02, conn_weight=1.62 * u.mS),
            syn=brainpy.Expon.desc(self.n_exc, tau=5. * u.ms),
            out=brainpy.CUBA.desc(scale=u.volt),
            post=self.E
        )
        self.E2I = brainpy.AlignPostProj(
            self.E.prefetch('V'),
            lambda x: self.E.get_spike(x) != 0.,
            comm=brainstate.nn.EventFixedProb(self.n_exc, self.n_inh, conn_num=0.02, conn_weight=1.62 * u.mS),
            syn=brainpy.Expon.desc(self.n_inh, tau=5. * u.ms),
            out=brainpy.CUBA.desc(scale=u.volt),
            post=self.I
        )
        self.I2E = brainpy.AlignPostProj(
            self.I.prefetch('V'),
            lambda x: self.I.get_spike(x) != 0.,
            comm=brainstate.nn.EventFixedProb(self.n_inh, self.n_exc, conn_num=0.02, conn_weight=-9.0 * u.mS),
            syn=brainpy.Expon.desc(self.n_exc, tau=10. * u.ms),
            out=brainpy.CUBA.desc(scale=u.volt),
            post=self.E
        )
        self.I2I = brainpy.AlignPostProj(
            self.I.prefetch('V'),
            lambda x: self.I.get_spike(x) != 0.,
            comm=brainstate.nn.EventFixedProb(self.n_inh, self.n_inh, conn_num=0.02, conn_weight=-9.0 * u.mS),
            syn=brainpy.Expon.desc(self.n_inh, tau=10. * u.ms),
            out=brainpy.CUBA.desc(scale=u.volt),
            post=self.I
        )

    def update(self, t):
        with brainstate.environ.context(t=t):
            self.E2E()
            self.E2I()
            self.I2E()
            self.I2I()
            self.E(20. * u.mA)
            self.I(20. * u.mA)
            return self.E.get_spike()


# network
net = EINet()
brainstate.nn.init_all_states(net)

# simulation
with brainstate.environ.context(dt=0.1 * u.ms):
    times = u.math.arange(0. * u.ms, 1000. * u.ms, brainstate.environ.get_dt())
    spikes = brainstate.transform.for_loop(
        net.update,
        times,
        pbar=brainstate.transform.ProgressBar(10)
    )

# visualization
t_indices, n_indices = u.math.where(spikes)
plt.scatter(times[t_indices], n_indices, s=1)
plt.xlabel('Time (ms)')
plt.ylabel('Neuron index')
plt.show()
