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
# - Brunel, Nicolas, and Vincent Hakim. “Fast global oscillations in networks of integrate-and-fire neurons with low firing rates.” Neural computation 11.7 (1999): 1621-1671.
#


import brainunit as u
import jax
import matplotlib.pyplot as plt

import brainpy
import brainstate
import braintools

Vr = 10. * u.mV
theta = 20. * u.mV
tau = 20. * u.ms
delta = 2. * u.ms
taurefr = 2. * u.ms
duration = 100. * u.ms
J = .1 * u.mV
muext = 25. * u.mV
sigmaext = 1.0 * u.mV
C = 1000
N = 5000
sparseness = C / N


class LIF(brainpy.state.Neuron):
    def __init__(self, in_size, **kwargs):
        super().__init__(in_size, **kwargs)

    def init_state(self, *args, **kwargs):
        # variables
        self.V = brainstate.HiddenState(braintools.init.param(braintools.init.Constant(Vr), self.varshape))
        self.t_last_spike = brainstate.ShortTermState(
            braintools.init.param(braintools.init.Constant(-1e7 * u.ms), self.varshape)
        )

    def update(self):
        # integrate membrane potential
        fv = lambda V: (-V + self.sum_current_inputs(muext, V)) / tau
        gv = lambda V: sigmaext / u.math.sqrt(tau)
        V = brainstate.nn.exp_euler_step(fv, gv, self.V.value)
        V = self.sum_delta_inputs(V)

        # refractory period
        t = brainstate.environ.get('t')
        in_ref = (t - self.t_last_spike.value) <= taurefr
        V = u.math.where(in_ref, self.V.value, V)

        # spike
        spike = V >= theta
        self.V.value = u.math.where(spike, Vr, V)
        self.t_last_spike.value = u.math.where(spike, t, self.t_last_spike.value)
        return spike


class Net(brainstate.nn.Module):
    def __init__(self, num):
        super().__init__()
        self.group = LIF(num)
        self.delay = brainstate.nn.Delay(jax.ShapeDtypeStruct((num,), bool), delta)
        self.syn = brainpy.state.DeltaProj(
            comm=brainstate.nn.EventFixedProb(num, num, sparseness, -J),
            post=self.group
        )

    def update(self, t, i):
        with brainstate.environ.context(t=t, i=i):
            self.syn(self.delay.retrieve_at_step(jax.numpy.asarray(delta / brainstate.environ.get_dt(), dtype=int)))
            spike = self.group()
            self.delay(spike)
            return spike


with brainstate.environ.context(dt=0.1 * u.ms):
    # initialize network
    net = Net(N)
    brainstate.nn.init_all_states(net)

    # simulation
    times = u.math.arange(0. * u.ms, duration, brainstate.environ.get_dt())
    indices = u.math.arange(times.size)
    spikes = brainstate.transform.for_loop(net.update, times, indices, pbar=brainstate.transform.ProgressBar(10))

# visualization
times = times.to_decimal(u.ms)
t_indices, n_indices = u.math.where(spikes)
plt.scatter(times[t_indices], n_indices, s=1)
plt.xlabel('Time (ms)')
plt.ylabel('Neuron index')
plt.xlim([0, duration.to_decimal(u.ms)])
plt.show()
