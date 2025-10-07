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
# - Wang X J, Buzsáki G. Gamma oscillation by synaptic inhibition in a hippocampal interneuronal network model[J]. Journal of neuroscience, 1996, 16(20): 6402-6413.
#

import brainunit as u
import matplotlib.pyplot as plt

import brainpy
import brainstate
import braintools


class HH(brainpy.Neuron):
    def __init__(
        self, in_size, ENa=55. * u.mV, EK=-90. * u.mV, EL=-65 * u.mV, C=1.0 * u.uF,
        gNa=35. * u.msiemens, gK=9. * u.msiemens, gL=0.1 * u.msiemens, V_th=20. * u.mV, phi=5.0
    ):
        super().__init__(in_size)

        # parameters
        self.ENa = ENa
        self.EK = EK
        self.EL = EL
        self.C = C
        self.gNa = gNa
        self.gK = gK
        self.gL = gL
        self.V_th = V_th
        self.phi = phi

    def init_state(self, *args, **kwargs):
        # variables
        self.V = brainstate.HiddenState(-70. * u.mV + brainstate.random.randn(*self.varshape) * 20 * u.mV)
        self.h = brainstate.HiddenState(braintools.init.param(braintools.init.Constant(0.6), self.varshape))
        self.n = brainstate.HiddenState(braintools.init.param(braintools.init.Constant(0.3), self.varshape))
        self.spike = brainstate.HiddenState(
            braintools.init.param(lambda s: u.math.zeros(s, dtype=bool), self.varshape))

    def dh(self, h, t, V):
        alpha = 0.07 * u.math.exp(-(V / u.mV + 58) / 20)
        beta = 1 / (u.math.exp(-0.1 * (V / u.mV + 28)) + 1)
        dhdt = alpha * (1 - h) - beta * h
        return self.phi * dhdt / u.ms

    def dn(self, n, t, V):
        alpha = -0.01 * (V / u.mV + 34) / (u.math.exp(-0.1 * (V / u.mV + 34)) - 1)
        beta = 0.125 * u.math.exp(-(V / u.mV + 44) / 80)
        dndt = alpha * (1 - n) - beta * n
        return self.phi * dndt / u.ms

    def dV(self, V, t, h, n, Iext):
        m_alpha = -0.1 * (V / u.mV + 35) / (u.math.exp(-0.1 * (V / u.mV + 35)) - 1)
        m_beta = 4 * u.math.exp(-(V / u.mV + 60) / 18)
        m = m_alpha / (m_alpha + m_beta)
        INa = self.gNa * m ** 3 * h * (V - self.ENa)
        IK = self.gK * n ** 4 * (V - self.EK)
        IL = self.gL * (V - self.EL)
        dVdt = (- INa - IK - IL + self.sum_current_inputs(Iext, V)) / self.C
        return dVdt

    def update(self, x=0. * u.uA):
        t = brainstate.environ.get('t')
        V = brainstate.nn.exp_euler_step(self.dV, self.V.value, t, self.h.value, self.n.value, x)
        h = brainstate.nn.exp_euler_step(self.dh, self.h.value, t, V)
        n = brainstate.nn.exp_euler_step(self.dn, self.n.value, t, V)
        self.spike.value = u.math.logical_and(self.V.value < self.V_th, V >= self.V_th)
        self.V.value = V
        self.h.value = h
        self.n.value = n
        return self.V.value


class Synapse(brainpy.Synapse):
    def __init__(self, in_size, alpha=12 / u.ms, beta=0.1 / u.ms):
        super().__init__(in_size=in_size)
        self.alpha = alpha
        self.beta = beta

    def init_state(self, *args, **kwargs):
        self.g = brainstate.HiddenState(
            braintools.init.param(braintools.init.ZeroInit(), self.varshape)
        )

    def update(self, pre_V):
        f_v = lambda v: 1 / (1 + u.math.exp(-v / u.mV / 2))
        ds = lambda s: self.alpha * f_v(pre_V) * (1 - s) - self.beta * s
        self.g.value = brainstate.nn.exp_euler_step(ds, self.g.value)
        return self.g.value


class GammaNet(brainstate.nn.Module):
    def __init__(self, num: int = 100):
        super().__init__()
        self.neu = HH(num)
        # self.syn = brainstate.nn.GABAa(num, alpha=12 / (u.ms * u.mM), beta=0.1 / u.ms)
        self.syn = Synapse(num)
        self.proj = brainpy.CurrentProj(
            self.syn.prefetch('g'),
            comm=brainstate.nn.AllToAll(
                self.neu.varshape, self.neu.varshape, include_self=False, w_init=0.1 * u.msiemens / num
            ),
            out=brainpy.COBA(E=-75. * u.mV),
            post=self.neu
        )

    def update(self, t):
        with brainstate.environ.context(t=t):
            self.proj()
            self.syn(self.neu(I_inp))
            # visualize spikes and membrane potentials of the first 5 neurons
            return self.neu.spike.value, self.neu.V.value[:5]


# background input
I_inp = 1.0 * u.uA

# network
net = GammaNet()
brainstate.nn.init_all_states(net)

# simulation
with brainstate.environ.context(dt=0.01 * u.ms):
    times = u.math.arange(0. * u.ms, 500. * u.ms, brainstate.environ.get_dt())
    spikes, vs = brainstate.transform.for_loop(net.update, times, pbar=brainstate.transform.ProgressBar(10))

# visualization
fig, gs = braintools.visualize.get_figure(1, 2, 4, 4)
fig.add_subplot(gs[0, 0])
plt.plot(times, vs.to_decimal(u.mV))
plt.xlabel('Time (ms)')
plt.ylabel('Membrane potential (mV)')

fig.add_subplot(gs[0, 1])
t_indices, n_indices = u.math.where(spikes)
plt.plot(times[t_indices], n_indices, 'k.')
plt.xlabel('Time (ms)')
plt.ylabel('Neuron index')
plt.show()
