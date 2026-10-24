# Copyright 2025 BrainX Ecosystem Limited. All Rights Reserved.
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


import brainevent
import matplotlib.pyplot as plt
import numpy as np
from jax import vmap
from scipy.io import loadmat

import brainpy as bp
import brainpy.math as bm
from brainpy import neurons


class MultiAreaNet(bp.Network):
    def __init__(
        self, hier, conn, delay_mat, muIE=0.0475, muEE=.0375, wII=.075,
        wEE=.01, wIE=.075, wEI=.0375, extE=15.4, extI=14.0, alpha=4., seed=None,
    ):
        super(MultiAreaNet, self).__init__()

        # data
        self.hier = hier
        self.conn = conn
        self.delay_mat = delay_mat

        # parameters
        self.muIE = muIE
        self.muEE = muEE
        self.wII = wII
        self.wEE = wEE
        self.wIE = wIE
        self.wEI = wEI
        self.extE = extE
        self.extI = extI
        self.alpha = alpha
        num_area = hier.size
        self.num_area = num_area

        # neuron models
        self.E = neurons.LIF((num_area, 1600),
                             V_th=-50., V_reset=-60.,
                             V_rest=-70., tau=20., tau_ref=2.,
                             noise=3. / bm.sqrt(20.),
                             V_initializer=bp.init.Uniform(-70., -50.),
                             method='exp_auto',
                             keep_size=True,
                             ref_var=True)
        self.I = neurons.LIF((num_area, 400), V_th=-50., V_reset=-60.,
                             V_rest=-70., tau=10., tau_ref=2., noise=3. / bm.sqrt(10.),
                             V_initializer=bp.init.Uniform(-70., -50.),
                             method='exp_auto',
                             keep_size=True,
                             ref_var=True)

        # delays
        self.intra_delay_step = int(2. / bm.get_dt())
        self.E_delay_steps = bm.asarray(delay_mat.T / bm.get_dt(), dtype=int)
        bm.fill_diagonal(self.E_delay_steps, self.intra_delay_step)
        self.Edelay = bm.LengthDelay(self.E.spike, delay_len=int(self.E_delay_steps.max()))
        self.Idelay = bm.LengthDelay(self.I.spike, delay_len=self.intra_delay_step)

        # synapses from I
        self.intra_I2E_w = -wEI
        self.intra_I2I_w = -wII
        self.intra_I2I_seed = np.random.randint(0, 10000, num_area)
        self.intra_I2E_seed = np.random.randint(0, 10000, num_area)

        # synapses from E
        self.E2E_seed = np.random.randint(0, 10000, (num_area, num_area))
        self.E2I_seed = np.random.randint(0, 10000, (num_area, num_area))
        self.E2E_weights = (1 + alpha * hier) * muEE * conn.T  # inter-area connections
        # intra-area connections
        self.E2E_weights = bm.fill_diagonal(self.E2E_weights, (1 + alpha * hier) * wEE, inplace=False)
        self.E2I_weights = (1 + alpha * hier) * muIE * conn.T  # inter-area connections
        # intra-area connections
        self.E2I_weights = bm.fill_diagonal(self.E2I_weights, (1 + alpha * hier) * wIE, inplace=False)

    def update(self, v1_input):
        self.E.input[0] += v1_input
        self.E.input += self.extE
        self.I.input += self.extI
        E_not_ref = bm.logical_not(self.E.refractory)
        I_not_ref = bm.logical_not(self.I.refractory)

        # synapses from E
        f_E2E = vmap(lambda spk, weight, seed: brainevent.BinaryArray(spk) @ brainevent.JITCScalarC((weight, 0.1, seed), shape=(1600, 1600)))
        f_E2I = vmap(lambda spk, weight, seed: brainevent.BinaryArray(spk) @ brainevent.JITCScalarC((weight, 0.1, seed), shape=(1600, 400)))
        for i in range(self.num_area):
            delayed_E_spikes = self.Edelay(self.E_delay_steps[i], i)
            self.E.V += f_E2E(delayed_E_spikes, self.E2E_weights[i], self.E2E_seed[i]) * E_not_ref  # E2E
            self.I.V += f_E2I(delayed_E_spikes, self.E2I_weights[i], self.E2I_seed[i]) * I_not_ref  # E2I

        # synapses from I
        delayed_I_spikes = self.Idelay(self.intra_delay_step)
        f_I2E = vmap(lambda spk, seed: brainevent.BinaryArray(spk) @ brainevent.JITCScalarC((self.intra_I2E_w, 0.1, seed), shape=(400, 1600)))
        self.E.V += f_I2E(delayed_I_spikes, self.intra_I2E_seed) * E_not_ref  # I2E
        f_I2I = vmap(lambda spk, seed: brainevent.BinaryArray(spk) @ brainevent.JITCScalarC((self.intra_I2I_w, 0.1, seed), shape=(400, 400)))
        self.I.V += f_I2I(delayed_I_spikes, self.intra_I2I_seed) * I_not_ref  # I2I

        # updates
        self.Edelay.update(self.E.spike)
        self.Idelay.update(self.I.spike)
        self.E.update()
        self.I.update()


def raster_plot(xValues, yValues, duration):
    ticks = np.round(np.arange(0, 29) + 0.5, 2)
    areas = ['V1', 'V2', 'V4', 'DP', 'MT', '8m', '5', '8l', 'TEO', '2', 'F1',
             'STPc', '7A', '46d', '10', '9/46v', '9/46d', 'F5', 'TEpd', 'PBr',
             '7m', '7B', 'F2', 'STPi', 'PROm', 'F7', '8B', 'STPr', '24c']
    N = len(ticks)
    plt.figure(figsize=(8, 6))
    plt.plot(xValues, yValues / (4 * 400), '.', markersize=1)
    plt.plot([0, duration], np.arange(N + 1).repeat(2).reshape(-1, 2).T, 'k-')
    plt.ylabel('Area')
    plt.yticks(np.arange(N))
    plt.xlabel('Time [ms]')
    plt.ylim(0, N)
    plt.yticks(ticks, areas)
    plt.xlim(0, duration)
    plt.tight_layout()
    plt.show()


# hierarchy values
hierVals = loadmat('../Joglekar_2018_data/hierValspython.mat')
hierValsnew = hierVals['hierVals'].flatten()
hier = bm.asarray(hierValsnew / max(hierValsnew))  # hierarchy normalized.

# fraction of labeled neurons
flnMatp = loadmat('../Joglekar_2018_data/efelenMatpython.mat')
conn = bm.asarray(flnMatp['flnMatpython'].squeeze())  # fln values..Cij is strength from j to i

# Distance
speed = 3.5  # axonal conduction velocity
distMatp = loadmat('../Joglekar_2018_data/subgraphWiring29.mat')
distMat = distMatp['wiring'].squeeze()  # distances between areas values..
delayMat = bm.asarray(distMat / speed)

pars = dict(extE=14.2, extI=14.7, wII=.075, wEE=.01, wIE=.075, wEI=.0375, muEE=.0375, muIE=0.0475)
inps = dict(value=15, duration=150)

inputs, length = bp.inputs.section_input(values=[0, inps['value'], 0.],
                                         durations=[300., inps['duration'], 500],
                                         return_length=True)

net = MultiAreaNet(hier, conn, delayMat, **pars)
runner = bp.DSRunner(net, monitors={'E.spike': lambda: net.E.spike.flatten()})
runner.run(inputs=inputs)

times, indices = np.where(runner.mon['E.spike'])
times = runner.mon.ts[times]
raster_plot(times, indices, length)
