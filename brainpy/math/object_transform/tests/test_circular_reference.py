# -*- coding: utf-8 -*-
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
from pprint import pprint

import brainpy as bp


class HH(bp.dyn.NeuDyn):
    def __init__(self, size, ENa=55., EK=-90., EL=-65, C=1.0,
                 gNa=35., gK=9., gL=0.1, V_th=20., phi=5.0, **kwargs):
        super(HH, self).__init__(size=size, **kwargs)

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

        # variables
        self.V = bp.math.ones(self.num) * -65.
        self.h = bp.math.ones(self.num) * 0.6
        self.n = bp.math.ones(self.num) * 0.32
        self.inputs = bp.math.zeros(self.num)
        self.spikes = bp.math.zeros(self.num, dtype=bp.math.bool_)

        self.pre = None

    @bp.odeint
    def integral(self, V, h, n, t, Iext):
        alpha = 0.07 * bp.math.exp(-(V + 58) / 20)
        beta = 1 / (bp.math.exp(-0.1 * (V + 28)) + 1)
        dhdt = self.phi * (alpha * (1 - h) - beta * h)

        alpha = -0.01 * (V + 34) / (bp.math.exp(-0.1 * (V + 34)) - 1)
        beta = 0.125 * bp.math.exp(-(V + 44) / 80)
        dndt = self.phi * (alpha * (1 - n) - beta * n)

        m_alpha = -0.1 * (V + 35) / (bp.math.exp(-0.1 * (V + 35)) - 1)
        m_beta = 4 * bp.math.exp(-(V + 60) / 18)
        m = m_alpha / (m_alpha + m_beta)
        INa = self.gNa * m ** 3 * h * (V - self.ENa)
        IK = self.gK * n ** 4 * (V - self.EK)
        IL = self.gL * (V - self.EL)
        dVdt = (- INa - IK - IL + Iext) / self.C

        return dVdt, dhdt, dndt

    def update(self, t, dt):
        V, h, n = self.integral(self.V, self.h, self.n, t, self.inputs)
        self.spikes[:] = bp.math.logical_and(self.V < self.V_th, V >= self.V_th)
        self.V[:] = V
        self.h[:] = h
        self.n[:] = n
        self.inputs[:] = 0.


def test_nodes():
    A = HH(1, name='X')
    B = HH(1, name='Y')
    A.pre = B
    B.pre = A

    net = bp.Network(A, B)
    abs_nodes = net.nodes(method='absolute')
    rel_nodes = net.nodes(method='relative')
    print()
    pprint(abs_nodes)
    pprint(rel_nodes)

    assert len(abs_nodes) == 3
    assert len(rel_nodes) == 5
