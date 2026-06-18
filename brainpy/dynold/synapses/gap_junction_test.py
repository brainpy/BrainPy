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
from absl.testing import parameterized

import brainpy as bp
import brainpy.math as bm
from brainpy.dynold.synapses import gap_junction


class Test_gap_junction(parameterized.TestCase):
    def test_gap_junction(self):
        bm.random.seed()
        neu = bp.neurons.HH(2, V_initializer=bp.init.Constant(-70.68))
        syn = gap_junction.GapJunction(neu, neu, conn=bp.connect.All2All(include_self=False))
        net = bp.Network(syn=syn, neu=neu)

        # 运行模拟
        runner = bp.DSRunner(net,
                             monitors=['neu.V'],
                             inputs=('neu.input', 35.))
        runner(10.)
        self.assertTupleEqual(runner.mon['neu.V'].shape, (100, 2))
